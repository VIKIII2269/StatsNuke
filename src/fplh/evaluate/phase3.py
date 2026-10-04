"""Phase 3 exit gate (IMPLEMENTATION_PLAN §4, ARCHITECTURE.md §11.3–11.6).

The player simulator runs walk-forward at horizon 1 over the tuning seasons and is scored
on the same player-fixtures as the benchmarks (the OpenFPL re-implementation, A0 and the
last-5 average; ``evaluate/benchmarks.py``):

* **gate:** MSE(simulator) − MSE(benchmark) has a 95 % gameweek-block CI upper bound below
  0 for every benchmark, and the point estimate is negative in at least 2 of 3 seasons;
* **guardrails:** MAE, Spearman within position (all rows and rows where the player
  played), top-10 precision, CRPS of the points pmf, ECE of P(60+) and P(haul ≥ 10);
* **ablations:** A4 (naive minutes) and A5 (raw goal rates), each a full walk-forward.

Team rates are the Phase 2 fused rates (M1 + de-vigged closing prices, fusion fitted on
the five seasons before the first deadline), walk-forward and cached.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from fplh.evaluate import metrics as m
from fplh.evaluate.goal_process import emulator_for
from fplh.evaluate.phase2 import (
    _fit_fusion,
    fit_goal_models,
    load_m1_params,
    training_predictions,
)
from fplh.evaluate.player_level import align, compare_runs, outcomes, player_losses, summary
from fplh.evaluate.team_level import FusedPredictor, M1Predictor
from fplh.evaluate.walk_forward import cached_walk_forward
from fplh.features.information_set import InformationSet, SilverStore
from fplh.features.spine import historical_deadlines
from fplh.lake.storage import Lake
from fplh.models.baselines import A0Player, NaiveLast5
from fplh.models.goal_process import GoalProcessParams
from fplh.models.market import load_devig_method
from fplh.models.openfpl import OpenFPLReplica
from fplh.models.player_sim import PMF_RANGE, PlayerSimulator
from fplh.rules.config import load_rules
from fplh.settings import get_settings

BENCHMARKS = ("openfpl_replica", "a0", "naive_last5")


@dataclass
class FusedRates:
    """Fixture predictor: the Phase 2 fused rates, the fusion fitted lazily (a cached
    walk-forward run never needs it)."""

    store: SilverStore = field(repr=False)
    before: pd.Timestamp
    name: str = "m3_fused_rates"
    version: str = "1"
    _inner: FusedPredictor | None = field(default=None, repr=False)

    def predict_fixtures(self, info: InformationSet, fixtures: pd.DataFrame) -> pd.DataFrame:
        if self._inner is None:
            params = load_m1_params()
            g1 = fit_goal_models(training_predictions(self.store, self.before, params))["G1"]
            method = load_devig_method()
            fusion = _fit_fusion(self.store, params, g1, self.before, method).fusion
            self._inner = FusedPredictor(M1Predictor(params, g1), fusion, devig_method=method)
        return self._inner.predict_fixtures(info, fixtures)


def simulator_params() -> GoalProcessParams:
    import yaml

    doc = yaml.safe_load((get_settings().configs_dir / "models" / "goal_process.yaml").read_text())
    return GoalProcessParams.from_dict(doc["levels"][doc["simulator_level"]])


def team_rates(
    lake: Lake, store: SilverStore, deadlines: list[pd.Timestamp], horizon: int = 1
) -> pd.DataFrame:
    run = cached_walk_forward(
        store,
        FusedRates(store, min(deadlines)),
        deadlines,
        lake=lake,
        unit="fixture",
        horizon=horizon,
    )
    p = run.predictions
    out: pd.DataFrame = p[["fixture_uid", "deadline_at"]].assign(
        mu_home=p["lambda_home"].to_numpy(float), mu_away=p["lambda_away"].to_numpy(float)
    )
    return out


def simulator(
    lake: Lake,
    store: SilverStore,
    deadlines: list[pd.Timestamp],
    n_sims: int = 2000,
    horizon: int = 1,
    **variant: str,
) -> PlayerSimulator:
    """The simulator over the fused rates. Runs are cached by name, so the name carries any
    ablation, a horizon beyond 1 and a non-default number of simulations."""
    params = simulator_params()
    emu = emulator_for(lake, params, grid=20, n_sims=100_000)
    suffix = "".join(f"_{k}-{v}" for k, v in sorted(variant.items()))
    if horizon != 1:
        suffix += f"_h{horizon}"
    if n_sims != 2000:
        suffix += f"_n{n_sims}"
    return PlayerSimulator(
        team_rates(lake, store, deadlines, horizon),
        emu,
        params,
        load_m1_params(),
        n_sims=n_sims,
        name=f"player_sim{suffix}",
        **variant,  # type: ignore[arg-type]
    )


def run(
    lake: Lake,
    store: SilverStore,
    predictor: object,
    deadlines: list[pd.Timestamp],
    horizon: int = 1,
) -> pd.DataFrame:
    """One cached walk-forward over all deadlines (stateful refit cadences stay intact)."""
    return cached_walk_forward(
        store,
        predictor,  # type: ignore[arg-type]
        deadlines,
        lake=lake,
        horizon=horizon,
    ).predictions


def crps_rows(pred: pd.DataFrame, actual: pd.Series) -> np.ndarray:
    lo, hi = PMF_RANGE
    pmf = pred[[f"pmf_{k}" for k in range(lo, hi + 1)]].to_numpy(float)
    cdf = np.cumsum(pmf, axis=1)
    ks = np.arange(lo, hi + 1)
    y = np.clip(actual.to_numpy(), lo, hi)
    step = (ks[None, :] >= y[:, None]).astype(float)
    out: np.ndarray = ((cdf - step) ** 2).sum(axis=1)
    return out


@dataclass
class Phase3Result:
    seasons: list[str]
    summary: pd.DataFrame
    gate: pd.DataFrame
    per_season: pd.DataFrame
    guardrails: dict[str, float]
    ablations: pd.DataFrame
    passed: bool
    losses: dict[str, pd.DataFrame] = field(default_factory=dict)


def evaluate_phase3(
    lake: Lake, seasons: list[str], *, n_sims: int = 2000, ablations: bool = True
) -> Phase3Result:
    store = SilverStore(lake)
    dim = store.get("dim_fixture")
    by_season = {s: list(historical_deadlines(dim, s)["deadline_at"]) for s in seasons}
    all_deadlines = [d for ds in by_season.values() for d in ds]

    preds: dict[str, pd.DataFrame] = {
        "a0": pd.concat(  # rules differ by season
            [run(lake, store, A0Player(load_rules(s)), ds) for s, ds in by_season.items()],
            ignore_index=True,
        ),
        "naive_last5": run(lake, store, NaiveLast5(), all_deadlines),
        "openfpl_replica": run(lake, store, OpenFPLReplica(), all_deadlines),
        "simulator": run(lake, store, simulator(lake, store, all_deadlines, n_sims), all_deadlines),
    }
    variants = {"A4 naive minutes": {"minutes": "naive"}, "A5 raw goal rates": {"attack": "raw"}}
    if ablations:
        for label, v in variants.items():
            ab = simulator(lake, store, all_deadlines, n_sims, 1, **v)
            preds[label] = run(lake, store, ab, all_deadlines)

    actual = outcomes(store)
    losses = align({k: player_losses(v, actual) for k, v in preds.items()})
    table = pd.DataFrame([{"model": k, **summary(v)} for k, v in losses.items()]).sort_values("mse")

    gate_rows, season_rows = [], []
    for b in BENCHMARKS:
        c = compare_runs(losses["simulator"], losses[b], "se")
        gate_rows.append({"benchmark": b, "metric": "MSE", **c})
        for season in seasons:
            a = losses["simulator"][losses["simulator"]["season"] == season]
            bb = losses[b][losses[b]["season"] == season]
            season_rows.append(
                {"benchmark": b, "season": season, "diff": float(a["se"].mean() - bb["se"].mean())}
            )
    gate = pd.DataFrame(gate_rows)
    per_season = pd.DataFrame(season_rows)
    wins = per_season.assign(win=per_season["diff"] < 0).groupby("benchmark")["win"].sum()
    passed = bool((gate["ci_high"] < 0).all() and (wins >= min(2, len(seasons))).all())

    # guardrails that need the simulator's distribution
    s = losses["simulator"].merge(
        preds["simulator"].drop(columns=["position", "expected_points"]),
        on=["player_uid", "fixture_uid", "deadline_at"],
    )
    guard = {
        "crps": float(crps_rows(s, s["total_points"]).mean()),
        "ece_p60": m.ece(s["p_60"].to_numpy(), (s["minutes"] >= 60).to_numpy()),
        "ece_haul": m.ece(s["p_haul"].to_numpy(), (s["total_points"] >= 10).to_numpy()),
        "mean_p60": float(s["p_60"].mean()),
        "rate_60": float((s["minutes"] >= 60).mean()),
        "mean_p_haul": float(s["p_haul"].mean()),
        "rate_haul": float((s["total_points"] >= 10).mean()),
        "mean_expected_points": float(s["expected_points"].mean()),
        "mean_points": float(s["total_points"].mean()),
    }
    abl = pd.DataFrame(
        [
            {"ablation": label, **compare_runs(losses["simulator"], losses[label], "se")}
            for label in variants
            if label in losses
        ]
    )
    return Phase3Result(
        seasons, table.reset_index(drop=True), gate, per_season, guard, abl, passed, losses
    )
