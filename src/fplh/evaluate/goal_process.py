"""G4–G6 ladder and the team-level §8.5 checks (ARCHITECTURE.md §7.5, §8.5, §11.6–11.7).

* Fit G4, G5 and G6 on matches observed before the tuning seasons (offsets are M1's
  pre-update expected goals), build their emulators (cached in gold by parameter hash).
* Ladder: scoreline-grid log loss of M1 deadline rates turned into grids by G0 (Poisson),
  G1 (Dixon–Coles) and each G-process emulator; paired gameweek-block comparisons.
* §8.5 checks: (1) emulator vs direct simulation at random mean-goal pairs; (2) market
  reproduction: invert de-vigged deadline prices under the emulator and simulate;
  (3) calibration of the draw rate and the scoreline cells up to 4–4; (4) in-play
  next-goal calibration from the actual state at minutes 15–75.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from fplh.evaluate import metrics as m
from fplh.evaluate.bootstrap import compare
from fplh.evaluate.phase2 import fit_goal_models, load_m1_params, training_predictions
from fplh.evaluate.team_level import GRID_COLS, M1Predictor, fixture_losses, grid_frame
from fplh.evaluate.walk_forward import cached_walk_forward
from fplh.features.information_set import InformationSet, SilverStore
from fplh.features.spine import historical_deadlines
from fplh.lake.storage import Lake
from fplh.models.baselines import market_probabilities
from fplh.models.goal_benchmarks import GoalModel, markets
from fplh.models.goal_process import (
    GoalProcessParams,
    build_cells,
    dispersion,
    estimate_stoppage,
    fit_goal_process,
)
from fplh.models.market import load_devig_method
from fplh.sim.emulator import Emulator, params_sha
from fplh.sim.team import StartState, simulate_team

LEVELS = ("G4", "G5", "G6")
CHECKPOINTS = (15, 30, 45, 60, 75)


def training_matches(
    store: SilverStore, before: pd.Timestamp
) -> tuple[pd.DataFrame, InformationSet]:
    """Matches observed before ``before`` with M1's pre-update expected goals."""
    hist = training_predictions(store, before, load_m1_params())
    info = InformationSet.at(before, store)
    us = info.table("us_match")[["season", "home_team", "away_team", "understat_match_id"]]
    matches = hist.merge(us, on=["season", "home_team", "away_team"]).rename(
        columns={"pred_home": "mu_home", "pred_away": "mu_away"}
    )
    return matches.reset_index(drop=True), info


def fit_levels(
    store: SilverStore, before: pd.Timestamp
) -> tuple[dict[str, GoalProcessParams], pd.DataFrame]:
    matches, info = training_matches(store, before)
    shots, events = info.table("fact_shot"), info.table("fact_match_event")
    recent = sorted(shots["season"].unique())[-2:]  # stoppage time drifts by era
    stoppage = estimate_stoppage(shots[shots["season"].isin(recent)])
    cells = build_cells(matches, events, shots, stoppage)
    tag = before.isoformat()
    levels = {lv: fit_goal_process(cells, lv, stoppage, tag) for lv in LEVELS}  # type: ignore[arg-type]
    return levels, dispersion(cells, matches["season"].to_numpy())


def emulator_for(lake: Lake, params: GoalProcessParams, **build: int) -> Emulator:
    settings = "_".join(f"{k}{v}" for k, v in sorted(build.items()))
    key = f"gold/emulator/{params_sha(params)}_{settings}.npz"
    if lake.exists(key):
        return Emulator.from_bytes(lake.get_bytes(key))
    emu = Emulator.build(params, **build)
    lake.put_bytes(key, emu.to_bytes(), overwrite=True)
    return emu


@dataclass
class DevigPrices:
    """Fixture predictor: de-vigged pre-match prices observable at the deadline."""

    method: str = "power"
    name: str = "devig_prices"
    version: str = "1"

    def predict_fixtures(self, info: InformationSet, fixtures: pd.DataFrame) -> pd.DataFrame:
        mp = market_probabilities(info, fixtures[["fixture_uid"]], self.method)
        out: pd.DataFrame = fixtures[["fixture_uid", "deadline_at", "horizon"]].merge(
            mp[["fixture_uid", "p_home", "p_draw", "p_away", "p_over25", "market_available"]],
            on="fixture_uid",
        )
        return out


@dataclass
class GoalProcessResult:
    seasons: list[str]
    summary: pd.DataFrame
    comparisons: pd.DataFrame
    dispersion: pd.DataFrame
    checks: dict[str, object] = field(default_factory=dict)
    params: dict[str, GoalProcessParams] = field(default_factory=dict)
    chosen: str = "G0"


def emulator_accuracy(
    emu: Emulator, n_points: int = 60, n_sims: int = 200_000, seed: int = 7
) -> pd.DataFrame:
    """Emulator grid vs a fresh direct simulation at random off-grid mean-goal pairs."""
    rng = np.random.default_rng(seed)
    mu = np.exp(rng.uniform(np.log(0.3), np.log(3.5), size=(n_points, 2)))
    xh, xa = emu.nominal_for_means(mu[:, 0], mu[:, 1])
    rows = []
    for start in range(0, n_points, 10):
        sl = slice(start, start + 10)
        nominal = np.column_stack([np.exp(xh[sl]), np.exp(xa[sl])])
        sim = simulate_team(emu.params, nominal, n_sims, seed + 1 + start)
        for i, (h, a) in enumerate(zip(sim.home, sim.away, strict=True)):
            k = start + i
            g = emu.grid_at_nominal(float(xh[k]), float(xa[k]))
            mk = markets(g)
            rows.append(
                {
                    "mu_home": mu[k, 0],
                    "mu_away": mu[k, 1],
                    "d_mean_home": float(h.mean() - mu[k, 0]),
                    "d_mean_away": float(a.mean() - mu[k, 1]),
                    "d_home": float((h > a).mean() - mk["home"]),
                    "d_draw": float((h == a).mean() - mk["draw"]),
                    "d_away": float((h < a).mean() - mk["away"]),
                    "d_over": float((h + a > 2.5).mean() - mk["over"]),
                }
            )
    return pd.DataFrame(rows)


def market_reproduction(models: dict[str, GoalModel], prices: pd.DataFrame) -> pd.DataFrame:
    """§8.5.1: invert each fixture's de-vigged 1X2 + O/U 2.5 under each goal model (two
    rates, four prices) and measure how closely the implied grid reproduces the prices.
    The emulator's own accuracy against direct simulation is ``emulator_accuracy``."""
    probs = prices[["p_home", "p_draw", "p_away", "p_over25"]].to_numpy(dtype=float)
    rows = []
    for name, model in models.items():
        for uid, row in zip(prices["fixture_uid"], probs, strict=True):
            over = None if np.isnan(row[3]) else float(row[3])
            mk = markets(model.grid(*model.invert(list(row[:3]), over)))
            got = np.array([mk["home"], mk["draw"], mk["away"], mk["over"]])
            err = np.abs(got - row)
            rows.append(
                {
                    "model": name,
                    "fixture_uid": uid,
                    "max_abs": float(np.nanmax(err)),
                    "mean_abs": float(np.nanmean(err)),
                }
            )
    return pd.DataFrame(rows)


def in_play_calibration(
    params: GoalProcessParams,
    emu: Emulator,
    rates: pd.DataFrame,
    events: pd.DataFrame,
    n_sims: int = 2000,
) -> pd.DataFrame:
    """P(next goal home / away / none) from the actual state at each checkpoint vs the
    actual next goal. ``rates``: fixture_uid, lam_mod_home, lam_mod_away."""
    goals = events[events["kind"].isin(["goal", "own_goal"])]
    reds = events[events["kind"] == "red"]
    xh, xa = emu.nominal_for_means(rates["lam_mod_home"], rates["lam_mod_away"])
    nominal = np.column_stack([np.exp(xh), np.exp(xa)])
    uids = rates["fixture_uid"].to_numpy()
    out = []
    for t0 in CHECKPOINTS:

        def count(df: pd.DataFrame, side: str, t0: int = t0) -> np.ndarray:
            c = df[(df["minute"] < t0) & (df["side"] == side)].groupby("fixture_uid").size()
            return c.reindex(uids, fill_value=0).to_numpy(dtype=np.int64)

        after = goals[goals["minute"] >= t0].sort_values(["fixture_uid", "minute", "seq"])
        first = after.drop_duplicates("fixture_uid").set_index("fixture_uid")["side"]
        actual = pd.Series(uids).map(first).map({"h": 0, "a": 1}).fillna(2).to_numpy(int)
        start = StartState(
            np.full(len(uids), t0, dtype=np.int64),
            count(goals, "h"),
            count(goals, "a"),
            count(reds, "h"),
            count(reds, "a"),
        )
        sim = simulate_team(params, nominal, n_sims, seed=100 + t0, start=start)
        probs = np.stack([(sim.next_goal == c).mean(axis=1) for c in (0, 1)], axis=1)
        probs = np.column_stack([probs, 1 - probs.sum(axis=1)])
        out.append(
            pd.DataFrame(
                {
                    "fixture_uid": uids,
                    "checkpoint": t0,
                    "p_home": probs[:, 0],
                    "p_away": probs[:, 1],
                    "p_none": probs[:, 2],
                    "actual": actual,
                }
            )
        )
    return pd.concat(out, ignore_index=True)


def _calibration(losses: pd.DataFrame) -> dict[str, object]:
    """Draw rate and scoreline cells up to 4–4: observed − predicted with block CIs."""
    block = losses["season"].astype(str) + ":" + losses["round"].astype(int).astype(str)
    zero = pd.Series(np.zeros(len(losses)), index=losses.index)
    draw = (losses["home_goals"] == losses["away_goals"]).astype(float) - losses["p_draw"]
    c = compare(draw, zero, block, n_boot=1000)
    inside = 0
    for h in range(5):
        for a in range(5):
            hit = ((losses["home_goals"] == h) & (losses["away_goals"] == a)).astype(float)
            cc = compare(hit - losses[f"g_{h}_{a}"], zero, block, n_boot=500)
            inside += cc.ci_low <= 0 <= cc.ci_high
    return {
        "draw_obs_minus_pred": c.mean_diff,
        "draw_ci": (c.ci_low, c.ci_high),
        "cells_within_ci": f"{inside}/25",
    }


def evaluate_goal_process(
    lake: Lake, seasons: list[str], *, n_sims_emulator: int = 100_000
) -> GoalProcessResult:
    store = SilverStore(lake)
    dim = store.get("dim_fixture")
    deadlines = [d for s in seasons for d in historical_deadlines(dim, s)["deadline_at"]]
    before = min(deadlines)
    levels, disp = fit_levels(store, before)
    emus = {lv: emulator_for(lake, p, grid=20, n_sims=n_sims_emulator) for lv, p in levels.items()}

    # M1 deadline rates (cached run) and the benchmark grids
    params = load_m1_params()
    m1 = cached_walk_forward(
        store, M1Predictor(params, name="m1_tuned"), deadlines, lake=lake, unit="fixture"
    )
    rates = m1.predictions[
        ["fixture_uid", "deadline_at", "horizon", "lam_mod_home", "lam_mod_away"]
    ]
    train = training_predictions(store, before, params)
    g01 = fit_goal_models(train)
    models: dict[str, GoalModel] = {"G0": g01["G0"], "G1": g01["G1"]}
    models.update({lv: e.as_goal_model() for lv, e in emus.items()})
    outcomes = dim[["fixture_uid", "home_goals", "away_goals", "round", "season"]].dropna(
        subset=["home_goals"]
    )
    losses = {}
    for name, gm in models.items():
        grid = grid_frame(rates["lam_mod_home"].to_numpy(), rates["lam_mod_away"].to_numpy(), gm)
        pred = pd.concat([rates.reset_index(drop=True), grid], axis=1)
        losses[name] = (
            fixture_losses(pred, outcomes).sort_values("fixture_uid").reset_index(drop=True)
        )
    rows = []
    for name, lo in losses.items():
        rows.append(
            {
                "model": name,
                "grid_log_loss": float(lo["loss_grid"].mean()),
                "log_loss": float(lo["loss_log"].mean()),
                "rps": float(lo["loss_rps"].mean()),
                "n": len(lo),
            }
        )
    summary = pd.DataFrame(rows)
    comps = []
    for a, b in (("G4", "G0"), ("G5", "G4"), ("G6", "G5"), ("G4", "G1")):
        la, lb = losses[a], losses[b]
        block = la["season"].astype(str) + ":" + la["round"].astype(int).astype(str)
        for metric in ("loss_grid", "loss_rps", "loss_log"):
            c = compare(la[metric], lb[metric], block, n_boot=2000)
            comps.append(
                {
                    "a": a,
                    "b": b,
                    "metric": metric.removeprefix("loss_"),
                    "mean_diff": c.mean_diff,
                    "ci_low": c.ci_low,
                    "ci_high": c.ci_high,
                    "dm_p": c.dm_pvalue,
                }
            )
    comparisons = pd.DataFrame(comps)

    # Ladder rule: the simulator needs in-match dynamics, so it uses the highest level not
    # significantly worse than its predecessor on scoreline log loss; G0 stays the
    # default pre-match grid unless a level is significantly better than it.
    grid_c = comparisons[comparisons["metric"] == "grid"]
    ci_low = {
        (str(a), str(b)): float(v)
        for a, b, v in zip(grid_c["a"], grid_c["b"], grid_c["ci_low"], strict=True)
    }
    chosen = "G4"
    for a, b in (("G5", "G4"), ("G6", "G5")):
        if ci_low[(a, b)] <= 0:
            chosen = a
        else:
            break

    checks: dict[str, object] = {}
    acc = emulator_accuracy(emus[chosen])
    checks["emulator_max_abs_prob_err"] = float(
        acc[["d_home", "d_draw", "d_away", "d_over"]].abs().max().max()
    )
    checks["emulator_mean_abs_prob_err"] = float(
        acc[["d_home", "d_draw", "d_away", "d_over"]].abs().mean().mean()
    )
    checks["emulator_max_abs_mean_err"] = float(
        acc[["d_mean_home", "d_mean_away"]].abs().max().max()
    )

    prices = cached_walk_forward(
        store, DevigPrices(load_devig_method()), deadlines, lake=lake, unit="fixture"
    ).predictions
    prices = prices[prices["market_available"].astype(bool)].dropna(subset=["p_home"])
    rep = market_reproduction({chosen: models[chosen], "G1": models["G1"]}, prices)
    mean_abs = rep.groupby("model")["mean_abs"].mean().to_dict()
    max_abs = rep.groupby("model")["max_abs"].mean().to_dict()
    checks[f"market_residual_mean_abs_{chosen}"] = float(mean_abs[chosen])
    checks["market_residual_mean_abs_G1"] = float(mean_abs["G1"])
    checks[f"market_residual_max_abs_{chosen}"] = float(max_abs[chosen])
    checks["n_market_fixtures"] = int(rep["fixture_uid"].nunique())
    checks["emulator_reference_se"] = float(np.sqrt(0.25 / 200_000))

    best = losses[chosen].merge(rates[["fixture_uid"]], on="fixture_uid")
    checks.update(_calibration(best))

    events = store.get("fact_match_event")
    events = events[events["fixture_uid"].isin(set(rates["fixture_uid"]))]
    played = rates[rates["fixture_uid"].isin(set(events["fixture_uid"]))].reset_index(drop=True)
    inplay = in_play_calibration(levels[chosen], emus[chosen], played, events)
    for code, name in enumerate(("home", "away", "none")):
        checks[f"inplay_ece_{name}"] = m.ece(
            inplay[f"p_{name}"].to_numpy(), (inplay["actual"] == code).to_numpy()
        )
    checks["inplay_n"] = len(inplay)
    return GoalProcessResult(seasons, summary, comparisons, disp, checks, levels, chosen)


__all__ = ["GRID_COLS", "evaluate_goal_process"]
