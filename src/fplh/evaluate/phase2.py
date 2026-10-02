"""Phase 2 evaluation (IMPLEMENTATION_PLAN §3): G0–G3 ladder, A1–A3 ablations and the
exit gate (fused ≥ market-only on RPS), all walk-forward over the tuning seasons.

Dependence parameters (G1–G3) and fusion weights are fitted only on the training
period (seasons before the first evaluated one) using M1's pre-update predictions, so
no evaluated outcome informs any fitted parameter.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
import yaml

from fplh.evaluate.bootstrap import compare
from fplh.evaluate.metrics import outcome_index
from fplh.evaluate.team_level import (
    FusedPredictor,
    M1Predictor,
    MarketPredictor,
    championship,
    fixture_losses,
    grid_frame,
    market_features,
    observed_matches,
    season_teams,
    summarise,
)
from fplh.evaluate.tracking import log_metrics
from fplh.evaluate.walk_forward import run_walk_forward
from fplh.features.information_set import InformationSet, SilverStore
from fplh.features.spine import historical_deadlines, target_fixtures
from fplh.lake.storage import Lake
from fplh.models.fusion import Fusion, FusionFit, fit_fusion_cv
from fplh.models.goal_benchmarks import GoalModel, benchmarks
from fplh.models.market import DEVIG as DEVIG_METHODS
from fplh.models.market import devig_calibration, load_devig_method
from fplh.models.team_strength import TeamStrengthParams, run_filter
from fplh.settings import get_settings


def load_m1_params(path: Path | None = None) -> TeamStrengthParams:
    path = path or get_settings().configs_dir / "models" / "team_strength.yaml"
    cfg = yaml.safe_load(path.read_text())
    return TeamStrengthParams(**cfg["params"])


CALIBRATION_BOOKS = ("pinnacle", "market_avg")


def closing_devig_calibration(
    store: SilverStore, before: pd.Timestamp, division: str = "E0"
) -> pd.DataFrame:
    """Mean 1X2 log loss of each de-vig method on the closing prices of matches observable
    before ``before`` (ARCHITECTURE.md §7.3), per book: closing prices are the sharpest,
    so the method that calibrates them best is the default for pre-match prices."""
    info = InformationSet.at(before, store)
    odds, res = info.table("snap_odds"), info.table("fd_match")
    if odds.empty or res.empty:
        return pd.DataFrame(columns=["bookmaker", "n", *sorted(DEVIG_METHODS)])
    c = odds[
        odds["is_closing"]
        & (odds["market"] == "1x2")
        & (odds["division"] == division)
        & odds["bookmaker"].isin(CALIBRATION_BOOKS)
    ]
    wide = (
        c.pivot_table(
            index=["fixture_uid", "bookmaker"], columns="outcome", values="price", aggfunc="last"
        )
        .dropna(subset=["home", "draw", "away"])
        .reset_index()
    )
    res = res[res["division"] == division].dropna(subset=["home_goals"])
    j = wide.merge(res[["fixture_uid", "home_goals", "away_goals"]], on="fixture_uid")
    rows = []
    for book, g in j.groupby("bookmaker"):
        y = outcome_index(g["home_goals"].astype(int), g["away_goals"].astype(int))
        losses = devig_calibration(g[["home", "draw", "away"]].to_numpy(float), y)
        rows.append({"bookmaker": book, "n": len(g), **losses})
    return pd.DataFrame(rows).sort_values("bookmaker").reset_index(drop=True)


def training_predictions(
    store: SilverStore, before: pd.Timestamp, params: TeamStrengthParams
) -> pd.DataFrame:
    """M1 pre-update predictions for every match observable before ``before``."""
    info = InformationSet.at(before, store)
    f = run_filter(
        observed_matches(info),
        params,
        season_teams(info),
        record=True,
        championship=championship(info),
    )
    hist = pd.DataFrame(f.history)
    out: pd.DataFrame = hist[hist["season"] != hist["season"].min()]  # first season: burn-in
    return out


def fit_goal_models(train: pd.DataFrame) -> dict[str, GoalModel]:
    lh, la = train["pred_home"].to_numpy(), train["pred_away"].to_numpy()
    hg, ag = train["home_goals"].to_numpy(), train["away_goals"].to_numpy()
    return {name: model.fit(lh, la, hg, ag) for name, model in benchmarks().items()}


@dataclass
class Phase2Result:
    seasons: list[str]
    summary: pd.DataFrame
    comparisons: pd.DataFrame
    goal_models: dict[str, dict[str, float]] = field(default_factory=dict)
    fusion: dict[str, object] = field(default_factory=dict)
    run_ids: dict[str, str] = field(default_factory=dict)


def evaluate_phase2(lake: Lake, seasons: list[str], *, log: bool = True) -> Phase2Result:
    store = SilverStore(lake)
    params = load_m1_params()
    method = load_devig_method()
    dim = store.get("dim_fixture")
    deadlines = [d for s in seasons for d in historical_deadlines(dim, s)["deadline_at"]]
    start = min(deadlines)
    train = training_predictions(store, start, params)
    goal_models = fit_goal_models(train)
    outcomes = dim[["fixture_uid", "home_goals", "away_goals", "round", "season"]].dropna(
        subset=["home_goals"]
    )

    runs: dict[str, pd.DataFrame] = {}
    run_ids: dict[str, str] = {}
    # M1 rates once (tuned params), regridded under every goal model (G-series ladder).
    wf = run_walk_forward(
        store,
        M1Predictor(params, goal_models["G0"], name="m1_tuned"),
        deadlines,
        unit="fixture",
        lake=lake,
    )
    run_ids["m1_tuned"] = wf.run_id
    for name, gm in goal_models.items():
        grid = grid_frame(
            wf.predictions["lam_mod_home"].to_numpy(), wf.predictions["lam_mod_away"].to_numpy(), gm
        )
        runs[f"M1+{name}"] = pd.concat(
            [
                wf.predictions[["fixture_uid", "deadline_at", "horizon"]].reset_index(drop=True),
                grid,
            ],
            axis=1,
        )

    # A2: goals only (ω = 0) vs the fitted goals+xG blend.  A3: untuned defaults.
    for label, p in (
        ("M1 goals-only (A2)", params.with_(omega=0.0)),
        ("M1 default params (A3)", TeamStrengthParams()),
    ):
        r = run_walk_forward(
            store,
            M1Predictor(p, goal_models["G0"], name=label.split()[1]),
            deadlines,
            unit="fixture",
            lake=lake,
        )
        runs[label] = r.predictions
        run_ids[label] = r.run_id

    # Market-only and fused (need pre-match odds in silver).
    market = run_walk_forward(
        store,
        MarketPredictor(goal_models["G1"], devig_method=method),
        deadlines,
        unit="fixture",
        lake=lake,
    )
    fusion_info: dict[str, object] = {"available": False}
    if not market.predictions.empty:
        runs["market-only"] = market.predictions
        run_ids["market-only"] = market.run_id
        fitted = _fit_fusion(store, params, goal_models["G1"], start, method)
        fusion = fitted.fusion
        fusion_info = {
            "available": True,
            "devig": method,
            "alpha": tuple(round(float(a), 6) for a in fusion.alpha),
            "bias": tuple(round(float(b), 6) for b in fusion.bias),
            "ridge": fitted.ridge,
            "cv_nll": {r: round(v, 6) for r, v in fitted.cv.items()},
        }
        fused = run_walk_forward(
            store,
            FusedPredictor(M1Predictor(params, goal_models["G1"]), fusion, devig_method=method),
            deadlines,
            unit="fixture",
            lake=lake,
        )
        runs["M3 fused"] = fused.predictions
        run_ids["M3 fused"] = fused.run_id

    losses = {k: fixture_losses(v, outcomes) for k, v in runs.items()}
    common = set.intersection(*(set(v["fixture_uid"]) for v in losses.values() if not v.empty))
    rows = []
    for k, v in losses.items():
        v = v[v["fixture_uid"].isin(common)]
        rows.append({"model": k, **summarise(v)})
        losses[k] = v.sort_values("fixture_uid").reset_index(drop=True)
    summary = pd.DataFrame(rows).sort_values("rps").reset_index(drop=True)

    comps = []
    pairs = [
        ("M1+G1", "M1+G0"),
        ("M1+G2", "M1+G0"),
        ("M1+G3", "M1+G0"),
        ("M1+G0", "M1 goals-only (A2)"),
        ("M1+G0", "M1 default params (A3)"),
    ]
    if "M3 fused" in losses:
        pairs += [("M3 fused", "market-only"), ("M3 fused", "M1+G1")]
    for a, b in pairs:
        la, lb = losses[a], losses[b]
        block = la["season"].astype(str) + ":" + la["round"].astype(int).astype(str)
        for metric in ("loss_rps", "loss_grid", "loss_log"):
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
                    "a_better": c.ci_high < 0,
                }
            )
    result = Phase2Result(
        seasons,
        summary,
        pd.DataFrame(comps),
        {k: v.params for k, v in goal_models.items()},
        fusion_info,
        run_ids,
    )
    if log:
        scope = f"{seasons[0]}..{seasons[-1]}"
        for row in summary.to_dict("records"):
            metrics = {k: float(row[k]) for k in ("log_loss", "rps", "grid_log_loss", "n")}
            log_metrics(lake, f"phase2:{row['model']}", metrics, scope=scope, tags={"phase": "2"})
    return result


def _fit_fusion(
    store: SilverStore,
    params: TeamStrengthParams,
    gm: GoalModel,
    before: pd.Timestamp,
    method: str = "multiplicative",
    n_seasons: int = 5,
) -> FusionFit:
    """Fusion weights from the last ``n_seasons`` training seasons: M1 rates at each
    deadline plus the market prices observable then; ridge by leave-one-season-out CV."""
    dim = store.get("dim_fixture")
    seasons = [
        s
        for s in sorted(dim.loc[dim["round"].notna(), "season"].unique())
        if historical_deadlines(dim, s)["deadline_at"].max() < before
    ][-n_seasons:]
    rows = []
    for s in seasons:
        for d in historical_deadlines(dim, s)["deadline_at"]:
            info = InformationSet.at(d, store)
            fx = target_fixtures(info, 1)
            if fx.empty:
                continue
            mod = M1Predictor(params, gm).rates(info, fx)
            mk = market_features(info, fx, gm, method)
            rows.append(
                fx[["fixture_uid"]]
                .merge(mod, on="fixture_uid")
                .merge(mk, on="fixture_uid", how="left")
                .assign(season=s)
            )
    train = (
        pd.concat(rows)
        .merge(dim[["fixture_uid", "home_goals", "away_goals"]], on="fixture_uid")
        .dropna(subset=["home_goals"])
    )
    return fit_fusion_cv(train, Fusion((1.0, 0.0, 0.0), (0.0, 0.0), gm))
