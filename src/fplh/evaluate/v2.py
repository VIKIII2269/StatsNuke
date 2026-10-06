"""Model v2 against v1 and the benchmarks, walk-forward on the player gate (ticket v2).

v2 = the simulator with news-aware minutes (``minutes="news"``) stacked with the OpenFPL
replica and the FPL crowd (``models.stack``). The stack trains on every earlier season's
walk-forward forecasts from ``train_from`` on, so the first gate season already has
several seasons of history; within a season it keeps learning as outcomes arrive.

Scored exactly as the Phase 3 gate: squared error per player-fixture on the rows every
model scored, gameweek-block bootstrap CIs and DM tests.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from fplh.evaluate.phase3 import run, simulator
from fplh.evaluate.player_level import align, compare_runs, outcomes, player_losses, summary
from fplh.features.information_set import SilverStore
from fplh.features.spine import historical_deadlines
from fplh.lake.storage import Lake
from fplh.models.openfpl import OpenFPLReplica
from fplh.models.stack import StackConfig, stack_frame, walk_forward_stack

V2 = "v2 (stack)"


@dataclass
class V2Result:
    summary: pd.DataFrame
    gate: pd.DataFrame
    per_season: pd.DataFrame
    predictions: pd.DataFrame


def season_runs(
    lake: Lake,
    store: SilverStore,
    groups: list[list[str]],
    with_v1: bool = True,
) -> dict[str, pd.DataFrame]:
    """Cached walk-forward runs, one run per group of seasons (the Phase 3 gate's three
    seasons are one cached run; earlier seasons one each)."""
    dim = store.get("dim_fixture")
    out: dict[str, list[pd.DataFrame]] = {"sim_v1": [], "sim_news": [], "replica": []}
    for group in groups:
        ds = [d for s in group for d in historical_deadlines(dim, s)["deadline_at"]]
        out["replica"].append(run(lake, store, OpenFPLReplica(), ds))
        if with_v1:
            out["sim_v1"].append(run(lake, store, simulator(lake, store, ds), ds))
        out["sim_news"].append(run(lake, store, simulator(lake, store, ds, minutes="news"), ds))
    return {k: pd.concat(v, ignore_index=True) for k, v in out.items() if v}


def stacked(
    store: SilverStore,
    sim: pd.DataFrame,
    rep: pd.DataFrame,
    deadlines: list[pd.Timestamp],
    config: StackConfig | None = None,
) -> pd.DataFrame:
    pm = store.get("fact_player_match")
    frame = stack_frame(
        sim, rep, store.get("fpl_round_transfers"), pm[["player_uid", "value", "observed_at"]]
    )
    y = pm[["player_uid", "fixture_uid", "total_points", "observed_at"]]
    return walk_forward_stack(frame, y, deadlines, config)


def evaluate_v2(
    lake: Lake,
    seasons: list[str],
    train_from: list[str],
    config: StackConfig | None = None,
) -> V2Result:
    store = SilverStore(lake)
    dim = store.get("dim_fixture")
    ds = [d for s in seasons for d in historical_deadlines(dim, s)["deadline_at"]]
    gate_runs = season_runs(lake, store, [seasons])
    train = season_runs(lake, store, [[s] for s in train_from], with_v1=False)
    runs = {
        k: pd.concat([train[k], v], ignore_index=True) if k in train else v
        for k, v in gate_runs.items()
    }
    preds = {
        V2: stacked(store, runs["sim_news"], runs["replica"], ds, config),
        "simulator v1": runs["sim_v1"],
        "simulator + news": runs["sim_news"],
        "openfpl_replica": runs["replica"],
    }
    keep = set(ds)
    preds = {k: v[v["deadline_at"].isin(keep)] for k, v in preds.items()}
    actual = outcomes(store)
    losses = align({k: player_losses(v, actual) for k, v in preds.items()})
    table = pd.DataFrame([{"model": k, **summary(v)} for k, v in losses.items()]).sort_values("mse")
    gate, per_season = [], []
    for b in ("simulator v1", "simulator + news", "openfpl_replica"):
        gate.append({"versus": b, **compare_runs(losses[V2], losses[b], "se")})
        for s in seasons:
            a = losses[V2][losses[V2]["season"] == s]
            bb = losses[b][losses[b]["season"] == s]
            per_season.append(
                {"versus": b, "season": s, "diff": float(a["se"].mean() - bb["se"].mean())}
            )
    news = compare_runs(losses["simulator + news"], losses["simulator v1"], "se")
    gate.append({"versus": "news vs v1 (simulator only)", **news})
    return V2Result(table, pd.DataFrame(gate), pd.DataFrame(per_season), preds[V2])
