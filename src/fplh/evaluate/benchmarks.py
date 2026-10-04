"""Phase 3 benchmarks, measured before the simulator exists: the naive floors (A0's
season per-90 × expected minutes, and the last-5 average) and the OpenFPL
re-implementation, walk-forward over the tuning seasons on the same player-fixtures."""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from fplh.evaluate.player_level import align, compare_runs, outcomes, player_losses, summary
from fplh.evaluate.tracking import log_metrics
from fplh.evaluate.walk_forward import run_walk_forward
from fplh.features.information_set import SilverStore
from fplh.features.spine import historical_deadlines
from fplh.lake.storage import Lake
from fplh.models.baselines import A0Player, NaiveLast5
from fplh.models.openfpl import OpenFPLReplica
from fplh.rules.config import load_rules

PAIRS = (("openfpl_replica", "naive_last5"), ("openfpl_replica", "a0"), ("naive_last5", "a0"))


@dataclass
class BenchmarkResult:
    seasons: list[str]
    summary: pd.DataFrame
    comparisons: pd.DataFrame
    run_ids: dict[str, list[str]] = field(default_factory=dict)
    losses: dict[str, pd.DataFrame] = field(default_factory=dict)


def evaluate_benchmarks(lake: Lake, seasons: list[str], *, log: bool = True) -> BenchmarkResult:
    store = SilverStore(lake)
    dim = store.get("dim_fixture")
    by_season = {s: list(historical_deadlines(dim, s)["deadline_at"]) for s in seasons}
    preds: dict[str, list[pd.DataFrame]] = {"a0": [], "naive_last5": [], "openfpl_replica": []}
    run_ids: dict[str, list[str]] = {k: [] for k in preds}
    replica = OpenFPLReplica()
    for season, deadlines in by_season.items():
        rules_config = f"fpl_{season.replace('-', '_')}.yaml"
        for name, predictor in (
            ("a0", A0Player(load_rules(season))),
            ("naive_last5", NaiveLast5()),
            ("openfpl_replica", replica),
        ):
            wf = run_walk_forward(store, predictor, deadlines, rules_config=rules_config, lake=lake)
            preds[name].append(wf.predictions)
            run_ids[name].append(wf.run_id)
    actual = outcomes(store)
    losses = align({k: player_losses(pd.concat(v), actual) for k, v in preds.items()})
    table = pd.DataFrame([{"model": k, **summary(v)} for k, v in losses.items()]).sort_values("mse")
    comps = pd.DataFrame(
        [
            {"a": a, "b": b, "metric": metric, **compare_runs(losses[a], losses[b], metric)}
            for a, b in PAIRS
            for metric in ("se", "ae")
        ]
    )
    if log:
        scope = f"{seasons[0]}..{seasons[-1]}"
        for row in table.to_dict("records"):
            metrics = {str(k): float(v) for k, v in row.items() if k != "model"}
            log_metrics(lake, f"phase3:{row['model']}", metrics, scope=scope, tags={"phase": "3"})
    return BenchmarkResult(seasons, table.reset_index(drop=True), comps, run_ids, losses)
