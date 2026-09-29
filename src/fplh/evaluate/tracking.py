"""Experiment tracking: a leaderboard in gold, plus MLflow when configured.

The default tracker needs no server: every (run, metric) is an upserted row of
``gold/leaderboard/leaderboard.parquet``. If the optional ``[tracking]`` extra is
installed and ``MLFLOW_TRACKING_URI`` is set, results are mirrored to MLflow too.
"""

from __future__ import annotations

import os
from collections.abc import Mapping

import pandas as pd

from fplh.lake.parquet import read_parquet, write_parquet
from fplh.lake.storage import Lake

LEADERBOARD_KEY = "gold/leaderboard/leaderboard.parquet"
KEYS = ["run_id", "scope", "metric"]


def log_metrics(
    lake: Lake,
    run_id: str,
    metrics: Mapping[str, float],
    *,
    scope: str = "all",
    tags: Mapping[str, str] | None = None,
) -> pd.DataFrame:
    """Upsert metrics for ``run_id`` / ``scope`` (e.g. a season); returns the board."""
    rows = pd.DataFrame(
        [
            {"run_id": run_id, "scope": scope, "metric": k, "value": float(v), **(tags or {})}
            for k, v in metrics.items()
        ]
    )
    board = (
        read_parquet(lake, LEADERBOARD_KEY)
        if lake.exists(LEADERBOARD_KEY)
        else pd.DataFrame(columns=[*KEYS, "value"])
    )
    board = pd.concat([board, rows], ignore_index=True).drop_duplicates(KEYS, keep="last")
    write_parquet(lake, LEADERBOARD_KEY, board, KEYS)
    _mirror_to_mlflow(run_id, metrics, scope, tags)
    return board


def _mirror_to_mlflow(
    run_id: str, metrics: Mapping[str, float], scope: str, tags: Mapping[str, str] | None
) -> None:
    if not os.environ.get("MLFLOW_TRACKING_URI"):
        return
    try:
        import mlflow
    except ImportError:
        return
    with mlflow.start_run(run_name=f"{run_id}:{scope}"):
        mlflow.set_tags({"run_id": run_id, "scope": scope, **(tags or {})})
        mlflow.log_metrics(dict(metrics))


def leaderboard(lake: Lake) -> pd.DataFrame:
    return read_parquet(lake, LEADERBOARD_KEY) if lake.exists(LEADERBOARD_KEY) else pd.DataFrame()
