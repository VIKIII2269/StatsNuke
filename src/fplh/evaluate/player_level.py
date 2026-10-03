"""Player-level walk-forward scoring and paired comparisons (ARCHITECTURE.md §11.3–11.4).

Every model is scored on the same player-fixtures: rows of the deadline spine whose
outcome exists in ``fact_player_match`` (players who left before the fixture have none).
The primary metric is squared error of expected points (MSE, proper for a mean
forecast); MAE, Spearman ρ within position (all rows and rows where the player played)
and top-10 precision per gameweek and position are guardrails. Comparisons use the
gameweek-block bootstrap and Diebold–Mariano test (``bootstrap.compare``).
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd

from fplh.evaluate import metrics as m
from fplh.evaluate.bootstrap import compare
from fplh.features.information_set import SilverStore

KEYS = ["player_uid", "fixture_uid"]


def outcomes(store: SilverStore) -> pd.DataFrame:
    pm = store.get("fact_player_match")[[*KEYS, "total_points", "minutes", "season"]]
    rounds = store.get("dim_fixture")[["fixture_uid", "round"]]
    out: pd.DataFrame = pm.merge(rounds, on="fixture_uid", how="left")
    return out


def player_losses(pred: pd.DataFrame, actual: pd.DataFrame) -> pd.DataFrame:
    """Per player-fixture: prediction, outcome, squared and absolute error, block."""
    j = pred[[*KEYS, "deadline_at", "position", "expected_points"]].merge(actual, on=KEYS)
    j = j.dropna(subset=["expected_points"])
    j["se"] = (j["expected_points"] - j["total_points"]) ** 2
    j["ae"] = (j["expected_points"] - j["total_points"]).abs()
    j["played"] = j["minutes"] > 0
    j["block"] = j["season"].astype(str) + ":" + j["round"].astype("Int64").astype(str)
    return j.sort_values([*KEYS, "deadline_at"]).reset_index(drop=True)


def top_k_precision(losses: pd.DataFrame, k: int = 10) -> float:
    """Mean over (gameweek, position) of |top-k predicted ∩ actual top k| / k, where the
    actual top k includes everyone tied with the k-th score."""
    scores = []
    for _, g in losses.groupby(["block", "position"]):
        if len(g) < k:
            continue
        predicted = set(g.nlargest(k, "expected_points").index)
        cutoff = g["total_points"].nlargest(k).iloc[-1]
        actual = set(g.index[g["total_points"] >= cutoff])
        scores.append(len(predicted & actual) / k)
    return float(np.mean(scores)) if scores else float("nan")


def summary(losses: pd.DataFrame) -> dict[str, float]:
    played = losses[losses["played"]]
    return {
        "mse": float(losses["se"].mean()),
        "mae": float(losses["ae"].mean()),
        "spearman_within_position": m.spearman_within(
            losses["expected_points"], losses["total_points"], losses["position"]
        ),
        "mse_played": float(played["se"].mean()),
        "spearman_within_position_played": m.spearman_within(
            played["expected_points"], played["total_points"], played["position"]
        ),
        "top10_precision": top_k_precision(losses),
        "n": float(len(losses)),
        "n_played": float(len(played)),
    }


def align(runs: Mapping[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """Restrict every run's losses to the player-fixtures all of them scored."""
    common = set.intersection(
        *(set(zip(v["player_uid"], v["fixture_uid"], strict=True)) for v in runs.values())
    )
    out = {}
    for name, v in runs.items():
        keep = [k in common for k in zip(v["player_uid"], v["fixture_uid"], strict=True)]
        out[name] = v[keep].sort_values(KEYS).reset_index(drop=True)
    return out


def compare_runs(
    a: pd.DataFrame, b: pd.DataFrame, metric: str = "se", n_boot: int = 2000
) -> dict[str, float]:
    """``a`` − ``b`` on aligned losses (negative favours ``a``)."""
    c = compare(a[metric], b[metric], a["block"], n_boot=n_boot)
    return {
        "mean_diff": c.mean_diff,
        "ci_low": c.ci_low,
        "ci_high": c.ci_high,
        "dm_p": c.dm_pvalue,
        "n_blocks": float(c.n_blocks),
    }
