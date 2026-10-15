"""Evaluation of model v2.1 (``models.chances``) against v2 (experiment P5.9).

Input: v2's horizon-1 forecasts (player_uid, fixture_uid, deadline_at, expected_points,
e_assists, p_play) for the tuning and test seasons. For each season, the chance model is
trained on Understat appearances before the season's first deadline, and the forecasts
are adjusted with features as of each row's deadline. Reported on all player-fixtures:
- Poisson deviance of FPL assists, v2 against the chance model;
- points MSE for each blend weight on the tuning season;
- the test MSE difference at ``ASSIST_WEIGHT``, with a gameweek-block bootstrap CI.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from fplh.features.information_set import SilverStore
from fplh.models.chances import ASSIST_POINTS, ASSIST_WEIGHT, AssistModel, adjust

WEIGHTS = (0.0, 0.25, 0.5, 0.75, 1.0)


@dataclass
class AssistsResult:
    tuning: pd.DataFrame  # weight → MSE on the tuning season
    test: pd.DataFrame  # per test season and overall: MSE v2, v2.1, difference, CI
    deviance: pd.DataFrame  # Poisson deviance of FPL assists, v2 vs chance model, per season


def _dev(y: np.ndarray, m: np.ndarray) -> np.ndarray:
    m = np.clip(m, 1e-6, None)
    out: np.ndarray = 2 * (np.where(y > 0, y * np.log(np.maximum(y, 1e-9) / m), 0.0) - (y - m))
    return out


def _block_ci(
    diff: np.ndarray, blocks: np.ndarray, n_boot: int = 2000, seed: int = 0
) -> tuple[float, float]:
    b = pd.Series(diff).groupby(blocks).agg(["sum", "size"])
    s, n = b["sum"].to_numpy(), b["size"].to_numpy()
    rng = np.random.default_rng(seed)
    draws = [
        s[k].sum() / n[k].sum() for k in (rng.integers(0, len(b), len(b)) for _ in range(n_boot))
    ]
    return float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))


def adjusted_forecasts(store: SilverStore, forecasts: pd.DataFrame) -> pd.DataFrame:
    """v2 forecasts with the chance model's assists, walk-forward by season, plus the
    actual FPL points and assists."""
    dim = store.get("dim_fixture").set_index("fixture_uid")
    f = forecasts.copy()
    f["season"] = f["fixture_uid"].map(dim["season"])
    f["gw"] = f["fixture_uid"].map(dim["round"])
    parts = []
    for _, g in f.groupby("season"):
        at = pd.to_datetime(g["deadline_at"], utc=True)
        model = AssistModel.fit(store, at.min())
        parts.append(adjust(g, store, at, model=model, weight=1.0))
    out = pd.concat(parts)
    pm = store.get("fact_player_match")[["player_uid", "fixture_uid", "total_points", "assists"]]
    pm = pm.drop_duplicates(["player_uid", "fixture_uid"])
    out = out.merge(pm, on=["player_uid", "fixture_uid"], how="inner")
    # adjust() was run at weight 1: recover v2's expected points
    out["expected_points"] = out["expected_points"] - ASSIST_POINTS * (
        out["assists_chance"] - out["e_assists_v2"]
    )
    return out


def evaluate_assists(
    store: SilverStore,
    forecasts: pd.DataFrame,
    tune: str = "2021-22",
    test: tuple[str, ...] = ("2022-23", "2023-24", "2024-25"),
) -> AssistsResult:
    d = adjusted_forecasts(store, forecasts)
    y_pts = d["total_points"].to_numpy(float)
    delta = (d["assists_chance"] - d["e_assists_v2"]).to_numpy(float)
    e = d["expected_points"].to_numpy(float)
    t = (d["season"] == tune).to_numpy()
    tuning = pd.DataFrame(
        [
            {"weight": w, "mse": float(((e + ASSIST_POINTS * w * delta - y_pts)[t] ** 2).mean())}
            for w in WEIGHTS
        ]
    )
    rows = []
    blocks = (d["season"].astype(str) + ":" + d["gw"].astype(str)).to_numpy()
    err0 = (e - y_pts) ** 2
    err1 = (e + ASSIST_POINTS * ASSIST_WEIGHT * delta - y_pts) ** 2
    for name, mask in [
        *((s, (d["season"] == s).to_numpy()) for s in test),
        ("all test", d["season"].isin(test).to_numpy()),
    ]:
        diff = (err1 - err0)[mask]
        lo, hi = _block_ci(diff, blocks[mask])
        rows.append(
            {
                "seasons": name,
                "n": int(mask.sum()),
                "mse_v2": err0[mask].mean(),
                "mse_v21": err1[mask].mean(),
                "diff": diff.mean(),
                "low": lo,
                "high": hi,
            }
        )
    y_a = d["assists"].to_numpy(float)
    dev = (
        pd.DataFrame(
            {
                "season": d["season"],
                "v2": _dev(y_a, d["e_assists_v2"].to_numpy(float)),
                "chance": _dev(y_a, d["assists_chance"].to_numpy(float)),
            }
        )
        .groupby("season")
        .mean()
        .reset_index()
    )
    return AssistsResult(tuning, pd.DataFrame(rows), dev)


def read_forecasts(paths: list[str]) -> pd.DataFrame:
    """Horizon-1 rows from parquet files or directories of them."""
    import glob
    import os

    files = [
        f
        for p in paths
        for f in (
            glob.glob(os.path.join(p, "**", "*.parquet"), recursive=True)
            if os.path.isdir(p)
            else [p]
        )
    ]
    cols = [
        "player_uid",
        "fixture_uid",
        "deadline_at",
        "horizon",
        "expected_points",
        "e_assists",
        "p_play",
    ]
    f = pd.concat([pd.read_parquet(x, columns=cols) for x in files], ignore_index=True)
    f = f[f["horizon"] == 1].drop(columns="horizon")
    out: pd.DataFrame = f.drop_duplicates(["player_uid", "fixture_uid"])
    return out
