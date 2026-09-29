"""Forecast metrics (ARCHITECTURE.md §11.2). Lower is better unless noted."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import numpy.typing as npt
import pandas as pd

Array = npt.ArrayLike
EPS = 1e-15


def log_loss(probs: Array, outcome: Array) -> float:
    """Mean −log p(observed). ``probs`` is (n, k); ``outcome`` holds class indices."""
    p = np.asarray(probs, dtype=float)
    y = np.asarray(outcome, dtype=int)
    return float(-np.mean(np.log(np.clip(p[np.arange(len(y)), y], EPS, 1.0))))


def rps(probs: Array, outcome: Array) -> float:
    """Ranked probability score for ordered outcomes (home, draw, away)."""
    p = np.asarray(probs, dtype=float)
    y = np.asarray(outcome, dtype=int)
    r = p.shape[1]
    onehot = np.eye(r)[y]
    cum = np.cumsum(p - onehot, axis=1)[:, : r - 1]
    return float(np.mean(np.sum(cum**2, axis=1) / (r - 1)))


def scoreline_grid_log_loss(
    grids: Array, home_goals: Array, away_goals: Array, cap: int = 6
) -> float:
    """Log loss over the joint score grid (n, cap+1, cap+1); scores above ``cap`` are
    pooled into the remainder mass ``1 − Σ grid``."""
    g = np.asarray(grids, dtype=float)
    h = np.asarray(home_goals, dtype=int)
    a = np.asarray(away_goals, dtype=int)
    inside = (h <= cap) & (a <= cap)
    p = np.where(
        inside,
        g[np.arange(len(h)), np.minimum(h, cap), np.minimum(a, cap)],
        1.0 - g.reshape(len(h), -1).sum(axis=1),
    )
    return float(-np.mean(np.log(np.clip(p, EPS, 1.0))))


def brier(p: Array, y: Array) -> float:
    pp, yy = np.asarray(p, dtype=float), np.asarray(y, dtype=float)
    return float(np.mean((pp - yy) ** 2))


def ece(p: Array, y: Array, bins: int = 10) -> float:
    """Expected calibration error with equal-width bins."""
    pp, yy = np.asarray(p, dtype=float), np.asarray(y, dtype=float)
    idx = np.minimum((pp * bins).astype(int), bins - 1)
    total = 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            total += m.mean() * abs(yy[m].mean() - pp[m].mean())
    return float(total)


def randomised_pit(
    cdf_below: Array, cdf_at: Array, rng: np.random.Generator
) -> npt.NDArray[np.float64]:
    """u = F(y−1) + v·[F(y) − F(y−1)], v ~ U(0,1); uniform if the forecast is calibrated."""
    lo, hi = np.asarray(cdf_below, dtype=float), np.asarray(cdf_at, dtype=float)
    u: npt.NDArray[np.float64] = lo + rng.uniform(size=lo.shape) * (hi - lo)
    return u


def crps_discrete(pmf: Array, y: Array) -> float:
    """Mean discrete CRPS Σ_k (F(k) − 1[y ≤ k])² for pmfs over 0..K (n, K+1)."""
    p = np.asarray(pmf, dtype=float)
    yy = np.asarray(y, dtype=int)
    cdf = np.cumsum(p, axis=1)
    k = np.arange(p.shape[1])
    step = (yy[:, None] <= k[None, :]).astype(float)
    return float(np.mean(np.sum((cdf - step) ** 2, axis=1)))


def mae(pred: Array, y: Array) -> float:
    return float(np.mean(np.abs(np.asarray(pred, dtype=float) - np.asarray(y, dtype=float))))


def rmse(pred: Array, y: Array) -> float:
    return float(
        np.sqrt(np.mean((np.asarray(pred, dtype=float) - np.asarray(y, dtype=float)) ** 2))
    )


def spearman_within(
    pred: Sequence[float] | pd.Series,
    y: Sequence[float] | pd.Series,
    groups: Sequence[str] | pd.Series,
) -> float:
    """Mean Spearman ρ between prediction and outcome within each group (e.g. position),
    weighted by group size; groups with < 3 rows or no variance are skipped."""
    df = pd.DataFrame({"p": list(pred), "y": list(y), "g": list(groups)})
    total, weight = 0.0, 0
    for _, d in df.groupby("g"):
        if len(d) < 3 or d["p"].nunique() < 2 or d["y"].nunique() < 2:
            continue
        rho = d["p"].rank().corr(d["y"].rank())
        total += rho * len(d)
        weight += len(d)
    return float(total / weight) if weight else float("nan")


def outcome_index(home_goals: Array, away_goals: Array) -> npt.NDArray[np.int64]:
    """0 = home win, 1 = draw, 2 = away win."""
    h, a = np.asarray(home_goals), np.asarray(away_goals)
    return np.where(h > a, 0, np.where(h == a, 1, 2)).astype(np.int64)
