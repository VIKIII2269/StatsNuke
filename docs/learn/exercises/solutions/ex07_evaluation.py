"""Reference solutions for exercise 07."""

from __future__ import annotations

import numpy as np
import pandas as pd


def my_log_loss(p: np.ndarray, y: np.ndarray) -> float:
    return float(-np.mean(np.log(np.clip(p[np.arange(len(y)), y], 1e-15, 1.0))))


def my_rps(p: np.ndarray, y: np.ndarray) -> float:
    r = p.shape[1]
    cum = np.cumsum(p - np.eye(r)[y], axis=1)[:, : r - 1]
    return float(np.mean(np.sum(cum**2, axis=1) / (r - 1)))


def my_ece(p: np.ndarray, y: np.ndarray, bins: int = 10) -> float:
    idx = np.minimum((p * bins).astype(int), bins - 1)
    total = 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            total += m.mean() * abs(y[m].mean() - p[m].mean())
    return float(total)


def block_bootstrap_ci(
    loss_a: np.ndarray, loss_b: np.ndarray, block: np.ndarray, n_boot: int = 2000, seed: int = 0
) -> tuple[float, float]:
    d = pd.Series(loss_a - loss_b).groupby(np.asarray(block), sort=True).mean().to_numpy()
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), size=(n_boot, len(d)))
    lo, hi = np.quantile(d[idx].mean(axis=1), [0.025, 0.975])
    return float(lo), float(hi)


def row_bootstrap_ci(
    loss_a: np.ndarray, loss_b: np.ndarray, n_boot: int = 2000, seed: int = 0
) -> tuple[float, float]:
    d = loss_a - loss_b
    rng = np.random.default_rng(seed)
    means = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(n_boot)])
    lo, hi = np.quantile(means, [0.025, 0.975])
    return float(lo), float(hi)
