"""Reference solutions for exercise 15."""

from __future__ import annotations

import numpy as np


def mc_se(samples: np.ndarray) -> float:
    return float(np.std(samples, ddof=1) / np.sqrt(len(samples)))


def proportion_se(p: float, s: int) -> float:
    return float(np.sqrt(p * (1 - p) / s))


def capped(p: np.ndarray, total: float) -> np.ndarray:
    q = np.clip(np.asarray(p, dtype=float), 1e-9, None)
    fixed = np.zeros(len(q), dtype=bool)
    for _ in range(len(q) + 1):
        free = ~fixed
        remaining = total - fixed.sum()
        if remaining <= 0 or not free.any():
            break
        q[free] = q[free] * remaining / q[free].sum()
        over = free & (q > 1)
        if not over.any():
            break
        q[over] = 1.0
        fixed |= over
    return np.minimum(q, 1.0)


def systematic(weights: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    order = rng.permutation(len(weights))
    w = weights[order]
    c = np.cumsum(w)
    u = rng.random()
    picked_sorted = (np.floor(c - u) - np.floor(c - w - u)) >= 1
    out = np.zeros(len(weights), dtype=bool)
    out[order] = picked_sorted
    return out


def crn_difference(seed: int, common: bool, n: int = 2000) -> float:
    za = np.random.default_rng(seed).normal(size=n)
    zb = za if common else np.random.default_rng(seed + 10**6).normal(size=n)
    a = np.maximum(za, 0.0) + 1
    b = np.maximum(zb, 0.1) + 1
    return float(a.mean() - b.mean())


def coherent(result: object) -> bool:
    r = result
    for k, team in ((0, r.home_goals), (1, r.away_goals)):  # type: ignore[attr-defined]
        own = r.sides[k].events["goals_scored"].sum(axis=1)  # type: ignore[attr-defined]
        og = r.sides[1 - k].events["own_goals"].sum(axis=1)  # type: ignore[attr-defined]
        if not np.array_equal(own + og, team):
            return False
    return True
