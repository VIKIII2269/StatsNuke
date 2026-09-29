"""Market probabilities (ARCHITECTURE.md §7.3), minimal Phase 1 version.

Multiplicative de-vig and a closed-form independent-Poisson inversion of 1X2 (+ O/U 2.5)
prices into scoring rates. Phase 2 adds power/Shin de-vig and the richer benchmarks.
"""

from __future__ import annotations

import math

import numpy as np
import numpy.typing as npt
from scipy.optimize import minimize
from scipy.stats import poisson

MAX_GOALS = 10


def devig_multiplicative(prices: npt.ArrayLike) -> npt.NDArray[np.float64]:
    """Decimal odds (…, k) → probabilities summing to 1 along the last axis."""
    implied = 1.0 / np.asarray(prices, dtype=float)
    out: npt.NDArray[np.float64] = implied / implied.sum(axis=-1, keepdims=True)
    return out


def poisson_markets(lh: float, la: float, line: float = 2.5) -> tuple[float, float, float, float]:
    """(P home, P draw, P away, P over ``line``) for independent Poisson goals."""
    k = np.arange(MAX_GOALS + 1)
    grid = np.outer(poisson.pmf(k, lh), poisson.pmf(k, la))
    grid /= grid.sum()
    home = float(np.tril(grid, -1).sum())
    draw = float(np.trace(grid))
    away = float(np.triu(grid, 1).sum())
    total = k[:, None] + k[None, :]
    over = float(grid[total > line].sum())
    return home, draw, away, over


def invert_poisson(
    p1x2: npt.ArrayLike, p_over: float | None = None, line: float = 2.5
) -> tuple[float, float]:
    """Scoring rates (λ home, λ away) whose Poisson markets best match the prices."""
    target = np.asarray(p1x2, dtype=float)

    def loss(x: npt.NDArray[np.float64]) -> float:
        h, d, a, o = poisson_markets(math.exp(x[0]), math.exp(x[1]), line)
        err = (h - target[0]) ** 2 + (d - target[1]) ** 2 + (a - target[2]) ** 2
        if p_over is not None:
            err += (o - p_over) ** 2
        return float(err)

    res = minimize(loss, x0=np.log([1.4, 1.1]), method="L-BFGS-B", bounds=[(-3, 2), (-3, 2)])
    return float(math.exp(res.x[0])), float(math.exp(res.x[1]))


def expected_floor_div(lam: float, per: int) -> float:
    """E[floor(G / per)] for G ~ Poisson(lam) (e.g. goals-conceded penalties)."""
    k = np.arange(0, 30)
    return float(np.sum(poisson.pmf(k, lam) * (k // per)))
