"""Market probabilities (ARCHITECTURE.md §7.3, M2).

De-vig methods (multiplicative, power, Shin) turn bookmaker odds into probabilities; the
independent-Poisson inversion turns 1X2 (+ O/U 2.5) probabilities into scoring rates.
Inversion under the dependent goal models (G1–G3) lives in ``goal_benchmarks``.
"""

from __future__ import annotations

import math

import numpy as np
import numpy.typing as npt
from scipy.optimize import brentq, minimize
from scipy.stats import poisson

MAX_GOALS = 10


def devig_multiplicative(prices: npt.ArrayLike) -> npt.NDArray[np.float64]:
    """Decimal odds (…, k) → probabilities summing to 1 along the last axis."""
    implied = 1.0 / np.asarray(prices, dtype=float)
    out: npt.NDArray[np.float64] = implied / implied.sum(axis=-1, keepdims=True)
    return out


def devig_power(prices: npt.ArrayLike) -> npt.NDArray[np.float64]:
    """p_k = π_k^c with c solving Σ π_k^c = 1 (shades longshots more)."""
    pi = 1.0 / np.asarray(prices, dtype=float)
    if abs(pi.sum() - 1.0) < 1e-12:
        return pi
    c = brentq(lambda c: float(np.sum(pi**c)) - 1.0, 1e-6, 50.0)
    out: npt.NDArray[np.float64] = pi**c
    return out / out.sum()


def devig_shin(prices: npt.ArrayLike) -> npt.NDArray[np.float64]:
    """Shin (1993): p_k = (√(z² + 4(1−z)π_k²/S) − z) / (2(1−z)), z solving Σ p_k = 1.

    ``z`` is the implied share of informed money; the method corrects the
    favourite–longshot bias and is usually the best calibrated on 1X2 markets.
    """
    pi = 1.0 / np.asarray(prices, dtype=float)
    s = pi.sum()

    def probs(z: float) -> npt.NDArray[np.float64]:
        out: npt.NDArray[np.float64] = (np.sqrt(z**2 + 4 * (1 - z) * pi**2 / s) - z) / (2 * (1 - z))
        return out

    if s <= 1.0:
        return pi / s
    z = brentq(lambda z: float(probs(z).sum()) - 1.0, 0.0, 0.999)
    return probs(z)


DEVIG = {"multiplicative": devig_multiplicative, "power": devig_power, "shin": devig_shin}


def devig(prices: npt.ArrayLike, method: str = "multiplicative") -> npt.NDArray[np.float64]:
    """De-vig one market (1-D prices) with the named method."""
    return DEVIG[method](prices)


def shin_z(prices: npt.ArrayLike) -> float:
    """Shin's implied informed-trading share ``z`` for one market."""
    pi = 1.0 / np.asarray(prices, dtype=float)
    s = pi.sum()
    if s <= 1.0:
        return 0.0
    return float(
        brentq(
            lambda z: (
                float(np.sum((np.sqrt(z**2 + 4 * (1 - z) * pi**2 / s) - z) / (2 * (1 - z)))) - 1.0
            ),
            0.0,
            0.999,
        )
    )


def devig_calibration(prices: npt.ArrayLike, outcome: npt.ArrayLike) -> dict[str, float]:
    """Mean log loss of each de-vig method on (n, k) closing prices vs observed outcomes
    (class indices). The default method is the one with the lowest loss (§7.3)."""
    p = np.asarray(prices, dtype=float)
    y = np.asarray(outcome, dtype=int)
    out = {}
    for name, fn in DEVIG.items():
        probs = np.array([fn(row) for row in p])
        out[name] = float(-np.mean(np.log(np.clip(probs[np.arange(len(y)), y], 1e-15, 1))))
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
