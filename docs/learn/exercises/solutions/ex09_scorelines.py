"""Reference solutions for exercise 09."""

from __future__ import annotations

import numpy as np
from scipy.stats import poisson

from fplh.models.goal_benchmarks import benchmarks

K = np.arange(11)


def grid_g0(lh: float, la: float) -> np.ndarray:
    g = poisson.pmf(K, lh)[:, None] * poisson.pmf(K, la)[None, :]
    return g / g.sum()


def grid_g1(lh: float, la: float, rho: float) -> np.ndarray:
    g = poisson.pmf(K, lh)[:, None] * poisson.pmf(K, la)[None, :]
    g[0, 0] *= 1 - lh * la * rho
    g[0, 1] *= 1 + lh * rho
    g[1, 0] *= 1 + la * rho
    g[1, 1] *= 1 - rho
    g = np.clip(g, 1e-300, None)
    return g / g.sum()


def home_away_draw(grid: np.ndarray) -> tuple[float, float, float]:
    return float(np.tril(grid, -1).sum()), float(np.trace(grid)), float(np.triu(grid, 1).sum())


def p_clean_sheet_home(grid: np.ndarray) -> float:
    return float(grid[:, 0].sum())


def fit_rho(lh: np.ndarray, la: np.ndarray, hg: np.ndarray, ag: np.ndarray) -> float:
    return benchmarks()["G1"].fit(lh, la, hg, ag).params["rho"]
