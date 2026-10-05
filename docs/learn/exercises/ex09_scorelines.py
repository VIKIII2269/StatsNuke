"""Exercise 09: scoreline grids, markets, and fitting Dixon–Coles ρ.

Run: uv run python docs/learn/exercises/ex09_scorelines.py
"""

from __future__ import annotations

import numpy as np
from _check import close, run, task
from scipy.stats import poisson

from fplh.models.goal_benchmarks import g0_grid, g1_grid, markets

K = np.arange(11)

# ------------------------------------------------------------------ demo
for name, g in (("G0", g0_grid(1.5, 1.1)), ("G1 ρ=−0.15", g1_grid(1.5, 1.1, -0.15))):
    m = markets(g)
    print(
        f"{name:11s} home {m['home']:.3f} draw {m['draw']:.3f} away {m['away']:.3f} "
        f"over {m['over']:.3f}"
    )

# simulated league with a true Dixon–Coles ρ = −0.12
rng = np.random.default_rng(9)
n = 6000
lh, la = rng.uniform(0.8, 2.2, n), rng.uniform(0.6, 1.8, n)
true_rho = -0.12
hg, ag = np.empty(n, int), np.empty(n, int)
for i in range(n):
    g = g1_grid(lh[i], la[i], true_rho).ravel()
    cell = rng.choice(g.size, p=g)
    hg[i], ag[i] = divmod(cell, 11)


# ------------------------------------------------------------------ your tasks
def grid_g0(lh: float, la: float) -> np.ndarray:
    """(11, 11) grid of independent Poisson goals, normalised to sum 1."""
    raise NotImplementedError


def grid_g1(lh: float, la: float, rho: float) -> np.ndarray:
    """Dixon–Coles: G0 with τ on 0-0, 0-1, 1-0, 1-1 (see the lesson's table), renormalised."""
    raise NotImplementedError


def home_away_draw(grid: np.ndarray) -> tuple[float, float, float]:
    """(P home win, P draw, P away win); rows are home goals."""
    raise NotImplementedError


def p_clean_sheet_home(grid: np.ndarray) -> float:
    """P(the away team scores 0)."""
    raise NotImplementedError


def fit_rho(lh: np.ndarray, la: np.ndarray, hg: np.ndarray, ag: np.ndarray) -> float:
    """Maximum-likelihood ρ given per-match rates: use benchmarks()['G1'].fit(...)."""
    raise NotImplementedError


@task("grid_g0 matches the repo")
def _() -> None:
    close(grid_g0(1.5, 1.1), g0_grid(1.5, 1.1))


@task("grid_g1 matches the repo for several ρ")
def _() -> None:
    for rho in (-0.2, -0.045, 0.0, 0.1):
        close(grid_g1(1.5, 1.1, rho), g1_grid(1.5, 1.1, rho))


@task("Dixon–Coles keeps over 2.5 but raises the draw (ρ < 0)")
def _() -> None:
    a, b = grid_g0(1.5, 1.1), grid_g1(1.5, 1.1, -0.15)
    close(markets(b)["over"], markets(a)["over"], tol=1e-9)
    assert home_away_draw(b)[1] > home_away_draw(a)[1]


@task("market helpers")
def _() -> None:
    g = g0_grid(1.5, 1.1)
    m = markets(g)
    close(home_away_draw(g), [m["home"], m["draw"], m["away"]])
    close(p_clean_sheet_home(g), poisson.pmf(0, 1.1), tol=1e-6)


@task("fit_rho recovers the simulated ρ (±0.05)")
def _() -> None:
    close(fit_rho(lh, la, hg, ag), true_rho, tol=0.05)


run(globals())
