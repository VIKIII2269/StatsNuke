"""Exercise 15: Monte Carlo error, systematic sampling, common random numbers, coherence.

Run: uv run python docs/learn/exercises/ex15_monte_carlo.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from _check import close, run, task

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # repo root, for tests.*
from fplh.sim.simulator import (
    FixtureInputs,
    League,
    TimingModel,
    _capped,
    simulate_fixture,
    summarise,
)
from tests.property.test_simulator_properties import PARAMS, RULES, side

# ------------------------------------------------------------------ demo
FX = FixtureInputs("s:h:a", (1.5, 1.1), (side("h", 1), side("a", 2)), 5)
RESULT = simulate_fixture(FX, PARAMS, RULES, League(), TimingModel(), 3000, seed=0)
print(summarise(RESULT)[["player_uid", "expected_points", "p_start", "p_60"]].head(5).round(3))


# ------------------------------------------------------------------ your tasks
def mc_se(samples: np.ndarray) -> float:
    """Standard error of the sample mean: sd (ddof=1) / √S."""
    raise NotImplementedError


def proportion_se(p: float, s: int) -> float:
    """√(p(1 − p)/S)."""
    raise NotImplementedError


def capped(p: np.ndarray, total: float) -> np.ndarray:
    """Scale p to sum to `total` with every entry ≤ 1: repeatedly rescale the uncapped entries
    to the remaining total and cap those that exceed 1."""
    raise NotImplementedError


def systematic(weights: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """1-D systematic sample: random order, cumulative weights c, one uniform u, and
    picked = floor(c − u) − floor(c − w − u) ≥ 1. Return a boolean mask in the ORIGINAL order."""
    raise NotImplementedError


def crn_difference(seed: int, common: bool, n: int = 2000) -> float:
    """Estimate E[A] − E[B] where A = max(Z, 0) + 1 and B = max(Z, 0.1) + 1 with Z ~ N(0, 1).
    If common, use the SAME Z draws for both; otherwise independent draws (seeds seed, seed+10⁶)."""
    raise NotImplementedError


def coherent(result: object) -> bool:
    """True if, for both sides and every simulation, player goals + opponents' own goals equal
    the team's goals. result.sides[k].events['goals_scored'] is (S, P); result.home_goals (S,)."""
    raise NotImplementedError


@task("Monte Carlo standard errors (§8.3 numbers)")
def _() -> None:
    x = np.random.default_rng(1).normal(0, 3, 10_000)
    close(mc_se(x), x.std(ddof=1) / 100)
    close(proportion_se(0.05, 10_000), 0.00218, tol=1e-5)


@task("capped matches the simulator's _capped")
def _() -> None:
    p = np.array([0.95, 0.9, 0.8, 0.5, 0.3, 0.2, 0.2, 0.1, 0.1, 0.1, 0.05, 0.02])
    close(capped(p, 10.0), _capped(p, 10.0), tol=1e-9)


@task("systematic: exact count and correct inclusion frequencies")
def _() -> None:
    rng = np.random.default_rng(2)
    w = capped(np.array([0.95, 0.9, 0.8, 0.5, 0.3, 0.2, 0.2, 0.1, 0.05]), 4.0)
    picks = np.array([systematic(w, rng) for _ in range(20_000)])
    assert (picks.sum(axis=1) == 4).all()
    close(picks.mean(axis=0), w, tol=0.015)


@task("common random numbers shrink the spread of the difference")
def _() -> None:
    with_crn = np.std([crn_difference(s, True) for s in range(60)])
    without = np.std([crn_difference(s, False) for s in range(60)])
    assert with_crn < 0.25 * without, (with_crn, without)


@task("the repo's simulator is coherent")
def _() -> None:
    assert coherent(RESULT)


run(globals())
