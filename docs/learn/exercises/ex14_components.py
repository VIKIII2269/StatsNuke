"""Exercise 14: attack allocation, defensive thresholds, saves and cards.

Run: uv run python docs/learn/exercises/ex14_components.py
"""

from __future__ import annotations

import numpy as np
from _check import close, run, task
from scipy.stats import nbinom

from fplh.models.gk import SaveModel, negbin_pmf

# ------------------------------------------------------------------ demo
sm = SaveModel()
print(f"expected saves, 90 min vs an opponent expecting 1.8 goals: {sm.mean(1.8, 90, 1.0):.2f}")
pmf = negbin_pmf(np.array([3.4]), 14.0, 15)[0]
print("P(saves = 0..6):", np.round(pmf[:7], 3))


# ------------------------------------------------------------------ your tasks
def goal_rate(shot_rate: float, xg_per_shot: float, finishing: float) -> float:
    """Non-penalty goals per 90 = r · q̄ · f."""
    raise NotImplementedError


def scorer_probs(rates: np.ndarray, on_pitch: np.ndarray) -> np.ndarray:
    """P(scorer = p) ∝ rate among players on the pitch (zero for those off)."""
    raise NotImplementedError


def simulate_allocation(team_goals: np.ndarray, rates: np.ndarray, seed: int) -> np.ndarray:
    """(S, P) goals per player: in simulation s, assign each of team_goals[s] goals to a player
    drawn with scorer_probs(rates, all on pitch). Totals per row must equal team_goals."""
    raise NotImplementedError


def p_defensive_points(mean: float, size: float, threshold: int) -> float:
    """P(C ≥ threshold), C ~ NegBin(mean, size)  (scipy: n = size, p = size/(size+mean))."""
    raise NotImplementedError


def expected_save_points(mean: float, size: float) -> float:
    """E[floor(saves / 3)] for saves ~ NegBin(mean, size). Use fplh.models.gk.negbin_pmf
    with upto=60 (the tail mass beyond is negligible)."""
    raise NotImplementedError


def p_yellow(rate_per_90: float, minutes: float) -> float:
    """1 − exp(−rate · minutes / 90)."""
    raise NotImplementedError


@task("goal_rate multiplies the three parts")
def _() -> None:
    close(goal_rate(3.0, 0.12, 1.05), 0.378)


@task("scorer_probs: the lesson's example")
def _() -> None:
    rates = np.array([0.40, 0.20, 0.05, *[0.01] * 8, 0.9])
    on = np.r_[np.ones(11, bool), False]  # the 0.9 striker is on the bench
    p = scorer_probs(rates, on)
    close(p[0], 0.40 / 0.73, tol=1e-9)
    close(p[-1], 0.0)
    close(p.sum(), 1.0)


@task("allocation is coherent: player goals sum to team goals")
def _() -> None:
    tg = np.random.default_rng(0).poisson(1.4, 2000)
    rates = np.array([0.5, 0.3, 0.1, 0.1])
    g = simulate_allocation(tg, rates, seed=1)
    assert (g.sum(axis=1) == tg).all()
    close(g.mean(axis=0) / tg.mean(), rates / rates.sum(), tol=0.03)


@task("defensive threshold probability")
def _() -> None:
    close(p_defensive_points(7.5, 7.5, 10), 1 - nbinom.cdf(9, 7.5, 0.5))


@task("expected save points with the repo's NegBin")
def _() -> None:
    k = np.arange(61)
    pmf = nbinom.pmf(k, 14.0, 14.0 / 17.4)
    close(expected_save_points(3.4, 14.0), float(pmf @ (k // 3)), tol=1e-6)


@task("p_yellow scales with minutes")
def _() -> None:
    close(p_yellow(0.2, 90), 1 - np.exp(-0.2))
    assert p_yellow(0.2, 30) < p_yellow(0.2, 90)


run(globals())
