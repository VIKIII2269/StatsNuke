"""Reference solutions for exercise 14."""

from __future__ import annotations

import numpy as np
from scipy.stats import nbinom

from fplh.models.gk import negbin_pmf


def goal_rate(shot_rate: float, xg_per_shot: float, finishing: float) -> float:
    return shot_rate * xg_per_shot * finishing


def scorer_probs(rates: np.ndarray, on_pitch: np.ndarray) -> np.ndarray:
    w = np.where(on_pitch, rates, 0.0)
    return w / w.sum()


def simulate_allocation(team_goals: np.ndarray, rates: np.ndarray, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    p = scorer_probs(rates, np.ones(len(rates), bool))
    return np.array([rng.multinomial(g, p) for g in team_goals])


def p_defensive_points(mean: float, size: float, threshold: int) -> float:
    return float(1 - nbinom.cdf(threshold - 1, size, size / (size + mean)))


def expected_save_points(mean: float, size: float) -> float:
    pmf = negbin_pmf(np.array([mean]), size, 60)[0]
    return float(pmf @ (np.arange(61) // 3))


def p_yellow(rate_per_90: float, minutes: float) -> float:
    return float(1 - np.exp(-rate_per_90 * minutes / 90))
