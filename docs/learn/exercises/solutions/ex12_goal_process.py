"""Reference solutions for exercise 12."""

from __future__ import annotations

import math

import numpy as np
from scipy.special import gammaln

from fplh.sim.emulator import Emulator


def p_no_goal(hazards: np.ndarray, exposures: np.ndarray) -> float:
    return float(np.exp(-np.sum(hazards * exposures)))


def trailing_effect(goals: np.ndarray, base_rate: float, trailing: np.ndarray) -> float:
    return math.log(goals[trailing].sum() / (base_rate * trailing.sum()))


def frailty_loglik(y: np.ndarray, lam: np.ndarray, a: float) -> float:
    n, big = y.sum(), lam.sum()
    return float(
        np.sum(y * np.log(lam) - gammaln(y + 1))
        + gammaln(n + a)
        - gammaln(a)
        + a * np.log(a)
        - (n + a) * np.log(a + big)
    )


def nominal_for(emu: Emulator, mu_home: float, mu_away: float) -> tuple[float, float]:
    x, y = emu.nominal_for_means([mu_home], [mu_away])
    return float(np.exp(x[0])), float(np.exp(y[0]))
