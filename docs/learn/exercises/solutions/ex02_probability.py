"""Reference solutions for exercise 02."""

from __future__ import annotations

import math

import numpy as np
from scipy.stats import nbinom


def poisson_pmf(k: int, lam: float) -> float:
    return math.exp(-lam) * lam**k / math.factorial(k)


def clean_sheet_prob(opponent_rate: float) -> float:
    return math.exp(-opponent_rate)


def nbinom_scipy_params(mean: float, size: float) -> tuple[float, float]:
    return size, size / (size + mean)


def p_at_least(threshold: int, mean: float, size: float) -> float:
    n, p = nbinom_scipy_params(mean, size)
    return float(1 - nbinom.cdf(threshold - 1, n, p))


def logit(p: np.ndarray) -> np.ndarray:
    return np.log(p / (1 - p))


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-x))


def lognormal_mean(m: float, s2: float) -> float:
    return math.exp(m + s2 / 2)
