"""Reference solutions for exercise 08."""

from __future__ import annotations

import numpy as np
from scipy.optimize import brentq

from fplh.models import market


def overround(prices: np.ndarray) -> float:
    return float(np.sum(1 / prices) - 1)


def devig_mult(prices: np.ndarray) -> np.ndarray:
    pi = 1 / prices
    return pi / pi.sum()


def devig_pow(prices: np.ndarray) -> np.ndarray:
    pi = 1 / prices
    c = brentq(lambda c: float(np.sum(pi**c)) - 1, 1e-6, 50)
    p = pi**c
    return p / p.sum()


def devig_shin(prices: np.ndarray) -> np.ndarray:
    pi = 1 / prices
    s = pi.sum()

    def probs(z: float) -> np.ndarray:
        return (np.sqrt(z**2 + 4 * (1 - z) * pi**2 / s) - z) / (2 * (1 - z))

    z = brentq(lambda z: float(probs(z).sum()) - 1, 0.0, 0.999)
    return probs(z)


def longshot_shading(prices: np.ndarray) -> float:
    i = int(np.argmax(prices))
    return float(devig_mult(prices)[i] - devig_pow(prices)[i])


def rates_from_market(p1x2: np.ndarray, p_over: float | None) -> tuple[float, float]:
    return market.invert_poisson(p1x2, p_over)
