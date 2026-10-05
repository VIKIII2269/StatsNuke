"""Reference solutions for exercise 17."""

from __future__ import annotations

import numpy as np


def ev(p: float, price: float) -> float:
    return p * price - 1


def kelly_fraction(p: float, price: float, kappa: float = 0.25, cap: float = 0.05) -> float:
    e = ev(p, price)
    if e <= 0:
        return 0.0
    return min(kappa * e / (price - 1), cap)


def clv(price_taken: float, p_fair_close: float) -> float:
    return price_taken * p_fair_close - 1


def no_edge_season(seed: int, n: int = 400) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    p = rng.uniform(0.25, 0.6, n)
    price = (1 / p) * np.exp(rng.normal(0, 0.03, n))
    won = rng.random(n) < p
    clvs = price * p - 1
    profit = np.where(won, price - 1, -1.0)
    return float(clvs.mean()), float(profit.sum() / n)
