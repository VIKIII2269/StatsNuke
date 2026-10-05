"""Reference solutions for exercise 04."""

from __future__ import annotations

import numpy as np
import pandas as pd


def posterior_mean(alpha: float, mu: float, shots: float, exposure: float) -> float:
    return (alpha + shots) / (alpha / mu + exposure)


def data_weight(alpha: float, mu: float, exposure: float) -> float:
    return exposure / (exposure + alpha / mu)


def mom_alpha(counts: np.ndarray, exposures: np.ndarray) -> float:
    c, e = pd.Series(counts, dtype=float), pd.Series(exposures, dtype=float)
    mu = c.sum() / max(e.sum(), 1e-9)
    big = e >= 5
    if big.sum() < 5:
        return 5.0 * mu
    rates = c[big] / e[big]
    between = rates.var() - (mu / e[big]).mean()
    return float(np.clip(mu**2 / max(between, 1e-6 * mu**2), 0.5, 200.0))


def shrink(counts: pd.Series, exposures: pd.Series, groups: pd.Series) -> pd.Series:
    out = pd.Series(np.nan, index=counts.index)
    for g in groups.unique():
        m = groups == g
        c, e = counts[m], exposures[m]
        mu = c.sum() / max(e.sum(), 1e-9)
        if mu <= 0:
            out[m] = 0.0
            continue
        a = mom_alpha(c.to_numpy(), e.to_numpy())
        out[m] = (a + c) / (a / mu + e)
    return out


def decay_weight(age_days: np.ndarray, half_life_days: float) -> np.ndarray:
    return np.power(2.0, -np.asarray(age_days, dtype=float) / half_life_days)
