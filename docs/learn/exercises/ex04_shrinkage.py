"""Exercise 04: Gamma–Poisson shrinkage, the method of moments and time decay.

Run: uv run python docs/learn/exercises/ex04_shrinkage.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from _check import close, run, task

from fplh.models.shrinkage import gamma_poisson

# ------------------------------------------------------------------ demo
# 400 simulated forwards: true shot rates ~ Gamma(mean 2, α = 8); 1–30 matches each
rng = np.random.default_rng(4)
n = 400
true_rate = rng.gamma(8.0, 2.0 / 8.0, n)
exposure = rng.integers(1, 31, n).astype(float)
shots = rng.poisson(true_rate * exposure).astype(float)
raw = shots / exposure
shrunk = gamma_poisson(pd.Series(shots), pd.Series(exposure), pd.Series(["FW"] * n))
print(
    f"MSE vs truth — raw: {np.mean((raw - true_rate) ** 2):.3f}, "
    f"fplh gamma_poisson: {np.mean((shrunk - true_rate) ** 2):.3f}"
)


# ------------------------------------------------------------------ your tasks
def posterior_mean(alpha: float, mu: float, shots: float, exposure: float) -> float:
    """E[r | data] for prior Gamma(α, α/μ) and Poisson data (shots over exposure 90s)."""
    raise NotImplementedError


def data_weight(alpha: float, mu: float, exposure: float) -> float:
    """w such that posterior mean = w · raw + (1 − w) · μ."""
    raise NotImplementedError


def mom_alpha(counts: np.ndarray, exposures: np.ndarray) -> float:
    """Method-of-moments prior strength for ONE group, exactly as shrinkage.gamma_poisson does:
    μ = Σc/Σe; use players with exposure ≥ 5 (if fewer than 5 such players return 5μ);
    between = var(c/e of those players, pandas default ddof=1) − mean(μ/e);
    α = clip(μ² / max(between, 1e-6 μ²), 0.5, 200)."""
    raise NotImplementedError


def shrink(counts: pd.Series, exposures: pd.Series, groups: pd.Series) -> pd.Series:
    """Your own gamma_poisson: per group, α from mom_alpha, then the posterior mean.
    (A group with μ ≤ 0 gets 0.)"""
    raise NotImplementedError


def decay_weight(age_days: np.ndarray, half_life_days: float) -> np.ndarray:
    """2^(−age / half-life)."""
    raise NotImplementedError


@task("posterior_mean: the worked example table")
def _() -> None:
    close(
        [
            posterior_mean(6, 2.0, 5, 1),
            posterior_mean(6, 2.0, 20, 6),
            posterior_mean(6, 2.0, 100, 30),
        ],
        [11 / 4, 26 / 9, 106 / 33],
    )


@task("posterior mean is the weighted average of raw and prior")
def _() -> None:
    a, mu, s, u = 3.0, 0.3, 2.0, 2.0
    w = data_weight(a, mu, u)
    close(posterior_mean(a, mu, s, u), w * s / u + (1 - w) * mu)


@task("mom_alpha matches the repo's shrinkage on one group")
def _() -> None:
    c, e = pd.Series(shots), pd.Series(exposure)
    a = mom_alpha(shots, exposure)
    mu = shots.sum() / exposure.sum()
    close((a + c) / (a / mu + e), gamma_poisson(c, e, pd.Series(["FW"] * n)))


@task("shrink reproduces fplh gamma_poisson with several groups")
def _() -> None:
    groups = pd.Series(rng.choice(["FW", "AM", "CB"], n))
    c, e = pd.Series(shots * (groups != "CB")), pd.Series(exposure)
    close(shrink(c, e, groups), gamma_poisson(c, e, groups))


@task("shrunk estimates beat raw rates against the truth")
def _() -> None:
    est = shrink(pd.Series(shots), pd.Series(exposure), pd.Series(["FW"] * n))
    assert np.mean((est - true_rate) ** 2) < np.mean((raw - true_rate) ** 2)


@task("decay_weight: one half-life halves the weight")
def _() -> None:
    close(decay_weight(np.array([0.0, 365.0, 730.0]), 365.0), [1.0, 0.5, 0.25])


run(globals())
