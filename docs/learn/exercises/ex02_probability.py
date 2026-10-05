"""Exercise 02: the distributions behind StatsNuke.

Run: uv run python docs/learn/exercises/ex02_probability.py
"""

from __future__ import annotations

import math

import numpy as np
from _check import close, run, task
from scipy.stats import binom, nbinom, poisson

# ------------------------------------------------------------------ demo
lam = 1.5
print("Poisson(1.5) pmf 0..4:", np.round(poisson.pmf(np.arange(5), lam), 4))
for n in (10, 100, 10_000):
    print(
        f"Binomial({n}, {lam}/{n}) P(0) = {binom.pmf(0, n, lam / n):.5f}",
        f"(Poisson: {math.exp(-lam):.5f})",
    )
rng = np.random.default_rng(0)
mu, k = 7.5, 7.5
counts = rng.poisson(rng.gamma(k, mu / k, size=200_000))  # Gamma(shape k, scale μ/k): mean μ
print(
    f"Gamma–Poisson mixture: mean {counts.mean():.2f}, var {counts.var():.2f}",
    "(NegBin: μ + μ²/k = 15.0)",
)


# ------------------------------------------------------------------ your tasks
def poisson_pmf(k: int, lam: float) -> float:
    """e^(−λ) λ^k / k!  — from the formula (use math.exp, math.factorial)."""
    raise NotImplementedError


def clean_sheet_prob(opponent_rate: float) -> float:
    """P(the opponent scores 0) when their goals are Poisson(opponent_rate)."""
    raise NotImplementedError


def nbinom_scipy_params(mean: float, size: float) -> tuple[float, float]:
    """Convert NegBin (mean μ, size k) to scipy's (n, p) with n = k, p = k / (k + μ)."""
    raise NotImplementedError


def p_at_least(threshold: int, mean: float, size: float) -> float:
    """P(C ≥ threshold) for C ~ NegBin(mean, size). Hint: 1 − F(threshold − 1)."""
    raise NotImplementedError


def logit(p: np.ndarray) -> np.ndarray:
    raise NotImplementedError


def sigmoid(x: np.ndarray) -> np.ndarray:
    raise NotImplementedError


def lognormal_mean(m: float, s2: float) -> float:
    """E[e^X] for X ~ Normal(m, s2)."""
    raise NotImplementedError


@task("poisson_pmf matches scipy")
def _() -> None:
    close([poisson_pmf(k, 1.5) for k in range(6)], poisson.pmf(np.arange(6), 1.5))


@task("clean_sheet_prob for λ = 1.5 is e^(−1.5)")
def _() -> None:
    close(clean_sheet_prob(1.5), math.exp(-1.5))


@task("nbinom_scipy_params gives the right mean and variance")
def _() -> None:
    n, p = nbinom_scipy_params(7.5, 7.5)
    close([nbinom.mean(n, p), nbinom.var(n, p)], [7.5, 7.5 + 7.5**2 / 7.5])


@task("p_at_least: a defender averaging 7.5 reaches 10 actions")
def _() -> None:
    n, p = 7.5, 7.5 / 15.0
    close(p_at_least(10, 7.5, 7.5), 1 - nbinom.cdf(9, n, p))


@task("more variance → more threshold crossings at the same mean")
def _() -> None:
    assert p_at_least(10, 7.5, 3.0) > p_at_least(10, 7.5, 30.0)


@task("logit and sigmoid are inverses; logit(0.2) = −1.386")
def _() -> None:
    p = np.array([0.01, 0.2, 0.5, 0.93])
    close(sigmoid(logit(p)), p)
    close(logit(np.array([0.2])), [math.log(0.25)])


@task("lognormal_mean includes the s²/2 term")
def _() -> None:
    close(lognormal_mean(0.3, 0.04), math.exp(0.32))


run(globals())
