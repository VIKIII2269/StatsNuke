"""Exercise 08: de-vig (multiplicative, power, Shin) and market inversion.

Run: uv run python docs/learn/exercises/ex08_devig.py
"""

from __future__ import annotations

import numpy as np
from _check import close, run, task
from scipy.optimize import brentq  # noqa: F401  (you will need it)

from fplh.models import market

# ------------------------------------------------------------------ demo
odds = np.array([2.10, 3.40, 3.60])
print("implied:", np.round(1 / odds, 4), "overround:", round(float((1 / odds).sum() - 1), 4))
for m in ("multiplicative", "power", "shin"):
    print(f"{m:15s}", np.round(market.devig(odds, m), 4))


# ------------------------------------------------------------------ your tasks
def overround(prices: np.ndarray) -> float:
    """Σ 1/o − 1."""
    raise NotImplementedError


def devig_mult(prices: np.ndarray) -> np.ndarray:
    raise NotImplementedError


def devig_pow(prices: np.ndarray) -> np.ndarray:
    """p = π^c with Σ π^c = 1; find c with brentq on [1e-6, 50]."""
    raise NotImplementedError


def devig_shin(prices: np.ndarray) -> np.ndarray:
    """Shin probabilities; find z in [0, 0.999] with brentq so they sum to 1."""
    raise NotImplementedError


def longshot_shading(prices: np.ndarray) -> float:
    """How much more power de-vig takes off the LONGEST price than multiplicative does:
    multiplicative p_longshot − power p_longshot (positive for a normal market)."""
    raise NotImplementedError


def rates_from_market(p1x2: np.ndarray, p_over: float | None) -> tuple[float, float]:
    """Poisson (λ_home, λ_away) reproducing the market — use fplh.models.market.invert_poisson."""
    raise NotImplementedError


@task("overround of 2.10/3.40/3.60 is 4.81%")
def _() -> None:
    close(overround(odds), 0.04806, tol=1e-4)


@task("devig_mult matches the repo")
def _() -> None:
    close(devig_mult(odds), market.devig_multiplicative(odds))


@task("devig_pow matches the repo")
def _() -> None:
    for o in (odds, np.array([1.25, 6.0, 12.0]), np.array([1.9, 1.95])):
        close(devig_pow(o), market.devig_power(o), tol=1e-8)


@task("devig_shin matches the repo")
def _() -> None:
    for o in (odds, np.array([1.25, 6.0, 12.0])):
        close(devig_shin(o), market.devig_shin(o), tol=1e-8)


@task("power shades the longshot more than multiplicative")
def _() -> None:
    assert longshot_shading(np.array([1.25, 6.0, 12.0])) > 0.005


@task("the totals price raises the inverted goal level")
def _() -> None:
    p = market.devig(odds, "multiplicative")
    lo = sum(rates_from_market(p, None))
    hi = sum(rates_from_market(p, 0.52))
    assert hi > lo + 0.3, (lo, hi)
    h, d, a, _ = market.poisson_markets(*rates_from_market(p, 0.52))
    close([h, d, a], p, tol=0.035)  # G0 cannot hit the draw exactly: see the lesson


run(globals())
