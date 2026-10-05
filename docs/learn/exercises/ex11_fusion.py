"""Exercise 11: the log-linear pool and a fitted fusion weight.

Run: uv run python docs/learn/exercises/ex11_fusion.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from _check import close, run, task

from fplh.models.fusion import Fusion

# ------------------------------------------------------------------ demo
rng = np.random.default_rng(11)


def synthetic(n: int, market_noise: float, model_noise: float) -> pd.DataFrame:
    """True rates, a 'market' and a 'model' that each see them with log-normal noise."""
    lh, la = rng.uniform(0.8, 2.2, n), rng.uniform(0.6, 1.8, n)
    return pd.DataFrame(
        {
            "lam_mkt_home": lh * np.exp(rng.normal(0, market_noise, n)),
            "lam_mkt_away": la * np.exp(rng.normal(0, market_noise, n)),
            "lam_mod_home": lh * np.exp(rng.normal(0, model_noise, n)),
            "lam_mod_away": la * np.exp(rng.normal(0, model_noise, n)),
            "tau_hours": 24.0,
            "liquidity": 0.0,
            "home_goals": rng.poisson(lh),
            "away_goals": rng.poisson(la),
        }
    )


GOOD_MARKET = synthetic(4000, 0.05, 0.35)
GOOD_MODEL = synthetic(4000, 0.35, 0.05)
fit = Fusion().fit(GOOD_MARKET, ridge=0.01)
print(
    "market better → fitted weight:",
    round(float(fit.weight(np.array([24.0]), np.array([0.0]), np.array([True]))[0]), 3),
)


# ------------------------------------------------------------------ your tasks
def sigmoid(x: float) -> float:
    raise NotImplementedError


def pooled_rate(lam_mkt: float, lam_mod: float, w: float, bias: float = 0.0) -> float:
    """exp(w log λ_mkt + (1 − w) log λ_mod + bias)."""
    raise NotImplementedError


def fitted_weight(df: pd.DataFrame) -> float:
    """Fit fplh's Fusion() on df with ridge 0.01 and return its weight at τ = 24 h, liq = 0,
    market present."""
    raise NotImplementedError


@task("sigmoid(1.58) ≈ 0.829")
def _() -> None:
    close(sigmoid(1.58), 0.82920, tol=1e-4)


@task("pooled_rate: the lesson's example")
def _() -> None:
    close(pooled_rate(1.6, 1.3, 0.829), 1.6**0.829 * 1.3**0.171)


@task("pooled_rate agrees with Fusion.rates")
def _() -> None:
    fu = Fusion(alpha=(1.58, 0.0, 0.0), bias=(-0.016, 0.001))
    df = GOOD_MARKET.head(5)
    lh, _la, w = fu.rates(df)
    for i in range(5):
        close(
            pooled_rate(df["lam_mkt_home"].iloc[i], df["lam_mod_home"].iloc[i], w[i], -0.016), lh[i]
        )


@task("the learned weight follows whichever input is better")
def _() -> None:
    assert fitted_weight(GOOD_MARKET) > 0.7
    assert fitted_weight(GOOD_MODEL) < 0.3


run(globals())
