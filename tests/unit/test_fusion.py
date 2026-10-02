from __future__ import annotations

import numpy as np
import pandas as pd

from fplh.models.fusion import Fusion, fit_fusion_cv


def frame(n: int, mkt_noise: float, mod_noise: float, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    lh, la = rng.uniform(0.8, 2.2, n), rng.uniform(0.6, 1.8, n)
    return pd.DataFrame(
        {
            "lam_mkt_home": lh * np.exp(rng.normal(0, mkt_noise, n)),
            "lam_mkt_away": la * np.exp(rng.normal(0, mkt_noise, n)),
            "lam_mod_home": lh * np.exp(rng.normal(0, mod_noise, n)),
            "lam_mod_away": la * np.exp(rng.normal(0, mod_noise, n)),
            "tau_hours": 24.0,
            "liquidity": 1.0,
            "home_goals": rng.poisson(lh),
            "away_goals": rng.poisson(la),
        }
    )


def test_weight_favours_the_sharper_source() -> None:
    sharp_market = Fusion((0.0, 0.0, 0.0)).fit(frame(1500, 0.05, 0.3))
    sharp_model = Fusion((0.0, 0.0, 0.0)).fit(frame(1500, 0.3, 0.05))
    w_market = sharp_market.rates(frame(5, 0.05, 0.3))[2][0]
    w_model = sharp_model.rates(frame(5, 0.3, 0.05))[2][0]
    assert w_market > 0.6 > 0.4 > w_model


def test_no_market_means_model_only() -> None:
    df = frame(3, 0.1, 0.1).assign(lam_mkt_home=np.nan, lam_mkt_away=np.nan)
    lh, la, w = Fusion((3.0, 0.0, 0.0)).rates(df)
    assert (w == 0).all()
    assert np.allclose(lh, df["lam_mod_home"]) and np.allclose(la, df["lam_mod_away"])


def seasons_with_bias(biases: list[float], n: int = 250) -> pd.DataFrame:
    """The market misses home goals by exp(b_s) in season s; the model is useless."""
    parts = []
    for s, b in enumerate(biases):
        df = frame(n, 0.0, 1.0, seed=s).assign(season=f"s{s}")
        rng = np.random.default_rng(100 + s)
        df["home_goals"] = rng.poisson(df["lam_mkt_home"] * np.exp(b))
        parts.append(df)
    return pd.concat(parts, ignore_index=True)


def test_cv_shrinks_a_bias_that_flips_between_seasons() -> None:
    start, ridges = Fusion((3.0, 0.0, 0.0)), (0.01, 10.0)
    steady = fit_fusion_cv(seasons_with_bias([0.15, 0.15, 0.15, 0.15]), start, ridges)
    flipping = fit_fusion_cv(seasons_with_bias([0.15, -0.15, 0.15, -0.15]), start, ridges)
    assert abs(steady.fusion.bias[0] - 0.15) < 0.06  # a real, stable bias is kept
    assert flipping.ridge > steady.ridge
    assert abs(flipping.fusion.bias[0]) < 0.03  # noise is shrunk away
    assert set(steady.cv) == set(ridges)
