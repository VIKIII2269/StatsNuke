"""M3: fusion of market-implied and model rates (ARCHITECTURE.md §7.4).

    log λ_k = w · log λ_k^mkt + (1 − w) · log λ_k^mod + b_k
    w       = σ(α₀ + α₁ log(1 + τ) + α₂ · liq)

τ is hours between the price snapshot and kickoff, ``liq`` a liquidity proxy (bookmakers
quoting, minus overround). Without a market price w = 0. Parameters minimise the
scoreline-grid negative log-likelihood under a fitted goal model, with a ridge penalty.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy.optimize import minimize

from fplh.models.goal_benchmarks import GoalModel, benchmarks

Array = npt.NDArray[np.float64]
FUSION_COLUMNS = [
    "lam_mkt_home",
    "lam_mkt_away",
    "lam_mod_home",
    "lam_mod_away",
    "tau_hours",
    "liquidity",
]


def _sigmoid(x: Array) -> Array:
    out: Array = 1.0 / (1.0 + np.exp(-x))
    return out


@dataclass
class Fusion:
    alpha: tuple[float, float, float] = (1.0, 0.0, 0.0)
    bias: tuple[float, float] = (0.0, 0.0)
    goal_model: GoalModel = field(default_factory=lambda: benchmarks()["G0"])

    def weight(
        self, tau_hours: Array, liquidity: Array, has_market: npt.NDArray[np.bool_]
    ) -> Array:
        a0, a1, a2 = self.alpha
        w = _sigmoid(a0 + a1 * np.log1p(np.maximum(tau_hours, 0.0)) + a2 * liquidity)
        return np.where(has_market, w, 0.0)

    def rates(self, df: pd.DataFrame) -> tuple[Array, Array, Array]:
        """(λ_home, λ_away, w) per row of a frame with FUSION_COLUMNS."""
        has = df["lam_mkt_home"].notna().to_numpy()
        w = self.weight(
            df["tau_hours"].fillna(0).to_numpy(float),
            df["liquidity"].fillna(0).to_numpy(float),
            has,
        )
        out = []
        for side, b in (("home", self.bias[0]), ("away", self.bias[1])):
            mod = np.log(df[f"lam_mod_{side}"].to_numpy(float))
            mkt = np.log(df[f"lam_mkt_{side}"].fillna(1.0).to_numpy(float))
            out.append(np.exp(w * mkt + (1 - w) * mod + b))
        return out[0], out[1], w

    def fit(self, df: pd.DataFrame, ridge: float = 0.01) -> Fusion:
        """``df`` has FUSION_COLUMNS plus home_goals, away_goals."""
        hg, ag = df["home_goals"].to_numpy(int), df["away_goals"].to_numpy(int)

        def nll(x: Array) -> float:
            cand = Fusion((x[0], x[1], x[2]), (x[3], x[4]), self.goal_model)
            lh, la, _ = cand.rates(df)
            return -float(np.mean(self.goal_model.log_prob(lh, la, hg, ag))) + ridge * float(
                np.sum(x[1:] ** 2)
            )

        x0 = np.array([*self.alpha, *self.bias])
        res = minimize(
            nll, x0, method="Nelder-Mead", options={"maxiter": 2000, "xatol": 1e-4, "fatol": 1e-7}
        )
        x = res.x
        return Fusion((x[0], x[1], x[2]), (x[3], x[4]), self.goal_model)


def liquidity(n_books: pd.Series, overround: pd.Series) -> pd.Series:
    """More books and a thinner margin → more informative price (roughly standardised)."""
    out: pd.Series = np.log1p(n_books.fillna(0)) - 20 * (overround.fillna(0.1) - 0.05)
    return out
