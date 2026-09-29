"""Model comparison with gameweek-level dependence (ARCHITECTURE.md §11.4).

Losses within a gameweek are correlated (shared fixtures, shared randomness), so the
bootstrap resamples whole gameweeks and Diebold–Mariano uses a Newey–West long-run
variance on per-gameweek loss differentials.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy.stats import t as student_t


@dataclass(frozen=True)
class Comparison:
    mean_diff: float  # mean per-gameweek loss of A minus B (negative: A better)
    ci_low: float
    ci_high: float
    dm_stat: float
    dm_pvalue: float
    n_blocks: int

    @property
    def significant(self) -> bool:
        return self.ci_low > 0 or self.ci_high < 0


def per_block_diff(
    loss_a: pd.Series, loss_b: pd.Series, block: pd.Series
) -> npt.NDArray[np.float64]:
    """Mean loss difference per block (gameweek), in block order."""
    df = pd.DataFrame(
        {"d": np.asarray(loss_a, float) - np.asarray(loss_b, float), "b": np.asarray(block)}
    )
    return df.groupby("b", sort=True)["d"].mean().to_numpy()


def newey_west_lrv(d: npt.NDArray[np.float64], lags: int | None = None) -> float:
    n = len(d)
    lags = math.floor(n ** (1 / 3)) if lags is None else lags
    x = d - d.mean()
    lrv = float(x @ x) / n
    for k in range(1, min(lags, n - 1) + 1):
        w = 1 - k / (lags + 1)
        lrv += 2 * w * float(x[k:] @ x[:-k]) / n
    return max(lrv, 1e-300)


def diebold_mariano(
    d: npt.NDArray[np.float64], lags: int | None = None, horizon: int = 1
) -> tuple[float, float]:
    """(statistic, two-sided p-value) for H0: E[d] = 0.

    Uses the Harvey–Leybourne–Newbold (1997) small-sample correction and Student-t
    critical values with n − 1 degrees of freedom; the plain normal version over-rejects
    at season-length samples (≈ 38 gameweeks).
    """
    n = len(d)
    h = horizon
    # h-step-ahead forecast errors are at most MA(h − 1): truncate there (DM 1995).
    lags = h - 1 if lags is None else lags
    stat = float(d.mean() / math.sqrt(newey_west_lrv(d, lags) / n))
    stat *= math.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    return stat, float(2 * student_t.sf(abs(stat), df=n - 1))


def compare(
    loss_a: pd.Series,
    loss_b: pd.Series,
    block: pd.Series,
    *,
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
    horizon: int = 1,
) -> Comparison:
    d = per_block_diff(loss_a, loss_b, block)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), size=(n_boot, len(d)))
    boot = d[idx].mean(axis=1)
    lo, hi = np.quantile(boot, [alpha / 2, 1 - alpha / 2])
    stat, p = diebold_mariano(d, horizon=horizon)
    return Comparison(float(d.mean()), float(lo), float(hi), stat, p, len(d))
