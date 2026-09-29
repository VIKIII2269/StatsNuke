from __future__ import annotations

import numpy as np
import pandas as pd

from fplh.evaluate.bootstrap import compare, diebold_mariano, newey_west_lrv


def _losses(
    rng: np.random.Generator, n_gw: int, per_gw: int, edge: float
) -> tuple[pd.Series, pd.Series, pd.Series]:
    gw = np.repeat(np.arange(n_gw), per_gw)
    shared = rng.normal(0, 1, n_gw)[gw]  # within-gameweek correlation
    a = shared + rng.normal(0, 1, len(gw))
    b = shared + rng.normal(0, 1, len(gw)) + edge
    return pd.Series(a), pd.Series(b), pd.Series(gw)


def test_size_under_the_null() -> None:
    """A-vs-A comparisons reject at ≈ 5 % (block structure respected)."""
    rng = np.random.default_rng(42)
    rejections = sum(diebold_mariano(_d(rng))[1] < 0.05 for _ in range(1000))
    assert 0.025 <= rejections / 1000 <= 0.075


def _d(rng: np.random.Generator) -> np.ndarray:
    a, b, gw = _losses(rng, n_gw=38, per_gw=20, edge=0.0)
    return (a - b).groupby(gw).mean().to_numpy()


def test_detects_a_real_edge() -> None:
    rng = np.random.default_rng(1)
    a, b, gw = _losses(rng, n_gw=38, per_gw=50, edge=0.2)
    c = compare(a, b, gw, n_boot=1000)
    assert c.significant and c.ci_high < 0 and c.dm_pvalue < 0.01
    assert c.n_blocks == 38


def test_newey_west_reduces_to_variance_without_lags() -> None:
    d = np.array([1.0, -1.0, 2.0, 0.0])
    assert np.isclose(newey_west_lrv(d, lags=0), np.var(d))
