"""Exercise 07: proper scores, calibration and gameweek-block bootstrap.

Run: uv run python docs/learn/exercises/ex07_evaluation.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from _check import close, run, task

from fplh.evaluate import metrics
from fplh.evaluate.bootstrap import compare, diebold_mariano, per_block_diff

# ------------------------------------------------------------------ demo
rng = np.random.default_rng(7)
n = 600
probs = rng.dirichlet([4, 2.5, 3], n)
outcome = np.array([rng.choice(3, p=p) for p in probs])
print(f"log loss {metrics.log_loss(probs, outcome):.4f}  RPS {metrics.rps(probs, outcome):.4f}")

# two models on 110 gameweeks × 50 rows; losses share a gameweek-level shock
gw = np.repeat(np.arange(110), 50)
shock = rng.normal(0, 1.0, 110)[gw]
loss_a = 3.60 + shock + rng.normal(0, 1, len(gw))
loss_b = 3.62 + shock + rng.normal(0, 1, len(gw)) + 0.5 * rng.normal(0, 1, 110)[gw]
c = compare(pd.Series(loss_a), pd.Series(loss_b), pd.Series(gw))
print(
    f"A − B: {c.mean_diff:+.4f}  95% CI [{c.ci_low:+.4f}, {c.ci_high:+.4f}]  DM p {c.dm_pvalue:.3f}"
)


# ------------------------------------------------------------------ your tasks
def my_log_loss(p: np.ndarray, y: np.ndarray) -> float:
    """Mean −log p[i, y_i], clipping probabilities to [1e-15, 1]."""
    raise NotImplementedError


def my_rps(p: np.ndarray, y: np.ndarray) -> float:
    """Ranked probability score for ordered outcomes (cumulative differences, / (r − 1))."""
    raise NotImplementedError


def my_ece(p: np.ndarray, y: np.ndarray, bins: int = 10) -> float:
    """Equal-width-bin ECE: Σ_b share_b · |mean(y in b) − mean(p in b)|,
    with bin = min(floor(p·bins), bins − 1)."""
    raise NotImplementedError


def block_bootstrap_ci(
    loss_a: np.ndarray, loss_b: np.ndarray, block: np.ndarray, n_boot: int = 2000, seed: int = 0
) -> tuple[float, float]:
    """95% CI of the mean per-block difference: average a − b within each block (sorted by
    block id), then resample blocks with replacement exactly like bootstrap.compare:
    rng = default_rng(seed); idx = rng.integers(0, nb, size=(n_boot, nb)); quantiles 2.5/97.5."""
    raise NotImplementedError


def row_bootstrap_ci(
    loss_a: np.ndarray, loss_b: np.ndarray, n_boot: int = 2000, seed: int = 0
) -> tuple[float, float]:
    """The WRONG way: resample individual rows (ignoring gameweeks)."""
    raise NotImplementedError


@task("my_log_loss matches metrics.log_loss")
def _() -> None:
    close(my_log_loss(probs, outcome), metrics.log_loss(probs, outcome))


@task("my_rps matches metrics.rps, and the worked example is 0.145")
def _() -> None:
    close(my_rps(probs, outcome), metrics.rps(probs, outcome))
    close(my_rps(np.array([[0.5, 0.3, 0.2]]), np.array([0])), 0.145)


@task("my_ece matches metrics.ece")
def _() -> None:
    p = rng.uniform(0, 1, 5000)
    y = (rng.uniform(0, 1, 5000) < p**1.3).astype(float)
    close(my_ece(p, y), metrics.ece(p, y))


@task("block_bootstrap_ci reproduces bootstrap.compare")
def _() -> None:
    lo, hi = block_bootstrap_ci(loss_a, loss_b, gw)
    close([lo, hi], [c.ci_low, c.ci_high], tol=1e-9)


@task("row bootstrap is too narrow when losses are correlated within gameweeks")
def _() -> None:
    blo, bhi = block_bootstrap_ci(loss_a, loss_b, gw)
    rlo, rhi = row_bootstrap_ci(loss_a, loss_b)
    assert (rhi - rlo) < 0.8 * (bhi - blo), ((rlo, rhi), (blo, bhi))


@task("DM on identical-in-distribution models rejects about 5% of the time")
def _() -> None:
    r = np.random.default_rng(11)
    rejections = sum(
        diebold_mariano(
            per_block_diff(
                pd.Series(r.normal(0, 1, 380)),
                pd.Series(r.normal(0, 1, 380)),
                pd.Series(np.repeat(np.arange(38), 10)),
            )
        )[1]
        < 0.05
        for _ in range(400)
    )
    assert 0.02 < rejections / 400 < 0.09, rejections / 400
    my_rps(np.array([[1.0, 0.0, 0.0]]), np.array([0]))  # (make this task need your rps)


run(globals())
