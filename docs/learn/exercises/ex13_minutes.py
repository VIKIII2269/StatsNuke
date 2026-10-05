"""Exercise 13: boosting maths, isotonic calibration, monotone constraints, era shifts.

Run: uv run python docs/learn/exercises/ex13_minutes.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from _check import close, run, task
from scipy.optimize import brentq, isotonic_regression  # noqa: F401  (you may need them)

from fplh.evaluate.metrics import ece
from fplh.models.minutes import _logit_shift, fit_stage

# ------------------------------------------------------------------ demo: synthetic minutes data
rng = np.random.default_rng(13)
n = 6000
X = pd.DataFrame(
    {
        "h_started_5": rng.integers(0, 6, n).astype(float),
        "chance_of_playing": np.where(
            rng.random(n) < 0.3, np.nan, rng.choice([0, 25, 50, 75, 100], n)
        ),
        "rest_days": rng.uniform(2, 10, n),
    }
)
logit = -4.5 + 0.9 * X["h_started_5"] + 0.02 * np.nan_to_num(X["chance_of_playing"], nan=75)
Y = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(float).to_numpy()
STAGE = fit_stage(X, Y, rounds=100)
P = STAGE.predict(X)
print(f"in-sample ECE {ece(P, Y):.4f}; mean predicted {P.mean():.3f} vs observed {Y.mean():.3f}")


# ------------------------------------------------------------------ your tasks
def pav(y: np.ndarray) -> np.ndarray:
    """Pool-adjacent-violators: the non-decreasing sequence closest (least squares) to y.
    Keep a stack of blocks (sum, count); merge while the last block's mean exceeds the new one's."""
    raise NotImplementedError


def logistic_grad_hess(p: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Gradient and Hessian of log loss w.r.t. the log-odds: g = p − y, h = p(1 − p)."""
    raise NotImplementedError


def leaf_value(g: np.ndarray, h: np.ndarray, lam: float = 1.0) -> float:
    """Optimal leaf weight −Σg / (Σh + λ)."""
    raise NotImplementedError


def era_shift(p: np.ndarray, target: float) -> float:
    """δ with mean(σ(logit p + δ)) = target (use brentq on [−10, 10])."""
    raise NotImplementedError


def start_curve(stage: object, chance: float = 75.0) -> np.ndarray:
    """The trained stage's predictions for h_started_5 = 0..5 at the given chance_of_playing
    and rest_days = 5 (build a DataFrame with the same columns as X; call stage.predict)."""
    raise NotImplementedError


@task("pav matches scipy.optimize.isotonic_regression")
def _() -> None:
    for seed in range(5):
        y = np.random.default_rng(seed).random(40).round(0)
        close(pav(y), isotonic_regression(y).x)


@task("logistic gradient and Hessian")
def _() -> None:
    g, h = logistic_grad_hess(np.array([0.8, 0.3]), np.array([0.0, 1.0]))
    close(g, [0.8, -0.7])
    close(h, [0.16, 0.21])


@task("leaf value is a Newton step")
def _() -> None:
    close(leaf_value(np.array([0.8, -0.7, 0.2]), np.array([0.16, 0.21, 0.16]), 1.0), -0.3 / 1.53)


@task("era_shift matches fplh.models.minutes._logit_shift")
def _() -> None:
    p = rng.uniform(0.05, 0.4, 500)
    close(era_shift(p, 0.163), _logit_shift(p, 0.163), tol=1e-8)


@task("the trained stage is monotone in recent starts (and calibrated)")
def _() -> None:
    curve = start_curve(STAGE)
    assert np.all(np.diff(curve) >= -1e-12), curve
    assert ece(P, Y) < 0.03


run(globals())
