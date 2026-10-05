"""Reference solutions for exercise 13."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import brentq


def pav(y: np.ndarray) -> np.ndarray:
    sums: list[float] = []
    counts: list[int] = []
    for v in np.asarray(y, dtype=float):
        sums.append(v)
        counts.append(1)
        while len(sums) > 1 and sums[-2] / counts[-2] > sums[-1] / counts[-1]:
            s, c = sums.pop(), counts.pop()
            sums[-1] += s
            counts[-1] += c
    return np.repeat([s / c for s, c in zip(sums, counts, strict=True)], counts)


def logistic_grad_hess(p: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return p - y, p * (1 - p)


def leaf_value(g: np.ndarray, h: np.ndarray, lam: float = 1.0) -> float:
    return float(-g.sum() / (h.sum() + lam))


def era_shift(p: np.ndarray, target: float) -> float:
    q = np.clip(p, 1e-6, 1 - 1e-6)
    lg = np.log(q / (1 - q))
    return brentq(lambda d: float(np.mean(1 / (1 + np.exp(-(lg + d))))) - target, -10, 10)


def start_curve(stage: object, chance: float = 75.0) -> np.ndarray:
    grid = pd.DataFrame(
        {"h_started_5": np.arange(6.0), "chance_of_playing": chance, "rest_days": 5.0}
    )
    return stage.predict(grid)  # type: ignore[attr-defined]
