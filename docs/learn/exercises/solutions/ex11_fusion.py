"""Reference solutions for exercise 11."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from fplh.models.fusion import Fusion


def sigmoid(x: float) -> float:
    return 1 / (1 + math.exp(-x))


def pooled_rate(lam_mkt: float, lam_mod: float, w: float, bias: float = 0.0) -> float:
    return math.exp(w * math.log(lam_mkt) + (1 - w) * math.log(lam_mod) + bias)


def fitted_weight(df: pd.DataFrame) -> float:
    f = Fusion().fit(df, ridge=0.01)
    return float(f.weight(np.array([24.0]), np.array([0.0]), np.array([True]))[0])
