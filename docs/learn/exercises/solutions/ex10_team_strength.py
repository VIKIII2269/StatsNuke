"""Reference solutions for exercise 10."""

from __future__ import annotations

import math

import pandas as pd

from fplh.models.team_strength import TeamStrengthParams, run_filter


def kalman_1d(m: float, v: float, y: float, r: float) -> tuple[float, float]:
    k = v / (v + r)
    return m + k * (y - m), (1 - k) * v


def propagate(
    m: float, v: float, phi_year: float, sigma: float, days: float
) -> tuple[float, float]:
    f = phi_year ** (days / 365)
    return f * m, f * f * v + sigma**2 * days / 365


def laplace_poisson(eta0: float, s2: float, y: int, steps: int = 30) -> tuple[float, float]:
    eta = eta0
    for _ in range(steps):
        grad = -(eta - eta0) / s2 + y - math.exp(eta)
        curv = 1 / s2 + math.exp(eta)
        eta += grad / curv
    return eta, 1 / (1 / s2 + math.exp(eta))


def home_multiplier(eta: float) -> float:
    return math.exp(eta)


def attack_order(matches: pd.DataFrame) -> list[str]:
    teams = sorted(set(matches["home_team"]) | set(matches["away_team"]))
    season = str(matches["season"].iloc[0])
    r = run_filter(matches, TeamStrengthParams(), {season: teams}).ratings()
    return r.sort_values("attack", ascending=False)["team"].tolist()
