"""Exercise 10: Kalman, propagation, a Laplace update, and the repo's M1 filter.

Run: uv run python docs/learn/exercises/ex10_team_strength.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from _check import close, run, task
from scipy.special import gammaln

from fplh.models.team_strength import TeamStrengthParams, run_filter

# ------------------------------------------------------------------ demo: a simulated league
rng = np.random.default_rng(10)
TEAMS = ["A", "B", "C", "D", "E", "F"]
ATTACK = np.array([0.4, 0.2, 0.0, 0.0, -0.2, -0.4])
DEFENCE = np.array([0.3, 0.1, 0.0, 0.0, -0.1, -0.3])
rows, t0 = [], pd.Timestamp("2020-08-01", tz="UTC")
for r in range(60):
    perm = rng.permutation(6)
    for k in range(3):
        h, a = perm[2 * k], perm[2 * k + 1]
        lh = np.exp(0.25 + 0.2 + ATTACK[h] - DEFENCE[a])
        la = np.exp(0.25 + ATTACK[a] - DEFENCE[h])
        rows.append(
            {
                "season": "2020-21",
                "kickoff_at": t0 + pd.Timedelta(days=7 * r, hours=k),
                "home_team": TEAMS[h],
                "away_team": TEAMS[a],
                "home_goals": rng.poisson(lh),
                "away_goals": rng.poisson(la),
                "home_xg": rng.gamma(16, lh / 16),
                "away_xg": rng.gamma(16, la / 16),
            }
        )
MATCHES = pd.DataFrame(rows)
f = run_filter(MATCHES, TeamStrengthParams(), {"2020-21": TEAMS})
print(f.ratings()[["team", "attack", "defence", "attack_sd"]].round(3).to_string(index=False))


# ------------------------------------------------------------------ your tasks
def kalman_1d(m: float, v: float, y: float, r: float) -> tuple[float, float]:
    """Posterior (mean, variance) for prior N(m, v) and observation y = θ + N(0, r)."""
    raise NotImplementedError


def propagate(
    m: float, v: float, phi_year: float, sigma: float, days: float
) -> tuple[float, float]:
    """Mean-reverting random walk over `days`:
    f = φ^(days/365); m' = f·m; v' = f²·v + σ²·days/365."""
    raise NotImplementedError


def laplace_poisson(eta0: float, s2: float, y: int, steps: int = 30) -> tuple[float, float]:
    """Prior η ~ N(eta0, s2); y ~ Poisson(e^η). Newton on the log-posterior
    −(η−eta0)²/(2 s2) + y η − e^η. Return (mode, 1 / negative curvature at the mode)."""
    raise NotImplementedError


def attack_order(matches: pd.DataFrame) -> list[str]:
    """Run fplh's run_filter with default TeamStrengthParams on `matches` (one season,
    teams TEAMS) and return the teams sorted from best to worst filtered attack."""
    raise NotImplementedError


def home_multiplier(eta: float) -> float:
    """How many times more goals the home side scores than the same team away (all else equal)."""
    raise NotImplementedError


@task("kalman_1d: the worked example")
def _() -> None:
    close(kalman_1d(0.0, 0.04, 0.5, 0.36), [0.05, 0.036])


@task("propagate: decay the mean, grow the variance")
def _() -> None:
    m, v = propagate(0.3, 0.01, 0.8, 0.2, 365)
    close([m, v], [0.24, 0.64 * 0.01 + 0.04])


@task("laplace_poisson mode matches a brute-force posterior")
def _() -> None:
    grid = np.linspace(-3, 3, 20001)
    eta0, s2, y = 0.2, 0.09, 4
    logp = -((grid - eta0) ** 2) / (2 * s2) + y * grid - np.exp(grid) - gammaln(y + 1)
    mode, var = laplace_poisson(eta0, s2, y)
    close(mode, grid[np.argmax(logp)], tol=1e-3)
    w = np.exp(logp - logp.max())
    w /= w.sum()
    mean = float(w @ grid)
    close(var, float(w @ (grid - mean) ** 2), tol=0.002)  # Laplace ≈ the true variance


@task("the filter recovers the extremes: A best, F worst attack")
def _() -> None:
    order = attack_order(MATCHES)
    assert order[0] == "A", order
    assert order[-1] == "F", order


@task("home_multiplier for the fitted η = 0.199 is about 1.22")
def _() -> None:
    close(home_multiplier(0.199), 1.2202, tol=1e-3)


run(globals())
