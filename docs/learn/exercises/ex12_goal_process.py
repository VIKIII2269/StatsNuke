"""Exercise 12: hazards, the Poisson-GLM view, Gamma frailty and the emulator.

Run: uv run python docs/learn/exercises/ex12_goal_process.py
"""

from __future__ import annotations

import numpy as np
import yaml
from _check import close, run, task
from scipy import integrate
from scipy.stats import gamma, poisson

from fplh.models.goal_process import GoalProcessParams
from fplh.settings import get_settings
from fplh.sim.emulator import Emulator
from fplh.sim.team import simulate_team

doc = yaml.safe_load((get_settings().configs_dir / "models" / "goal_process.yaml").read_text())
G6 = GoalProcessParams.from_dict(doc["levels"]["G6"])

# ------------------------------------------------------------------ demo
sim = simulate_team(G6, np.array([[1.5, 1.1]]), 20_000, seed=0)
print(
    f"G6 at nominal (1.5, 1.1): mean goals {sim.home.mean():.3f}, {sim.away.mean():.3f}; "
    f"draws {(sim.home == sim.away).mean():.3f}"
)
print(f"own red ×{np.exp(G6.red_own):.2f}, opponent red ×{np.exp(G6.red_opp):.2f}")
EMU = Emulator.build(G6, grid=8, n_sims=4000)

# a simulated league where trailing teams score exp(0.4) ≈ 1.49× faster
rng = np.random.default_rng(12)
BETA_TRUE, BASE = 0.4, 1.3 / 90
n_matches = 3000
trailing_rows, goal_rows = [], []
for _ in range(n_matches):
    a = b = 0
    for _minute in range(90):
        trailing = a < b
        ga = rng.poisson(BASE * np.exp(BETA_TRUE * trailing))
        gb = rng.poisson(BASE * np.exp(BETA_TRUE * (b < a)))
        trailing_rows.append(trailing)
        goal_rows.append(ga)
        a, b = a + ga, b + gb
TRAILING = np.array(trailing_rows)
GOALS = np.array(goal_rows)


# ------------------------------------------------------------------ your tasks
def p_no_goal(hazards: np.ndarray, exposures: np.ndarray) -> float:
    """P(no event) under a piecewise-constant hazard: exp(−Σ h·E)."""
    raise NotImplementedError


def trailing_effect(goals: np.ndarray, base_rate: float, trailing: np.ndarray) -> float:
    """MLE of β in  goals_t ~ Poisson(base · exp(β · trailing_t))  with base known (an offset).
    Setting the derivative to zero gives exp(β) = Σ goals[trailing] / (base · #trailing)."""
    raise NotImplementedError


def frailty_loglik(y: np.ndarray, lam: np.ndarray, a: float) -> float:
    """log ∫ Π_i Pois(y_i; ε λ_i) · Gamma(ε; shape a, rate a) dε  in closed form:
    Σ y log λ − Σ log y! + lgamma(N + a) − lgamma(a) + a log a − (N + a) log(a + Λ),
    with N = Σ y and Λ = Σ λ."""
    raise NotImplementedError


def nominal_for(emu: Emulator, mu_home: float, mu_away: float) -> tuple[float, float]:
    """Nominal per-90 rates whose G6 mean goals are (mu_home, mu_away): use
    emu.nominal_for_means (it works in logs) and return the rates themselves."""
    raise NotImplementedError


@task("p_no_goal for a constant 1.2/90 over 90 minutes is e^−1.2")
def _() -> None:
    close(p_no_goal(np.full(90, 1.2 / 90), np.ones(90)), np.exp(-1.2))
    close(p_no_goal(np.array([0.01, 0.02]), np.array([10, 0.5])), np.exp(-0.11))


@task("trailing_effect recovers β = 0.4 (±0.1)")
def _() -> None:
    close(trailing_effect(GOALS, BASE, TRAILING), BETA_TRUE, tol=0.1)


@task("frailty_loglik matches numerical integration")
def _() -> None:
    y, lam, a = np.array([0, 1, 0, 2]), np.array([0.3, 0.4, 0.2, 0.6]), 4.0

    def integrand(e: float) -> float:
        return float(np.prod(poisson.pmf(y, e * lam)) * gamma.pdf(e, a, scale=1 / a))

    num, _ = integrate.quad(integrand, 0, 50)
    close(frailty_loglik(y, lam, a), np.log(num), tol=1e-6)


@task("nominal_for hits the target mean goals (±0.06)")
def _() -> None:
    nh, na = nominal_for(EMU, 1.6, 1.0)
    s = simulate_team(G6, np.array([[nh, na]]), 40_000, seed=3)
    close([s.home.mean(), s.away.mean()], [1.6, 1.0], tol=0.06)


run(globals())
