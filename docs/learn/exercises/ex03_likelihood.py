"""Exercise 03: likelihood, Newton's method, a Poisson GLM and ridge.

Run: uv run python docs/learn/exercises/ex03_likelihood.py
"""

from __future__ import annotations

import numpy as np
from _check import close, run, task
from scipy.special import gammaln

# ------------------------------------------------------------------ demo
goals = np.array([2, 0, 1, 3, 1])
grid = np.linspace(0.5, 3, 6)
ll = [float(np.sum(goals * np.log(lam) - lam - gammaln(goals + 1))) for lam in grid]
print("log-likelihood over λ:", dict(zip(np.round(grid, 2), np.round(ll, 3), strict=True)))
print("the peak is at the sample mean:", goals.mean())

# simulated league: home teams score more (true log home advantage 0.25)
rng = np.random.default_rng(3)
n = 4000
home = rng.integers(0, 2, n)
true_beta = np.array([0.15, 0.25])  # intercept, home effect (log scale)
X = np.column_stack([np.ones(n), home])
y = rng.poisson(np.exp(X @ true_beta))


# ------------------------------------------------------------------ your tasks
def poisson_loglik(lam: float, y: np.ndarray) -> float:
    """Σ_i [y_i log λ − λ − log y_i!]  (use scipy.special.gammaln for log y!)."""
    raise NotImplementedError


def newton_log_rate(y: np.ndarray, eta0: float = 0.0, steps: int = 20) -> float:
    """MLE of η = log λ by Newton: gradient Σy − n e^η, second derivative −n e^η.
    Return η after convergence."""
    raise NotImplementedError


def glm_nll_and_grad(beta: np.ndarray, X: np.ndarray, y: np.ndarray) -> tuple[float, np.ndarray]:
    """Negative log-likelihood (dropping log y!) and its gradient for y ~ Poisson(exp(X β)).
    Gradient of the NLL: −Xᵀ (y − μ)."""
    raise NotImplementedError


def fit_glm(X: np.ndarray, y: np.ndarray, ridge: float = 0.0) -> np.ndarray:
    """Minimise NLL + ridge · Σ β_j² over j ≥ 1 (do not penalise the intercept), using
    scipy.optimize.minimize(..., jac=True, method="L-BFGS-B") and glm_nll_and_grad."""
    raise NotImplementedError


@task("poisson_loglik peaks at the sample mean")
def _() -> None:
    vals = [poisson_loglik(lam, goals) for lam in (1.2, 1.4, 1.6)]
    assert vals[1] > vals[0], vals
    assert vals[1] > vals[2], vals
    close(poisson_loglik(1.4, goals), np.sum(goals * np.log(1.4) - 1.4 - gammaln(goals + 1)))


@task("newton_log_rate converges to log(mean)")
def _() -> None:
    close(newton_log_rate(goals), np.log(1.4), tol=1e-8)


@task("glm gradient matches finite differences")
def _() -> None:
    b = np.array([0.1, -0.2])
    f0, g = glm_nll_and_grad(b, X, y)
    eps = 1e-6
    fd = [(glm_nll_and_grad(b + eps * e, X, y)[0] - f0) / eps for e in np.eye(2)]
    close(g, fd, tol=1e-2)


@task("fit_glm recovers the home advantage (±0.05)")
def _() -> None:
    beta = fit_glm(X, y)
    close(beta, true_beta, tol=0.05)


@task("ridge shrinks the home effect toward 0")
def _() -> None:
    free, shrunk = fit_glm(X, y), fit_glm(X, y, ridge=500.0)
    assert abs(shrunk[1]) < abs(free[1]), (free, shrunk)


run(globals())
