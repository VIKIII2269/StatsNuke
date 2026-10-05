"""Reference solutions for exercise 03."""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize
from scipy.special import gammaln


def poisson_loglik(lam: float, y: np.ndarray) -> float:
    return float(np.sum(y * np.log(lam) - lam - gammaln(y + 1)))


def newton_log_rate(y: np.ndarray, eta0: float = 0.0, steps: int = 20) -> float:
    eta, n, s = eta0, len(y), float(np.sum(y))
    for _ in range(steps):
        grad = s - n * np.exp(eta)
        hess = -n * np.exp(eta)
        eta = eta - grad / hess
    return float(eta)


def glm_nll_and_grad(beta: np.ndarray, X: np.ndarray, y: np.ndarray) -> tuple[float, np.ndarray]:
    eta = X @ beta
    mu = np.exp(eta)
    return float(-(y @ eta - mu.sum())), -(X.T @ (y - mu))


def fit_glm(X: np.ndarray, y: np.ndarray, ridge: float = 0.0) -> np.ndarray:
    def f(beta: np.ndarray) -> tuple[float, np.ndarray]:
        v, g = glm_nll_and_grad(beta, X, y)
        pen = np.r_[0.0, beta[1:]]
        return v + ridge * float(pen @ pen), g + 2 * ridge * pen

    return minimize(f, np.zeros(X.shape[1]), jac=True, method="L-BFGS-B").x
