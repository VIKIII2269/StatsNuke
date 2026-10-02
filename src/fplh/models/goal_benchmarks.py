"""Closed-form joint scoreline models G0–G3 (ARCHITECTURE.md §7.5, §11.7).

Every model maps expected goals ``(λ_home, λ_away)`` plus its dependence parameters to a
normalised scoreline grid over 0..MAX_GOALS. The grid gives 1X2 / totals probabilities,
drives market inversion, and is scored by scoreline-grid log loss.

* G0 independent Poisson.
* G1 Dixon–Coles: τ-adjusted low scores, parameter ρ.
* G2 bivariate Poisson with diagonal inflation (Karlis & Ntzoufras 2003): shared
  component λ₃ and inflation weight π on a Poisson(θ) diagonal. Marginal means are held
  at (λ_home, λ_away) by setting λ₁ = λ_home − λ₃, λ₂ = λ_away − λ₃.
* G3 Conway–Maxwell–Poisson marginals, dispersion ν (ν > 1 under-, ν < 1 over-dispersed),
  each rate chosen so the marginal mean equals the target.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt
from scipy.optimize import brentq, minimize
from scipy.special import gammaln
from scipy.stats import poisson

MAX_GOALS = 10
K = np.arange(MAX_GOALS + 1)
Grid = npt.NDArray[np.float64]


def _normalise(g: Grid) -> Grid:
    out: Grid = g / g.sum()
    return out


def g0_grid(lh: float, la: float) -> Grid:
    return _normalise(np.outer(poisson.pmf(K, lh), poisson.pmf(K, la)))


def g1_grid(lh: float, la: float, rho: float) -> Grid:
    g = np.outer(poisson.pmf(K, lh), poisson.pmf(K, la))
    g[0, 0] *= 1 - lh * la * rho
    g[0, 1] *= 1 + lh * rho
    g[1, 0] *= 1 + la * rho
    g[1, 1] *= 1 - rho
    return _normalise(np.clip(g, 1e-300, None))


def _bivariate_poisson(l1: float, l2: float, l3: float) -> Grid:
    x = K[:, None]
    y = K[None, :]
    base = -(l1 + l2 + l3) + x * math.log(l1) - gammaln(x + 1) + y * math.log(l2) - gammaln(y + 1)
    s = np.zeros((len(K), len(K)))
    ratio = l3 / (l1 * l2)
    for k in range(len(K)):
        # C(x,k)·C(y,k)·k! = x!·y! / (k!·(x−k)!·(y−k)!)
        ok = (x >= k) & (y >= k)
        log_comb = (
            gammaln(x + 1)
            + gammaln(y + 1)
            - gammaln(k + 1)
            - gammaln(np.maximum(x - k, 0) + 1)
            - gammaln(np.maximum(y - k, 0) + 1)
        )
        s += np.where(ok, np.exp(log_comb) * ratio**k, 0.0)
    out: Grid = np.exp(base) * s
    return out


def g2_grid(lh: float, la: float, lam3: float, pi: float, theta: float) -> Grid:
    l3 = min(lam3, 0.95 * min(lh, la))
    bp = _bivariate_poisson(max(lh - l3, 1e-6), max(la - l3, 1e-6), max(l3, 1e-12))
    bp = _normalise(bp)
    diag = np.zeros_like(bp)
    q = poisson.pmf(K, theta)
    diag[K, K] = q / q.sum()
    return _normalise((1 - pi) * bp + pi * diag)


def _com_pmf(lam: float, nu: float) -> npt.NDArray[np.float64]:
    logp = K * math.log(lam) - nu * gammaln(K + 1)
    logp -= logp.max()
    p = np.exp(logp)
    out: npt.NDArray[np.float64] = p / p.sum()
    return out


def _com_rate_for_mean(mean: float, nu: float) -> float:
    root = brentq(lambda log_lam: float(_com_pmf(math.exp(log_lam), nu) @ K) - mean, -12.0, 12.0)
    return math.exp(root)


def g3_grid(lh: float, la: float, nu: float) -> Grid:
    ph = _com_pmf(_com_rate_for_mean(lh, nu), nu)
    pa = _com_pmf(_com_rate_for_mean(la, nu), nu)
    return _normalise(np.outer(ph, pa))


# ---------------------------------------------------------------------------------------


def markets(grid: Grid, line: float = 2.5) -> dict[str, float]:
    total = K[:, None] + K[None, :]
    return {
        "home": float(np.tril(grid, -1).sum()),
        "draw": float(np.trace(grid)),
        "away": float(np.triu(grid, 1).sum()),
        "over": float(grid[total > line].sum()),
        "btts": float(grid[1:, 1:].sum()),
    }


@dataclass
class GoalModel:
    """A scoreline model: name, parameter names, initial values and bounds."""

    name: str
    grid_fn: Callable[..., Grid]
    params: dict[str, float] = field(default_factory=dict)
    bounds: dict[str, tuple[float, float]] = field(default_factory=dict)

    def grid(self, lh: float, la: float) -> Grid:
        return self.grid_fn(lh, la, **self.params)

    def log_prob(
        self,
        lh: npt.ArrayLike,
        la: npt.ArrayLike,
        hg: npt.ArrayLike,
        ag: npt.ArrayLike,
        params: dict[str, float] | None = None,
    ) -> npt.NDArray[np.float64]:
        """log P(score) per match; vectorised for G0/G1 (untruncated, < 1e-6 from the grid)."""
        p = self.params if params is None else params
        lh_, la_ = np.asarray(lh, float), np.asarray(la, float)
        h, a = (
            np.minimum(np.asarray(hg, int), MAX_GOALS),
            np.minimum(np.asarray(ag, int), MAX_GOALS),
        )
        if self.grid_fn in (g0_grid, g1_grid):
            out = poisson.logpmf(h, lh_) + poisson.logpmf(a, la_)
            if self.grid_fn is g1_grid:
                rho = p["rho"]
                tau = np.ones_like(lh_)
                tau = np.where((h == 0) & (a == 0), 1 - lh_ * la_ * rho, tau)
                tau = np.where((h == 0) & (a == 1), 1 + lh_ * rho, tau)
                tau = np.where((h == 1) & (a == 0), 1 + la_ * rho, tau)
                tau = np.where((h == 1) & (a == 1), 1 - rho, tau)
                out = out + np.log(np.clip(tau, 1e-300, None))
            result: npt.NDArray[np.float64] = out
            return result
        return np.array(
            [
                math.log(max(self.grid_fn(x, y, **p)[i, j], 1e-300))
                for x, y, i, j in zip(lh_, la_, h, a, strict=True)
            ]
        )

    def fit(
        self, lh: npt.ArrayLike, la: npt.ArrayLike, hg: npt.ArrayLike, ag: npt.ArrayLike
    ) -> GoalModel:
        """Maximum-likelihood dependence parameters given per-match rates and scores."""
        if not self.params:
            return self
        names = list(self.params)

        def nll(x: npt.NDArray[np.float64]) -> float:
            return -float(np.sum(self.log_prob(lh, la, hg, ag, dict(zip(names, x, strict=True)))))

        res = minimize(
            nll,
            x0=np.array([self.params[n] for n in names]),
            method="L-BFGS-B",
            bounds=[self.bounds[n] for n in names],
        )
        return GoalModel(
            self.name, self.grid_fn, dict(zip(names, map(float, res.x), strict=True)), self.bounds
        )

    def invert(self, p1x2: Sequence[float], p_over: float | None = None) -> tuple[float, float]:
        """Rates (λ_home, λ_away) whose grid best reproduces the market probabilities."""
        target = np.asarray(p1x2, dtype=float)

        def loss(x: npt.NDArray[np.float64]) -> float:
            m = markets(self.grid(math.exp(x[0]), math.exp(x[1])))
            err = (
                (m["home"] - target[0]) ** 2
                + (m["draw"] - target[1]) ** 2
                + (m["away"] - target[2]) ** 2
            )
            if p_over is not None:
                err += (m["over"] - p_over) ** 2
            return float(err)

        res = minimize(
            loss,
            x0=np.log([1.4, 1.1]),
            method="L-BFGS-B",
            bounds=[(-3.0, 2.0), (-3.0, 2.0)],
            options={"ftol": 1e-15, "gtol": 1e-12},
        )
        return float(math.exp(res.x[0])), float(math.exp(res.x[1]))


def benchmarks() -> dict[str, GoalModel]:
    """G0–G3 with neutral starting parameters (fit before use)."""
    return {
        "G0": GoalModel("G0", g0_grid),
        "G1": GoalModel("G1", g1_grid, {"rho": 0.0}, {"rho": (-0.3, 0.3)}),
        "G2": GoalModel(
            "G2",
            g2_grid,
            {"lam3": 0.05, "pi": 0.02, "theta": 1.0},
            {"lam3": (0.0, 0.5), "pi": (0.0, 0.3), "theta": (0.05, 3.0)},
        ),
        "G3": GoalModel("G3", g3_grid, {"nu": 1.0}, {"nu": (0.5, 2.0)}),
    }
