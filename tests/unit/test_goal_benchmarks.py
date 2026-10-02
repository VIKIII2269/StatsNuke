from __future__ import annotations

import numpy as np
import pytest

from fplh.models.goal_benchmarks import benchmarks, g0_grid, g1_grid, g2_grid, g3_grid, markets


@pytest.mark.parametrize(("lh", "la"), [(1.5, 1.1), (0.4, 2.8), (3.2, 0.3)])
def test_grids_are_normalised(lh: float, la: float) -> None:
    for g in (
        g0_grid(lh, la),
        g1_grid(lh, la, -0.1),
        g2_grid(lh, la, 0.1, 0.05, 1.0),
        g3_grid(lh, la, 1.2),
    ):
        assert g.sum() == pytest.approx(1.0)
        assert (g >= 0).all()


def test_neutral_parameters_reduce_to_poisson() -> None:
    base = g0_grid(1.6, 1.1)
    assert np.abs(g1_grid(1.6, 1.1, 0.0) - base).max() < 1e-12
    assert np.abs(g2_grid(1.6, 1.1, 0.0, 0.0, 1.0) - base).max() < 1e-9
    assert np.abs(g3_grid(1.6, 1.1, 1.0) - base).max() < 1e-5


def test_marginal_means_are_preserved() -> None:
    k = np.arange(11)
    for g in (g2_grid(1.6, 1.1, 0.2, 0.0, 1.0), g3_grid(1.6, 1.1, 1.3)):
        assert g.sum(axis=1) @ k == pytest.approx(1.6, abs=2e-3)
        assert g.sum(axis=0) @ k == pytest.approx(1.1, abs=2e-3)


def test_dependence_moves_draws() -> None:
    base = markets(g0_grid(1.3, 1.1))["draw"]
    assert markets(g1_grid(1.3, 1.1, -0.1))["draw"] > base  # ρ < 0 inflates low draws
    assert markets(g2_grid(1.3, 1.1, 0.0, 0.1, 1.0))["draw"] > base
    assert markets(g3_grid(1.3, 1.1, 1.3))["draw"] > base  # underdispersion → more draws


@pytest.mark.parametrize("name", ["G0", "G1", "G2", "G3"])
def test_inversion_reproduces_synthetic_prices(name: str) -> None:
    model = benchmarks()[name]
    lh, la = 1.72, 0.93
    mk = markets(model.grid(lh, la))
    got = model.invert([mk["home"], mk["draw"], mk["away"]], mk["over"])
    back = markets(model.grid(*got))
    for key in ("home", "draw", "away", "over"):
        assert back[key] == pytest.approx(mk[key], abs=1e-6)
    assert got == pytest.approx((lh, la), abs=1e-4)


def test_fit_recovers_dixon_coles_rho() -> None:
    rng = np.random.default_rng(0)
    n = 4000
    lh, la = rng.uniform(0.8, 2.0, n), rng.uniform(0.6, 1.6, n)
    hg, ag = np.empty(n, int), np.empty(n, int)
    for i in range(n):
        g = g1_grid(lh[i], la[i], -0.15).ravel()
        cell = rng.choice(g.size, p=g)
        hg[i], ag[i] = divmod(cell, 11)
    fitted = benchmarks()["G1"].fit(lh, la, hg, ag)
    assert fitted.params["rho"] == pytest.approx(-0.15, abs=0.06)
