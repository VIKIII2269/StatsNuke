from __future__ import annotations

import math

import numpy as np
import pytest

from fplh.evaluate import metrics as m


def test_log_loss_and_rps_by_hand() -> None:
    p = [[0.5, 0.3, 0.2], [0.2, 0.3, 0.5]]
    y = [0, 1]
    assert m.log_loss(p, y) == pytest.approx(-(math.log(0.5) + math.log(0.3)) / 2)
    # row 1: cum diffs (0.5-1, 0.8-1) → (0.25+0.04)/2 = 0.145
    # row 2: cum diffs (0.2, 0.5-1) → (0.04+0.25)/2 = 0.145
    assert m.rps(p, y) == pytest.approx(0.145)
    assert m.rps([[1, 0, 0]], [0]) == 0.0
    assert m.rps([[0, 0, 1]], [0]) == 1.0


def test_scoreline_grid_pools_remainder() -> None:
    grid = np.zeros((1, 7, 7))
    grid[0, 1, 0] = 0.9  # remainder 0.1
    assert m.scoreline_grid_log_loss(grid, [1], [0]) == pytest.approx(-math.log(0.9))
    assert m.scoreline_grid_log_loss(grid, [7], [0]) == pytest.approx(-math.log(0.1))


def test_brier_and_ece() -> None:
    assert m.brier([0.8, 0.3], [1, 0]) == pytest.approx((0.04 + 0.09) / 2)
    # bin 0.8: mean p 0.8, freq 0.5 → |0.3| × 2/4; bin 0.2: p 0.2, freq 0 → 0.2 × 2/4
    assert m.ece([0.8, 0.8, 0.2, 0.2], [1, 0, 0, 0]) == pytest.approx(0.25)


def test_crps_discrete_by_hand() -> None:
    # pmf over 0..2 = (0.5, 0.5, 0); y = 1 → F = (0.5, 1, 1), step = (0, 1, 1) → 0.25
    assert m.crps_discrete([[0.5, 0.5, 0.0]], [1]) == pytest.approx(0.25)
    assert m.crps_discrete([[0.0, 1.0, 0.0]], [1]) == 0.0


def test_pit_is_uniform_for_a_calibrated_forecast() -> None:
    rng = np.random.default_rng(0)
    from math import exp, factorial

    lam = 1.3
    y = rng.poisson(lam, 20000)

    def cdf(k: np.ndarray) -> np.ndarray:
        return np.array(
            [
                sum(exp(-lam) * lam**i / factorial(i) for i in range(int(v) + 1)) if v >= 0 else 0.0
                for v in k
            ]
        )

    u = m.randomised_pit(cdf(y - 1), cdf(y), rng)
    hist, _ = np.histogram(u, bins=10, range=(0, 1))
    assert np.all(np.abs(hist / len(u) - 0.1) < 0.012)


def test_point_metrics_and_spearman() -> None:
    assert m.mae([1, 2], [2, 4]) == 1.5
    assert m.rmse([0, 0], [3, 4]) == pytest.approx(math.sqrt(12.5))
    rho = m.spearman_within([1, 2, 3, 3, 2, 1], [1, 2, 3, 1, 2, 3], ["A", "A", "A", "B", "B", "B"])
    assert rho == pytest.approx(0.0)
    assert m.outcome_index([2, 1, 0], [1, 1, 3]).tolist() == [0, 1, 2]
