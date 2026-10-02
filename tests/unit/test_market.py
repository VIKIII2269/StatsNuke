from __future__ import annotations

import numpy as np
import pytest

from fplh.models.market import devig, devig_multiplicative, devig_power, devig_shin, shin_z

BOOKS = [[1.8, 3.8, 4.5], [1.25, 6.5, 13.0], [2.9, 3.3, 2.6], [1.05, 15.0, 41.0]]


@pytest.mark.parametrize("odds", BOOKS)
@pytest.mark.parametrize("method", ["multiplicative", "power", "shin"])
def test_probabilities_sum_to_one(odds: list[float], method: str) -> None:
    p = devig(odds, method)
    assert p.sum() == pytest.approx(1.0, abs=1e-9)
    assert (p > 0).all()
    assert np.argsort(p).tolist() == np.argsort(1 / np.array(odds)).tolist()  # order preserved


@pytest.mark.parametrize("odds", BOOKS)
def test_power_and_shin_shade_longshots(odds: list[float]) -> None:
    mult, power, shin = devig_multiplicative(odds), devig_power(odds), devig_shin(odds)
    fav, long = int(np.argmax(mult)), int(np.argmin(mult))
    assert power[fav] >= mult[fav] and power[long] <= mult[long]
    assert shin[fav] >= mult[fav] and shin[long] <= mult[long]


def test_shin_z_is_a_share() -> None:
    for odds in BOOKS:
        assert 0.0 <= shin_z(odds) < 1.0
    assert shin_z([2.0, 2.0]) == 0.0  # no margin → no informed money


def test_fair_book_is_unchanged() -> None:
    fair = [2.0, 4.0, 4.0]
    for method in ("multiplicative", "power", "shin"):
        assert devig(fair, method) == pytest.approx([0.5, 0.25, 0.25])


def test_devig_calibration_prefers_the_generating_method() -> None:
    """Books built as p_true^0.9 (scaled to a 6 % margin) favour longshots exactly as
    power de-vig assumes, so power must calibrate at least as well as multiplicative."""
    from fplh.models.market import devig_calibration

    rng = np.random.default_rng(0)
    rows, ys = [], []
    for _ in range(3000):
        true = rng.dirichlet([4, 2.5, 3])
        implied = true**0.9
        rows.append(1 / (implied / implied.sum() * 1.06))
        ys.append(rng.choice(3, p=true))
    loss = devig_calibration(np.array(rows), np.array(ys))
    assert set(loss) == {"multiplicative", "power", "shin"}
    assert loss["power"] <= loss["multiplicative"]
