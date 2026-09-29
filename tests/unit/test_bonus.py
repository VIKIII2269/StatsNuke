from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fplh.rules.bonus import assign_bonus, assign_bonus_array


@pytest.mark.parametrize(
    ("bps", "expected"),
    [
        ([40, 30, 20, 10], [3, 2, 1, 0]),
        ([40, 40, 30, 10], [3, 3, 1, 0]),  # tie for 1st
        ([40, 30, 30, 10], [3, 2, 2, 0]),  # tie for 2nd
        ([40, 30, 20, 20], [3, 2, 1, 1]),  # tie for 3rd
        ([40, 40, 40, 10], [3, 3, 3, 0]),  # three-way tie for 1st
        ([5], [3]),
    ],
)
def test_official_tie_rule(bps: list[int], expected: list[int]) -> None:
    s = pd.Series(bps)
    got = assign_bonus(s, pd.Series([1] * len(bps)))
    assert got.tolist() == expected
    assert assign_bonus_array(np.array(bps)).tolist() == expected


def test_ranked_within_fixture() -> None:
    bps = pd.Series([10, 50, 20, 5])
    fixture = pd.Series([1, 2, 1, 2])
    assert assign_bonus(bps, fixture).tolist() == [2, 3, 3, 2]


def test_ineligible_rows_excluded() -> None:
    bps = pd.Series([30, 20, 10, 0, 0])
    played = pd.Series([True, True, False, True, True])
    # the unused sub is skipped; 0-BPS players who played tie for third
    assert assign_bonus(bps, pd.Series([1] * 5), eligible=played).tolist() == [3, 2, 0, 1, 1]


def test_array_version_batches_simulations() -> None:
    bps = np.array([[40, 30, 20], [10, 10, 30]])
    np.testing.assert_array_equal(assign_bonus_array(bps), [[3, 2, 1], [2, 2, 3]])
