from __future__ import annotations

import numpy as np
import pandas as pd
from hypothesis import given, settings
from hypothesis import strategies as st

from fplh.rules import load_rules, score
from fplh.rules.bonus import assign_bonus, assign_bonus_array
from fplh.rules.engine import COMPONENTS

RULES = load_rules("2026/27")

event_row = st.fixed_dictionaries(
    {
        "position": st.sampled_from(["GK", "DEF", "MID", "FWD"]),
        "minutes": st.integers(0, 90),
        "goals_scored": st.integers(0, 4),
        "assists": st.integers(0, 3),
        "goals_conceded": st.integers(0, 7),
        "own_goals": st.integers(0, 1),
        "penalties_saved": st.integers(0, 1),
        "penalties_missed": st.integers(0, 1),
        "yellow_cards": st.integers(0, 1),
        "red_cards": st.integers(0, 1),
        "saves": st.integers(0, 12),
        "bonus": st.integers(0, 3),
        "clearances_blocks_interceptions": st.integers(0, 20),
        "tackles": st.integers(0, 10),
        "recoveries": st.integers(0, 15),
    }
)
frames = st.lists(event_row, min_size=1, max_size=30).map(pd.DataFrame)


@given(frames, st.randoms())
def test_row_order_does_not_matter(df: pd.DataFrame, rnd: object) -> None:
    perm = np.random.default_rng(0).permutation(len(df))
    a = score(df, RULES)["total"].iloc[perm].to_numpy()
    b = score(df.iloc[perm], RULES)["total"].to_numpy()
    np.testing.assert_array_equal(a, b)


@given(frames, frames)
def test_rows_score_independently(a: pd.DataFrame, b: pd.DataFrame) -> None:
    """Scoring a concatenation (e.g. both DGW fixtures) = concatenating the scores."""
    both = score(pd.concat([a, b], ignore_index=True), RULES)["total"].to_numpy()
    sep = np.concatenate([score(a, RULES)["total"], score(b, RULES)["total"]])
    np.testing.assert_array_equal(both, sep)


@given(frames)
def test_components_sum_and_appearance_bounds(df: pd.DataFrame) -> None:
    out = score(df, RULES)
    assert (out[list(COMPONENTS)].sum(axis=1) == out["total"]).all()
    assert out["appearance"].isin([0, 1, 2]).all()
    assert ((out["appearance"] == 0) == (df["minutes"] == 0)).all()
    assert (out["clean_sheet"][df["minutes"] < 60] == 0).all()


@given(frames)
def test_extra_goal_adds_position_value(df: pd.DataFrame) -> None:
    plus = df.assign(goals_scored=df["goals_scored"] + 1)
    delta = score(plus, RULES)["total"] - score(df, RULES)["total"]
    expected = df["position"].map(RULES.goal)
    assert (delta == expected).all()


@settings(max_examples=200)
@given(st.lists(st.integers(-5, 80), min_size=1, max_size=30))
def test_bonus_properties(bps: list[int]) -> None:
    s = pd.Series(bps)
    award = assign_bonus(s, pd.Series([1] * len(bps)))
    assert award.isin([0, 1, 2, 3]).all()
    top = s == s.max()
    assert (award[top] == 3).all()  # everyone tied for the top gets 3
    np.testing.assert_array_equal(award.to_numpy(), assign_bonus_array(np.array(bps)))
    # more BPS never means less bonus
    order = np.argsort(bps)
    assert (np.diff(award.to_numpy()[order]) >= 0).all()


@settings(max_examples=200)
@given(st.lists(st.tuples(st.integers(-5, 80), st.booleans()), min_size=1, max_size=30))
def test_bonus_mask_matches_pandas(rows: list[tuple[int, bool]]) -> None:
    bps = pd.Series([b for b, _ in rows])
    played = pd.Series([p for _, p in rows])
    expected = assign_bonus(bps, pd.Series([1] * len(rows)), eligible=played)
    got = assign_bonus_array(bps.to_numpy(), eligible=played.to_numpy())
    np.testing.assert_array_equal(expected.to_numpy(), got)
