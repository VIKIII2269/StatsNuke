from __future__ import annotations

import numpy as np
import pandas as pd
from hypothesis import given, settings
from hypothesis import strategies as st

from fplh.features.trailing import trailing_means

T0 = pd.Timestamp("2030-08-01", tz="UTC")


def brute(history: pd.DataFrame, queries: pd.DataFrame, k: int) -> list[float]:
    out = []
    for q in queries.itertuples():
        h = history[(history["key"] == q.key) & (history["observed_at"] <= q.deadline_at)]
        h = h.sort_values(["observed_at", "kickoff_at"], kind="mergesort")["x"].dropna()
        tail = history.loc[h.index].sort_values(["observed_at", "kickoff_at"], kind="mergesort")
        vals = (
            tail["x"].iloc[-k:]
            if False
            else history[(history["key"] == q.key) & (history["observed_at"] <= q.deadline_at)]
            .sort_values(["observed_at", "kickoff_at"], kind="mergesort")["x"]
            .iloc[-k:]
        )
        vals = vals.dropna()
        out.append(float(vals.mean()) if len(vals) else np.nan)
    return out


@settings(max_examples=30, deadline=None)
@given(
    st.lists(
        st.tuples(
            st.sampled_from("abc"), st.integers(0, 60), st.one_of(st.none(), st.floats(-5, 5))
        ),
        min_size=0,
        max_size=40,
    ),
    st.lists(st.tuples(st.sampled_from("abcd"), st.integers(0, 70)), min_size=1, max_size=10),
    st.integers(1, 6),
)
def test_matches_brute_force(
    rows: list[tuple[str, int, float | None]], qs: list[tuple[str, int]], k: int
) -> None:
    history = pd.DataFrame(
        {
            "key": pd.Series([r[0] for r in rows], dtype=object),
            "kickoff_at": pd.Series(
                [T0 + pd.Timedelta(days=r[1]) for r in rows], dtype="datetime64[ns, UTC]"
            ),
            "x": pd.Series([np.nan if r[2] is None else r[2] for r in rows], dtype=float),
        }
    )
    history["observed_at"] = history["kickoff_at"] + pd.Timedelta(hours=33)
    queries = pd.DataFrame(
        {"key": [q[0] for q in qs], "deadline_at": [T0 + pd.Timedelta(days=q[1]) for q in qs]},
        index=range(100, 100 + len(qs)),
    )
    got = trailing_means(history, "key", ["x"], [k], queries)
    assert list(got.index) == list(queries.index)
    expected = brute(history, queries, k)
    np.testing.assert_allclose(got[f"x_{k}"].to_numpy(), expected, atol=1e-9, equal_nan=True)
    counts = [
        int(((history["key"] == q.key) & (history["observed_at"] <= q.deadline_at)).sum())
        for q in queries.itertuples()
    ]
    assert got["n"].tolist() == counts


def test_observation_time_not_kickoff_decides_visibility() -> None:
    history = pd.DataFrame(
        {
            "key": ["p", "p"],
            "kickoff_at": [T0, T0 + pd.Timedelta(days=3)],
            "observed_at": [T0 + pd.Timedelta(hours=33), T0 + pd.Timedelta(days=3, hours=33)],
            "x": [1.0, 5.0],
        }
    )
    # a deadline one day after the second kickoff cannot see it yet
    q = pd.DataFrame({"key": ["p"], "deadline_at": [T0 + pd.Timedelta(days=4)]})
    got = trailing_means(history, "key", ["x"], [3], q, prefix="pm_")
    assert got.loc[0, "pm_x_3"] == 1.0 and got.loc[0, "pm_n"] == 1
