"""Exercise 05: observation times, leak-free features, as-of joins and entity links.

Run: uv run python docs/learn/exercises/ex05_data.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from _check import run, task
from rapidfuzz.fuzz import token_set_ratio

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # repo root, for tests.*
from fplh.entities.teams import normalise_name
from fplh.features.information_set import InformationSet, SilverStore
from fplh.features.leakage import check_leakage
from fplh.features.spine import SPINE_KEYS, asof_join
from tests.synthetic_silver import deadlines, make

# ------------------------------------------------------------------ demo
frames = make()
ds = deadlines(frames)
info = InformationSet.at(ds[2], SilverStore.from_frames(frames))
pm = info.table("fact_player_match")
print(
    f"deadline {ds[2]}: {len(pm)} player-match rows visible, latest observed {info.max_observed_at}"
)
print("dim_fixture columns exposed:", list(info.table("dim_fixture").columns))
print(normalise_name("Martin Ødegaard"), "|", token_set_ratio("martin odegaard", "odegaard"))


def leaky(info: InformationSet, spine: pd.DataFrame) -> pd.DataFrame:
    """Reaches past the information set: sees the whole season."""
    all_rows = info._store.get("fact_player_match")
    goals = all_rows.groupby("player_uid")["goals_scored"].sum().rename("goals")
    return spine[SPINE_KEYS].merge(goals, left_on="player_uid", right_index=True, how="left")


print("leaky builder problems:", len(check_leakage(frames, ds[1:3], {"leaky": leaky})))


# ------------------------------------------------------------------ your tasks
def observation_time(event_at: pd.Series, lag_hours: float) -> pd.Series:
    """o_f = e_f + ℓ_s for backfilled facts."""
    raise NotImplementedError


def visible(df: pd.DataFrame, deadline: pd.Timestamp) -> pd.DataFrame:
    """Rows of a time-indexed table knowable at the deadline (observed_at ≤ D)."""
    raise NotImplementedError


def goals_so_far(info: InformationSet, spine: pd.DataFrame) -> pd.DataFrame:
    """Feature builder: each spine player's total goals_scored in fact_player_match visible at
    the deadline, as column 'goals_so_far' (0 if none). Return SPINE_KEYS + goals_so_far.
    Use info.table(...) only."""
    raise NotImplementedError


def latest_price(spine: pd.DataFrame, snaps: pd.DataFrame) -> pd.Series:
    """As-of join by hand: for each spine row (player_uid, deadline_at), the price of the latest
    snapshot with observed_at ≤ deadline_at (NaN if none). Hint: pd.merge_asof with by=,
    left_on="deadline_at", right_on="observed_at", both sorted by time. Return a Series
    aligned with spine's row order."""
    raise NotImplementedError


def link_decision(score: float, overlap: float, shared: int) -> str:
    """Return 'auto', 'review' or 'none' following entities/players.py:
    score ≥ 92 and overlap ≥ 0.8 → auto; 80 ≤ score < 92 and overlap ≥ 0.95 and shared ≥ 3 → auto;
    other score ≥ 80 → review; else none."""
    raise NotImplementedError


@task("observation_time adds the publication lag")
def _() -> None:
    e = pd.Series(pd.to_datetime(["2024-10-05 15:00"], utc=True))
    assert observation_time(e, 24).iloc[0] == pd.Timestamp("2024-10-06 15:00", tz="UTC")


@task("visible filters to observed_at ≤ D")
def _() -> None:
    d = ds[3]
    v = visible(frames["fact_player_match"], d)
    assert (v["observed_at"] <= d).all()
    assert len(v) == len(
        InformationSet.at(d, SilverStore.from_frames(frames)).table("fact_player_match")
    )


@task("goals_so_far passes check_leakage at every deadline")
def _() -> None:
    assert check_leakage(frames, ds[1:5], {"mine": goals_so_far}) == []


@task("goals_so_far is not constant zero (it really counts)")
def _() -> None:
    i = InformationSet.at(ds[-1], SilverStore.from_frames(frames))
    from fplh.features.spine import build_spine

    f = goals_so_far(i, build_spine(i, 1))
    assert f["goals_so_far"].sum() > 0
    assert list(f.columns[:4]) == SPINE_KEYS


@task("latest_price equals the repo's DuckDB asof_join")
def _() -> None:
    t = pd.to_datetime
    spine = pd.DataFrame(
        {
            "player_uid": ["p7", "p7", "p8"],
            "deadline_at": t(
                ["2024-10-05 11:00", "2024-10-01 11:00", "2024-10-05 11:00"], utc=True
            ),
        }
    )
    snaps = pd.DataFrame(
        {
            "player_uid": ["p7", "p7", "p7", "p8"],
            "observed_at": t(
                ["2024-10-03 09:00", "2024-10-05 08:00", "2024-10-05 14:00", "2024-10-06 00:00"],
                utc=True,
            ),
            "price": [75, 76, 77, 50],
        }
    )
    want = asof_join(spine, snaps, ["player_uid"], ["price"])["price"].astype(float)
    got = pd.Series(latest_price(spine, snaps)).astype(float).reset_index(drop=True)
    pd.testing.assert_series_equal(got, want, check_names=False)
    assert got.iloc[0] == 76  # the 14:00 snapshot is after the 11:00 deadline


@task("link_decision follows the rules")
def _() -> None:
    cases = [
        ((95, 0.85, 10), "auto"),
        ((95, 0.5, 10), "review"),
        ((85, 0.97, 3), "auto"),
        ((85, 0.97, 2), "review"),
        ((70, 1.0, 20), "none"),
    ]
    for args, want in cases:
        assert link_decision(*args) == want, (args, link_decision(*args))


run(globals())
