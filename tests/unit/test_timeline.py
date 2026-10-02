from __future__ import annotations

import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from fplh.lake.silver.timeline import derive_lineups, match_events

KICKOFF = pd.Timestamp("2030-08-10 14:00", tz="UTC")


def row(
    rid: int,
    side: str,
    minutes: int,
    *,
    sub: bool = False,
    rin: int = 0,
    rout: int = 0,
    red: int = 0,
    uid: str | None = None,
) -> dict[str, object]:
    return {
        "season": "2030-31",
        "understat_match_id": 1,
        "fixture_uid": "2030-31:a:b",
        "roster_id": rid,
        "understat_player_id": 1000 + rid,
        "player_uid": uid or f"fpl:{rid}",
        "player_name": f"P{rid}",
        "side": side,
        "position": "Sub" if sub else "MC",
        "minutes": minutes,
        "red_cards": red,
        "roster_in": rin,
        "roster_out": rout,
        "event_at": KICKOFF,
        "observed_at": KICKOFF + pd.Timedelta(hours=24),
    }


def roster() -> pd.DataFrame:
    h = [
        row(1, "h", 60, rin=12),  # replaced at 60
        row(2, "h", 50, rin=13),  # replaced at 50 by 13, who is replaced at 80 by 14
        row(3, "h", 33, red=1),  # sent off at 33
        row(4, "h", 90, red=1),  # red at or after 90
        row(5, "h", 90, rin=15),  # replaced in stoppage time (sub shows 2 minutes)
        *[row(r, "h", 90) for r in range(6, 12)],
        row(12, "h", 30, sub=True, rout=1),
        row(13, "h", 30, sub=True, rout=2, rin=14),
        row(14, "h", 10, sub=True, rout=13),
        row(15, "h", 2, sub=True, rout=5),
    ]
    a = [row(r, "a", 90) for r in range(21, 32)]
    return pd.DataFrame(h + a)


def test_lineup_rules() -> None:
    lineups, notes = derive_lineups(roster())
    r = lineups.set_index("roster_id")
    assert r.loc[1, ["on_minute", "off_minute", "off_reason"]].tolist() == [0, 60, "sub"]
    assert r.loc[12, ["on_minute", "off_minute", "off_reason"]].tolist() == [60, 90, "full"]
    assert r.loc[13, ["on_minute", "off_minute", "off_reason"]].tolist() == [50, 80, "sub"]
    assert r.loc[14, ["on_minute", "off_minute"]].tolist() == [80, 90]
    assert r.loc[3, ["off_minute", "off_reason", "red_minute", "red_late"]].tolist() == [
        33,
        "red",
        33,
        False,
    ]
    assert r.loc[4, ["red_minute", "red_late"]].tolist() == [90, True]
    assert r.loc[5, "off_minute"] == 88 and r.loc[15, "on_minute"] == 88
    assert r.loc[13, "replaced_roster_id"] == 2 and r.loc[13, "replaced_by_roster_id"] == 14
    assert not r.loc[12, "started"] and r.loc[1, "started"]
    assert notes == {"sub_pairs": 2, "sub_minute_disagreements": 0, "unpaired_subs": 0}


def test_events_keep_the_running_score() -> None:
    lineups, _ = derive_lineups(roster())
    shots = pd.DataFrame(
        [
            {
                "understat_match_id": 1,
                "shot_id": 1,
                "minute": 10,
                "side": "h",
                "result": "Goal",
                "situation": "OpenPlay",
                "player_uid": "fpl:6",
                "understat_player_id": 1006,
                "assister_uid": "fpl:7",
            },
            {
                "understat_match_id": 1,
                "shot_id": 2,
                "minute": 70,
                "side": "h",
                "result": "OwnGoal",
                "situation": "OpenPlay",
                "player_uid": "fpl:8",
                "understat_player_id": 1008,
                "assister_uid": pd.NA,
            },
            {
                "understat_match_id": 1,
                "shot_id": 3,
                "minute": 85,
                "side": "a",
                "result": "Goal",
                "situation": "Penalty",
                "player_uid": "fpl:21",
                "understat_player_id": 1021,
                "assister_uid": pd.NA,
            },
        ]
    ).assign(
        season="2030-31",
        fixture_uid="2030-31:a:b",
        event_at=KICKOFF,
        observed_at=KICKOFF + pd.Timedelta(hours=24),
    )
    us = pd.DataFrame({"understat_match_id": [1], "home_team": ["a"], "away_team": ["b"]})
    ev = match_events(lineups, shots, us)
    goals = ev[ev["kind"].isin(["goal", "own_goal"])]
    assert goals["side"].tolist() == ["h", "a", "a"]  # the own goal counts for the away side
    assert goals["other_player_uid"].iloc[0] == "fpl:7"
    assert goals["is_penalty"].tolist() == [False, False, True]
    last = ev.iloc[-1]
    assert (last["score_home_after"], last["score_away_after"]) == (1, 2)
    assert last["reds_home_after"] == 2
    subs = ev[ev["kind"] == "sub"].set_index("minute")
    assert subs.loc[60, "other_player_uid"] == "fpl:1"
    assert subs.loc[80, "other_player_uid"] == "fpl:13"
    assert ev["minute"].is_monotonic_increasing


@settings(max_examples=60, deadline=None)
@given(st.lists(st.tuples(st.integers(1, 89), st.booleans()), min_size=0, max_size=5))
def test_never_more_than_eleven_on_the_pitch(subs: list[tuple[int, bool]]) -> None:
    rows = [row(r, "h", 90) for r in range(1, 12)]
    next_id = 12
    for i, (minute, chained) in enumerate(subs):
        starter = rows[i]
        starter.update(minutes=minute, roster_in=next_id)
        rows.append(row(next_id, "h", 90 - minute, sub=True, rout=int(str(starter["roster_id"]))))
        if chained and minute < 85:
            rows[-1].update(minutes=3, roster_in=next_id + 1)
            rows.append(row(next_id + 1, "h", 90 - minute - 3, sub=True, rout=next_id))
            next_id += 1
        next_id += 1
    lineups, _ = derive_lineups(pd.DataFrame(rows))
    for m in range(90):
        on = ((lineups["on_minute"] <= m) & (lineups["off_minute"] > m)).sum()
        assert on == 11


def test_empty_roster() -> None:
    lineups, notes = derive_lineups(pd.DataFrame(columns=list(row(1, "h", 90))))
    assert lineups.empty and notes["sub_pairs"] == 0


@pytest.mark.parametrize("minutes", [0, 45])
def test_unpaired_sub_is_counted(minutes: int) -> None:
    rows = [row(r, "h", 90) for r in range(1, 12)] + [row(12, "h", minutes, sub=True)]
    lineups, notes = derive_lineups(pd.DataFrame(rows))
    assert notes["unpaired_subs"] == 1
    assert lineups.set_index("roster_id").loc[12, "on_minute"] == 90 - minutes
