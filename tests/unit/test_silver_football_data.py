from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from fplh.entities.teams import TeamResolver
from fplh.lake.silver.football_data import (
    normalise_file,
    odds_column_report,
    prematch_observed_at,
    read_football_data_csv,
)

SAMPLE = (Path(__file__).parent / "data" / "fd_e0_2526_sample.csv").read_bytes()


def ts(s: str) -> pd.Timestamp:
    return pd.Timestamp(s, tz="UTC")


@pytest.mark.parametrize(
    ("kickoff", "observed"),
    [
        ("2025-08-16 14:00", "2025-08-15 15:00"),  # Saturday → Friday 16:00 BST
        ("2025-08-18 19:00", "2025-08-15 15:00"),  # Monday → previous Friday
        ("2025-12-03 19:30", "2025-12-02 16:00"),  # Wednesday → Tuesday 16:00 GMT
        ("2025-08-15 19:00", "2025-08-15 15:00"),  # Friday evening → same Friday
        (
            "2025-12-02 15:00",
            "2025-12-02 14:00",
        ),  # kickoff on the collection day: capped at kickoff − 1 h
        ("2014-12-26 15:00", "2014-12-26 14:00"),  # Boxing Day (Friday) 15:00
    ],
)
def test_prematch_observation_rule(kickoff: str, observed: str) -> None:
    assert prematch_observed_at(ts(kickoff)) == ts(observed)


def test_parses_results_stats_and_odds() -> None:
    match, odds, notes = normalise_file(
        SAMPLE, start_year=2025, division="E0", teams=TeamResolver.from_config()
    )
    assert notes["rows"] == 2 and notes["kickoff_time_imputed"] == 0
    first = match.iloc[0]
    assert (first["home_team"], first["away_team"], first["home_goals"], first["away_goals"]) == (
        "liverpool",
        "bournemouth",
        4,
        2,
    )
    assert first["kickoff_at"] == ts("2025-08-15 19:00")  # 20:00 UK (BST)
    assert first["observed_at"] - first["event_at"] == pd.Timedelta(hours=48)
    pin = odds[(odds["bookmaker"] == "pinnacle") & (odds["home_team"] == "liverpool")]
    assert set(pin["market"]) == {"1x2", "total"}
    closing = pin[pin["is_closing"] & (pin["market"] == "1x2")].set_index("outcome")["price"]
    assert closing.to_dict() == {"home": 1.3, "draw": 6.3, "away": 10.2}
    assert (
        odds.loc[odds["is_closing"], "observed_at"] == odds.loc[odds["is_closing"], "kickoff_at"]
    ).all()
    assert {"bet365", "market_avg", "market_max", "pinnacle"} <= set(odds["bookmaker"])


def test_old_format_without_time_and_ragged_rows() -> None:
    csv = (
        "Div,Date,HomeTeam,AwayTeam,FTHG,FTAG,B365H,B365D,B365A\n"
        "E0,13/08/05,Arsenal,Newcastle,2,0,1.4,4.0,7.5,,\n"
        "E0,14/08/05,Chelsea,Wigan,1,0,1.2,5.0,11\n"
        ",,,,,,,,\n"
    ).encode("cp1252")
    teams = TeamResolver.from_config()
    match, odds, notes = normalise_file(csv, start_year=2005, division="E0", teams=teams)
    assert notes["ragged_rows_padded"] == 1 and notes["kickoff_time_imputed"] == 2
    assert len(match) == 2 and match["kickoff_time_imputed"].all()
    assert match.iloc[0]["kickoff_at"] == ts("2005-08-13 14:00")  # 15:00 BST default
    assert "wigan" in set(match["away_team"])  # football-data names define canonical uids
    assert len(odds) == 6


def test_reader_handles_cp1252() -> None:
    df, _ = read_football_data_csv(
        "Div,HomeTeam,AwayTeam,Referee\nE0,A,B,Mike Dean é\n".encode("cp1252")
    )
    assert df.iloc[0]["Referee"].endswith("é")


def test_odds_column_report() -> None:
    old = b"Div,Date,HomeTeam,AwayTeam,FTHG,FTAG,B365H,B365D,B365A\nE0,13/08/05,A,B,1,0,2,3,4\n"
    report = odds_column_report({"0506": old, "2526": SAMPLE}).set_index("season")
    assert report.loc["2526", "pinnacle_closing_1x2"] == 1.0
    assert report.loc["0506", "pinnacle_closing_1x2"] == 0.0
    assert report.loc["2526", "matches"] == 2


def test_report_sorts_by_season_start() -> None:
    old = b"Div,Date,HomeTeam,AwayTeam,FTHG,FTAG\nE0,13/08/94,A,B,1,0\n"
    report = odds_column_report({"2526": SAMPLE, "9495": old})
    assert report["season"].tolist() == ["9495", "2526"]


def test_cap_is_always_after_the_rounds_deadline() -> None:
    """Deadline = first kickoff − 90 min ≤ this kickoff − 90 min < kickoff − 1 h."""
    for kickoff in ("2025-12-02 15:00", "2014-12-26 12:45", "2025-08-16 11:30"):
        k = ts(kickoff)
        observed = prematch_observed_at(k)
        if observed.date() == k.date() and observed > k - pd.Timedelta(hours=4):
            assert observed > k - pd.Timedelta(minutes=90)
