from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from fplh.entities.teams import TeamResolver
from fplh.lake.bronze import BronzeRecord, write_bronze
from fplh.lake.silver.odds_api import normalise_props
from fplh.lake.storage import Lake
from fplh.live.props import evaluate, match_players, name_key

KO = pd.Timestamp("2026-10-10 11:30", tz="UTC")


def test_props_from_the_event_endpoint(tmp_path: Path) -> None:
    lake = Lake(str(tmp_path))
    payload = {
        "id": "e1",
        "commence_time": "2026-10-10T11:30:00Z",
        "home_team": "Arsenal",
        "away_team": "Chelsea",
        "bookmakers": [
            {
                "key": "draftkings",
                "markets": [
                    {
                        "key": "player_goal_scorer_anytime",
                        "outcomes": [
                            {"name": "Yes", "description": "Bukayo Saka", "price": 2.4},
                            {"name": "No", "description": "Bukayo Saka", "price": 1.5},
                        ],
                    }
                ],
            }
        ],
    }
    write_bronze(
        lake,
        BronzeRecord(
            "odds_api",
            "event_odds",
            "u",
            200,
            datetime(2026, 10, 9, tzinfo=UTC),
            json.dumps(payload).encode(),
            {"event": "e1"},
        ),
    )
    t = TeamResolver.from_config()
    out = normalise_props(lake, t)
    assert len(out) == 1 and out["player_name"].iloc[0] == "Bukayo Saka"
    assert out["price"].iloc[0] == 2.4
    assert out["fixture_uid"].iloc[0] == f"2026-27:{t.uid('Arsenal')}:{t.uid('Chelsea')}"


def test_names_match_within_the_fixture() -> None:
    players = pd.DataFrame(
        {
            "player_uid": ["fpl:1", "fpl:2", "fpl:3", "fpl:4"],
            "team": ["ars", "ars", "che", "lee"],
            "first_name": ["Bukayo", "Gabriel", "Cole", "Joe"],
            "second_name": ["Saka", "dos Santos Magalhães", "Palmer", "Rodon"],
            "web_name": ["Saka", "Gabriel", "Palmer", "Rodon"],
        }
    )
    props = pd.DataFrame(
        {
            "home_team": ["ars"] * 4,
            "away_team": ["che"] * 4,
            "player_name": ["Bukayo Saka", "Gabriel", "C. Palmer", "Joe Rodon"],
        }
    )
    out = match_players(props, players)
    assert list(out["player_uid"]) == ["fpl:1", "fpl:2", "fpl:3", None]  # Rodon not in it
    assert name_key("Magalhães") == "magalhaes"


def test_log_loss_with_a_walk_forward_margin() -> None:
    props = pd.DataFrame(
        {
            "fixture_uid": ["f1", "f1", "f2", "f2"],
            "player_uid": ["a", "b", "a", "b"],
            "bookmaker": ["dk"] * 4,
            "price": [2.0, 5.0, 2.0, 5.0],
            "observed_at": [KO - pd.Timedelta(hours=1)] * 4,
            "kickoff_at": [KO] * 4,
        }
    )
    forecasts = pd.DataFrame(
        {
            "gw": [6, 6, 7, 7],
            "fixture_uid": ["f1", "f1", "f2", "f2"],
            "player_uid": ["a", "b", "a", "b"],
            "p_goal": [0.4, 0.1, 0.4, 0.1],
            "p_play": [0.8, 1.0, 0.8, 1.0],
        }
    )
    outcomes = pd.DataFrame(
        {
            "fixture_uid": ["f1", "f1", "f2", "f2"],
            "player_uid": ["a", "b", "a", "b"],
            "minutes": [90, 90, 90, 0],  # b did not play in f2: void
            "goals_scored": [1, 0, 0, 0],
        }
    )
    rows, summary = evaluate(props, forecasts, outcomes)
    assert len(rows) == 3 and list(summary["gw"]) == [6, 7]
    assert np.isclose(rows.loc[rows["fixture_uid"] == "f1", "c"], 0.85).all()
    # GW7's margin is learned from GW6 only: 1 goal / (0.5 + 0.2) implied
    assert np.isclose(rows.loc[rows["fixture_uid"] == "f2", "c"].iloc[0], 1 / 0.7)
    assert np.isclose(rows["ours"].iloc[0], 0.5)  # 0.4 / 0.8, given that he plays


def test_odds_snapshots_keep_pre_match_prices_only(tmp_path: Path) -> None:
    """The odds endpoint also lists matches in play; a snapshot taken after one kicked
    off (Liverpool v Man City, 11 Oct) failed the silver build until these were dropped."""
    from fplh.lake.silver.odds_api import normalise

    lake = Lake(str(tmp_path))

    def event(kickoff: str, home: str, away: str) -> dict[str, object]:
        outcomes = [{"name": home, "price": 2.0}, {"name": away, "price": 3.5}]
        outcomes.append({"name": "Draw", "price": 3.4})
        return {
            "commence_time": kickoff,
            "home_team": home,
            "away_team": away,
            "bookmakers": [{"key": "skybet", "markets": [{"key": "h2h", "outcomes": outcomes}]}],
        }

    payload = [
        event("2026-10-11T15:30:00Z", "Liverpool", "Manchester City"),  # in play
        event("2026-10-12T19:00:00Z", "Coventry City", "Newcastle United"),
    ]
    write_bronze(
        lake,
        BronzeRecord(
            "odds_api",
            "odds",
            "u",
            200,
            datetime(2026, 10, 11, 16, 0, tzinfo=UTC),
            json.dumps(payload).encode(),
            {"sport": "soccer_epl"},
        ),
    )
    out = normalise(lake, TeamResolver.from_config())
    assert len(out) == 3 and (out["observed_at"] <= out["kickoff_at"]).all()
