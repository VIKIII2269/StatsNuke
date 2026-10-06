from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import pytest

import fplh.live.team as live
from fplh.delivery.report import render, team_results, team_week
from fplh.entities.teams import TeamResolver
from fplh.features.information_set import SilverStore
from fplh.lake.bronze import BronzeRecord, write_bronze
from fplh.lake.silver.fpl_live import schedule
from fplh.lake.storage import Lake
from fplh.optimize.milp import State
from fplh.rules.config import load_rules
from fplh.rules.team import score_gameweek

SEASON = "2026-27"
DEADLINE = pd.Timestamp("2026-10-10 10:00", tz="UTC")
TEAMS = [f"t{i}" for i in range(20)]
POS = ["GK", "GK", "DEF", "DEF", "MID", "MID", "FWD"]


def frames() -> dict[str, pd.DataFrame]:
    players, snap = [], []
    for i, team in enumerate(TEAMS):
        for k, pos in enumerate(POS):
            uid = f"fpl:{100 * i + k}"
            players.append(uid)
            snap.append(
                {
                    "season": SEASON,
                    "code": 100 * i + k,
                    "element": 100 * i + k,
                    "position": pos,
                    "team": team,
                    "now_cost": 45 + (i + k) % 6 * 5,
                    "status": "a",
                    "observed_at": DEADLINE - pd.Timedelta(days=1),
                }
            )
    fixtures = [
        {
            "fixture_uid": f"{SEASON}:{TEAMS[2 * j]}:{TEAMS[2 * j + 1]}:{r}",  # unique ids
            "season": SEASON,
            "home_team": TEAMS[2 * j],
            "away_team": TEAMS[2 * j + 1],
            "kickoff_at": DEADLINE + pd.Timedelta(hours=1.5 + 168 * (r - 6)),
            "round": r,
        }
        for r in range(6, 11)
        for j in range(10)
    ]
    events = [
        {
            "season": SEASON,
            "gameweek": g,
            "deadline_at": DEADLINE + pd.Timedelta(days=7 * (g - 6)),
            "finished": False,
            "data_checked": False,
            "average_entry_score": None,
            "highest_score": None,
            "observed_at": DEADLINE - pd.Timedelta(days=2),
        }
        for g in range(6, 11)
    ]
    return {
        "snap_fpl_player": pd.DataFrame(snap),
        "dim_fixture": pd.DataFrame(fixtures),
        "fpl_event": pd.DataFrame(events),
        "fact_player_match": pd.DataFrame(
            columns=["season", "round", "player_uid", "total_points", "minutes", "position"]
        ),
    }


def fake_forecast(lake: Lake, store: SilverStore, deadline: pd.Timestamp) -> pd.DataFrame:
    fx = store.get("dim_fixture")
    snap = store.get("snap_fpl_player")
    rows = []
    for _, f in fx.iterrows():
        for _, p in snap[snap["team"].isin([f["home_team"], f["away_team"]])].iterrows():
            rows.append(
                {
                    "player_uid": f"fpl:{p['code']}",
                    "fixture_uid": f["fixture_uid"],
                    "deadline_at": deadline,
                    "expected_points": 1.0 + (p["code"] % 7) * 0.4 + p["now_cost"] / 50,
                }
            )
    return pd.DataFrame(rows)


@pytest.fixture
def setup(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Lake, SilverStore]:
    monkeypatch.setattr(live, "forecast", fake_forecast)
    monkeypatch.setattr(live, "save_replica", lambda *a, **k: None)
    return Lake(str(tmp_path)), SilverStore.from_frames(frames())


def test_advise_decides_a_valid_squad_once_per_gameweek(setup: tuple[Lake, SilverStore]) -> None:
    lake, store = setup
    now = DEADLINE - pd.Timedelta(hours=10)
    assert live.advise(lake, store, SEASON, DEADLINE - pd.Timedelta(days=3)) is None  # too early
    rec = live.advise(lake, store, SEASON, now)
    assert rec is not None and rec["gameweek"] == 6
    assert len(rec["xi"]) == 11 and len(rec["bench"]) == 4 and len(rec["squad"]) == 15
    clubs = pd.Series([u.split(":")[1][:-2] or "0" for u in rec["squad"]]).value_counts()
    assert clubs.max() <= 3
    assert rec["bank"] >= 0
    assert live.advise(lake, store, SEASON, now) is None  # decided already
    team = live.load(lake, SEASON)
    assert team is not None and set(team.state.squad) == set(rec["squad"])
    assert team.state.gameweek == 7
    again = live.LiveTeam.from_json(team.to_json())
    assert again.state == team.state and again.weeks == team.weeks


def test_score_matches_the_official_rules(setup: tuple[Lake, SilverStore]) -> None:
    lake, store = setup
    rec = live.advise(lake, store, SEASON, DEADLINE - pd.Timedelta(hours=10))
    assert rec is not None
    picks = [*rec["xi"], *rec["bench"]]
    snap = store.get("snap_fpl_player").set_index("code")
    pm = pd.DataFrame(
        {
            "season": SEASON,
            "round": 6,
            "player_uid": picks,
            "total_points": [(i % 5) + 1 for i in range(15)],
            "minutes": [0 if i == 3 else 90 for i in range(15)],  # one starter missed out
            "position": [snap.loc[int(p.split(":")[1]), "position"] for p in picks],
        }
    )
    f = frames()
    f["fact_player_match"] = pm
    ev = f["fpl_event"]
    ev.loc[ev["gameweek"] == 6, ["data_checked", "average_entry_score", "highest_score"]] = (
        True,
        48,
        120,
    )
    store2 = SilverStore.from_frames(f)
    assert live.score(lake, store2, SEASON) == [6]
    week = live.load(lake, SEASON).weeks["6"]  # type: ignore[union-attr]
    rules = load_rules("2026/27")
    expected = score_gameweek(
        picks,
        dict(zip(pm["player_uid"], pm["position"], strict=True)),
        dict(zip(pm["player_uid"], pm["total_points"], strict=True)),
        dict(zip(pm["player_uid"], pm["minutes"], strict=True)),
        rec["captain"],
        rec["vice"],
        rec["chip"],
        {str(k): int(v) for k, v in rules.game.xi_min.items()},
    )
    assert week["gross"] == expected.points and week["average"] == 48
    assert live.score(lake, store2, SEASON) == []  # scored once
    team = live.load(lake, SEASON)
    assert team is not None
    md = render([team_week(6, week, {}), team_results(team)])
    assert "GW6" in md and "average 48" in md and "| 6 |" in md


def test_schedule_gives_rounds_for_unplayed_fixtures(tmp_path: Path) -> None:
    lake = Lake(str(tmp_path))
    resolver = TeamResolver.from_config()
    names = ["Arsenal", "Chelsea"]
    boot = {
        "events": [{"id": 1, "deadline_time": "2026-08-14T17:30:00Z"}],
        "teams": [{"id": i + 1, "name": n} for i, n in enumerate(names)],
    }
    fixtures = [
        {
            "id": 1,
            "event": 1,
            "kickoff_time": "2026-08-15T14:00:00Z",
            "team_h": 1,
            "team_a": 2,
            "team_h_score": 2,
            "team_a_score": 1,
            "finished": True,
        },
        {
            "id": 2,
            "event": 20,
            "kickoff_time": "2027-01-02T15:00:00Z",
            "team_h": 2,
            "team_a": 1,
            "team_h_score": None,
            "team_a_score": None,
            "finished": False,
        },
        {
            "id": 3,
            "event": None,
            "kickoff_time": None,
            "team_h": 1,
            "team_a": 2,
            "team_h_score": None,
            "team_a_score": None,
            "finished": False,
        },  # postponed, unscheduled
    ]
    write_bronze(
        lake,
        BronzeRecord(
            "fpl",
            "fixtures",
            "u",
            200,
            datetime(2026, 10, 1, tzinfo=UTC),
            json.dumps(fixtures).encode(),
        ),
    )
    out = schedule(lake, resolver, boot)
    assert list(out["round"]) == [1, 20] and out["season"].iloc[0] == SEASON
    assert out["home_goals"].iloc[0] == 2 and pd.isna(out["home_goals"].iloc[1])
    assert out["home_team"].iloc[0] == resolver.uid("Arsenal")


def test_live_team_state_starts_fresh_per_season(setup: tuple[Lake, SilverStore]) -> None:
    lake, _ = setup
    live.save(lake, live.LiveTeam("2025-26", State(30, {"fpl:1": 50}, 0, 1, {})))
    assert live.load(lake, SEASON) is None


def test_plan_next_is_provisional_and_refreshed_every_12_hours(
    setup: tuple[Lake, SilverStore],
) -> None:
    lake, store = setup
    now = DEADLINE - pd.Timedelta(hours=23)
    rec = live.plan_next(lake, store, SEASON, now)
    assert rec is not None and rec["provisional"] and rec["gameweek"] == 6
    assert len(rec["xi"]) == 11
    assert live.load(lake, SEASON) is None  # nothing committed
    assert live.plan_next(lake, store, SEASON, now + pd.Timedelta(hours=6)) is None
    assert live.plan_next(lake, store, SEASON, now + pd.Timedelta(hours=13)) is not None
    live.advise(lake, store, SEASON, DEADLINE - pd.Timedelta(hours=9))
    assert live.plan_next(lake, store, SEASON, DEADLINE - pd.Timedelta(hours=8), force=True) is None


def test_site_snapshot_builds_from_the_live_state(
    setup: tuple[Lake, SilverStore], tmp_path: Path
) -> None:
    from fplh.web.export import build

    lake, store = setup
    now = DEADLINE - pd.Timedelta(hours=20)
    live.plan_next(lake, store, SEASON, now)
    log = tmp_path / "log.md"
    log.write_text("# Log\n\n## FPL\n\n| # | Idea | Status |\n|---|---|---|\n| F1 | **x** | ✅ |\n")
    snap = build(lake, store, SEASON, now, log_path=log)
    json.dumps(snap, allow_nan=False)  # strict JSON
    assert snap["next"]["gw"] == 6
    assert snap["plan"]["provisional"] is True
    assert snap["forecast"]["gws"] == [6, 7, 8, 9, 10]
    top = snap["players"][0]
    assert len(top["xp"]) == 5 and top["xp5"] == pytest.approx(sum(top["xp"]), abs=0.05)
    assert {p["id"] for p in snap["players"]} >= set(snap["plan"]["xi"])
    assert snap["lab"][0]["tables"][0]["rows"] == [["F1", "**x**", "✅"]]
    names = {c["name"] for c in snap["data"]["checks"]}
    assert "Fixture schedule" in names and "Forecast for the next deadline" in names
