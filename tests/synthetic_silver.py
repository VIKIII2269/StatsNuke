"""A synthetic silver season (4 teams, 6 rounds, double round-robin) for harness tests."""

from __future__ import annotations

import itertools

import numpy as np
import pandas as pd

SEASON = "2030-31"
TEAMS = ["alpha", "beta", "gamma", "delta"]
LAG = pd.Timedelta(hours=33)


def make(seed: int = 0) -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    pairs = list(itertools.permutations(TEAMS, 2))  # 12 fixtures, 2 per round
    start = pd.Timestamp("2030-08-10 14:00", tz="UTC")
    fixtures = []
    for i, (h, a) in enumerate(pairs):
        rnd = i // 2 + 1
        kickoff = start + pd.Timedelta(days=7 * (rnd - 1), hours=3 * (i % 2))
        fixtures.append(
            {
                "fixture_uid": f"{SEASON}:{h}:{a}",
                "season": SEASON,
                "home_team": h,
                "away_team": a,
                "kickoff_at": kickoff,
                "round": rnd,
                "fpl_fixture_id": i + 1,
                "home_goals": int(rng.poisson(1.5)),
                "away_goals": int(rng.poisson(1.1)),
            }
        )
    dim_fixture = pd.DataFrame(fixtures)
    dim_fixture["round"] = dim_fixture["round"].astype("Int64")

    positions = ["GK", "DEF", "DEF", "MID", "FWD"]
    players = [
        (f"fpl:{1000 + 10 * i + k}", t, pos)
        for i, t in enumerate(TEAMS)
        for k, pos in enumerate(positions)
    ]
    code_of = {uid: int(uid.split(":")[1]) for uid, _, _ in players}
    rows = []
    for f in fixtures:
        for uid, team, pos in players:
            if team not in (f["home_team"], f["away_team"]):
                continue
            home = team == f["home_team"]
            conceded = f["away_goals"] if home else f["home_goals"]
            mins = int(rng.choice([0, 30, 90], p=[0.1, 0.2, 0.7]))
            rows.append(
                {
                    "season": SEASON,
                    "player_uid": uid,
                    "code": code_of[uid],
                    "element": players.index((uid, team, pos)) + 1,
                    "fixture_uid": f["fixture_uid"],
                    "fpl_fixture_id": f["fpl_fixture_id"],
                    "position": pos,
                    "team": team,
                    "opponent": f["away_team"] if home else f["home_team"],
                    "was_home": home,
                    "kickoff_at": f["kickoff_at"],
                    "minutes": mins,
                    "goals_scored": int(rng.poisson(0.2)) if mins else 0,
                    "assists": 0,
                    "saves": 3 if pos == "GK" and mins else 0,
                    "bonus": 0,
                    "yellow_cards": 0,
                    "red_cards": 0,
                    "own_goals": 0,
                    "penalties_missed": 0,
                    "penalties_saved": 0,
                    "goals_conceded": conceded if mins else 0,
                    "clean_sheets": int(mins >= 60 and conceded == 0),
                    "starts": int(mins >= 60),
                    "clearances_blocks_interceptions": int(rng.poisson(4)),
                    "tackles": int(rng.poisson(2)),
                    "recoveries": int(rng.poisson(5)),
                    "total_points": 2,
                    "transfers_in": int(rng.integers(0, 5000)) * (f["round"] > 1),
                    "transfers_out": int(rng.integers(0, 5000)) * (f["round"] > 1),
                    "selected": int(rng.integers(5000, 50000)),
                    "round": f["round"],
                    "event_at": f["kickoff_at"],
                    "observed_at": f["kickoff_at"] + LAG,
                }
            )
    pm = pd.DataFrame(rows)
    snaps = []
    for day in range(0, 50, 3):
        obs = start - pd.Timedelta(days=5) + pd.Timedelta(days=day)
        for uid, team, pos in players:
            snaps.append(
                {
                    "season": SEASON,
                    "element": players.index((uid, team, pos)) + 1,
                    "code": code_of[uid],
                    "team": team,
                    "position": pos,
                    "status": "a",
                    "chance_of_playing_next_round": float(rng.choice([100, 75, 0])),
                    "now_cost": 50,
                    "selected_by_percent": 5.0,
                    "ep_next": float(rng.uniform(0, 8)),
                    "transfers_in_event": int(rng.integers(0, 5000)),
                    "transfers_out_event": int(rng.integers(0, 5000)),
                    "total_players": 100_000,
                    "observed_at": obs,
                }
            )
    snap = pd.DataFrame(snaps)
    us = dim_fixture.assign(
        understat_match_id=range(len(dim_fixture)),
        is_result=True,
        home_xg=dim_fixture["home_goals"] * 0.8 + 0.3,
        away_xg=dim_fixture["away_goals"] * 0.8 + 0.3,
        event_at=dim_fixture["kickoff_at"],
        observed_at=dim_fixture["kickoff_at"] + pd.Timedelta(hours=24),
    )
    return {
        "dim_fixture": dim_fixture,
        "fact_player_match": pm,
        "snap_fpl_player": snap,
        "us_match": us,
    }


def deadlines(frames: dict[str, pd.DataFrame]) -> list[pd.Timestamp]:
    fx = frames["dim_fixture"]
    return list(fx.groupby("round")["kickoff_at"].min() - pd.Timedelta(minutes=90))


def with_understat(frames: dict[str, pd.DataFrame], seed: int = 0) -> dict[str, pd.DataFrame]:
    """Add Understat rosters (11 starters + up to 3 subs per side, an occasional red),
    shots whose goals reproduce the scores, and the derived ``fact_match_event``."""
    from fplh.lake.silver.timeline import derive_lineups, match_events

    rng = np.random.default_rng(seed)
    us = frames["us_match"]
    rosters, shots = [], []
    rid, sid = 1, 1
    for m in us.itertuples():
        stamp = {
            "season": m.season,
            "understat_match_id": m.understat_match_id,
            "fixture_uid": m.fixture_uid,
            "event_at": m.event_at,
            "observed_at": m.observed_at,
        }
        for side, team, goals in (
            ("h", m.home_team, m.home_goals),
            ("a", m.away_team, m.away_goals),
        ):
            ids = list(range(rid, rid + 11))
            side_rows = {
                r: {
                    **stamp,
                    "roster_id": r,
                    "understat_player_id": 5000 + (r % 40),
                    "player_uid": f"us:{team}:{r - ids[0]}",
                    "player_name": f"{team}-{r - ids[0]}",
                    "side": side,
                    "team": team,
                    "position": "GK" if r == ids[0] else "MC",
                    "minutes": 90,
                    "red_cards": 0,
                    "roster_in": 0,
                    "roster_out": 0,
                }
                for r in ids
            }
            rid += 11
            for k in range(int(rng.integers(0, 4))):
                minute = int(rng.integers(46, 89))
                off = side_rows[ids[10 - k]]
                off.update(minutes=minute, roster_in=rid)
                side_rows[rid] = {
                    **off,
                    "roster_id": rid,
                    "position": "Sub",
                    "player_uid": f"us:{team}:{11 + k}",
                    "player_name": f"{team}-{11 + k}",
                    "minutes": 90 - minute,
                    "roster_in": 0,
                    "roster_out": off["roster_id"],
                }
                rid += 1
            if rng.random() < 0.1:
                side_rows[ids[1]].update(minutes=int(rng.integers(20, 80)), red_cards=1)
            rosters.extend(side_rows.values())
            for g in range(int(goals)):
                shooter = side_rows[ids[1 + (g % 9)]]
                shots.append(
                    {
                        **stamp,
                        "shot_id": sid,
                        "minute": int(rng.integers(1, 90)),
                        "side": side,
                        "team": team,
                        "result": "Goal",
                        "situation": "OpenPlay",
                        "xg": 0.3,
                        "understat_player_id": shooter["understat_player_id"],
                        "player_uid": shooter["player_uid"],
                        "assister_uid": pd.NA,
                    }
                )
                sid += 1
    roster = pd.DataFrame(rosters)
    lineups, _ = derive_lineups(roster)
    shot_df = pd.DataFrame(shots)
    events = match_events(lineups, shot_df, us)
    return {
        **frames,
        "fact_player_match_understat": lineups,
        "fact_shot": shot_df,
        "fact_match_event": events,
    }
