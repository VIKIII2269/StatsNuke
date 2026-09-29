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
                    "observed_at": obs,
                }
            )
    snap = pd.DataFrame(snaps)
    return {"dim_fixture": dim_fixture, "fact_player_match": pm, "snap_fpl_player": snap}


def deadlines(frames: dict[str, pd.DataFrame]) -> list[pd.Timestamp]:
    fx = frames["dim_fixture"]
    return list(fx.groupby("round")["kickoff_at"].min() - pd.Timedelta(minutes=90))
