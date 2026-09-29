"""A tiny, internally consistent bronze lake spanning every Phase 1 source.

Season 2025/26, two fixtures between Liverpool and Bournemouth, four players. The
numbers agree across FPL (vaastav), football-data and Understat, so every quality gate
passes; tests corrupt one thing at a time to prove each gate bites.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any

from fplh.lake.bronze import BronzeRecord, write_bronze
from fplh.lake.storage import Lake

OBS = datetime(2026, 9, 1, 6, 0, tzinfo=UTC)

PLAYERS = [
    # element, code, first, second, web, element_type, team_id, understat_id, understat_name
    (1, 1001, "Mohamed", "Salah", "M.Salah", 3, 1, 1250, "Mohamed Salah"),
    (2, 1002, "Virgil", "van Dijk", "Virgil", 2, 1, 6026, "Virgil van Dijk"),
    (3, 1003, "Antoine", "Semenyo", "Semenyo", 3, 2, 8260, "Antoine Semenyo"),
    (4, 1004, "Kepa", "Arrizabalaga Revuelta", "Kepa", 1, 2, 1100, "Kepa"),
]
# fixture id, round, kickoff, home team id, away team id, home goals, away goals
FIXTURES = [
    (1, 1, "2025-08-15T19:00:00Z", 1, 2, 2, 1),
    (2, 20, "2026-01-03T15:00:00Z", 2, 1, 0, 0),
]
TEAM_NAMES = {1: "Liverpool", 2: "Bournemouth"}
# per (fixture, element): minutes, goals, own_goals, shots
STATS = {
    (1, 1): (90, 2, 0, 3),
    (1, 2): (90, 0, 0, 1),
    (1, 3): (90, 1, 0, 2),
    (1, 4): (90, 0, 0, 0),
    (2, 1): (80, 0, 0, 1),
    (2, 2): (90, 0, 0, 0),
    (2, 3): (90, 0, 0, 2),
    (2, 4): (90, 0, 0, 0),
}


def _csv(rows: list[dict[str, Any]]) -> bytes:
    cols = list(rows[0])
    lines = [",".join(cols)] + [
        ",".join("" if r[c] is None else str(r[c]) for c in cols) for r in rows
    ]
    return ("\n".join(lines) + "\n").encode()


def _put(
    lake: Lake, source: str, endpoint: str, params: dict[str, str], payload: bytes, minute: int = 0
) -> None:
    write_bronze(
        lake,
        BronzeRecord(
            source=source,
            endpoint=endpoint,
            url=f"https://example.test/{source}/{endpoint}",
            http_status=200,
            observed_at=OBS.replace(minute=minute),
            payload=payload,
            params=params,
        ),
    )


def merged_gw_rows() -> list[dict[str, Any]]:
    rows = []
    for fid, rnd, kickoff, h, a, hg, ag in FIXTURES:
        for el, _code, first, second, _web, et, team, *_ in PLAYERS:
            mins, goals, og, _shots = STATS[(fid, el)]
            home = team == h
            conceded = (ag if home else hg) if mins else 0
            rows.append(
                {
                    "name": f"{first} {second}",
                    "position": {1: "GK", 2: "DEF", 3: "MID", 4: "FWD"}[et],
                    "team": TEAM_NAMES[team],
                    "xP": 9.9,
                    "assists": 0,
                    "bonus": 0,
                    "bps": 10,
                    "clean_sheets": int(mins >= 60 and conceded == 0),
                    "element": el,
                    "fixture": fid,
                    "goals_conceded": conceded,
                    "goals_scored": goals,
                    "kickoff_time": kickoff,
                    "minutes": mins,
                    "opponent_team": a if home else h,
                    "own_goals": og,
                    "penalties_missed": 0,
                    "penalties_saved": 0,
                    "red_cards": 0,
                    "round": rnd,
                    "saves": 0,
                    "selected": 1000,
                    "starts": 1,
                    "team_a_score": ag,
                    "team_h_score": hg,
                    "total_points": 2,
                    "value": 50,
                    "was_home": home,
                    "yellow_cards": 0,
                    "GW": rnd,
                }
            )
    return rows


def football_data_csv() -> bytes:
    rows = []
    for _fid, _rnd, kickoff, h, a, hg, ag in FIXTURES:
        d = datetime.fromisoformat(kickoff.replace("Z", "+00:00"))
        rows.append(
            {
                "Div": "E0",
                "Date": d.strftime("%d/%m/%Y"),
                "Time": (
                    d.replace(hour=d.hour + (1 if d.month in (4, 5, 6, 7, 8, 9, 10) else 0))
                ).strftime("%H:%M"),
                "HomeTeam": TEAM_NAMES[h],
                "AwayTeam": TEAM_NAMES[a],
                "FTHG": hg,
                "FTAG": ag,
                "FTR": "H" if hg > ag else ("A" if ag > hg else "D"),
                "HS": 10,
                "AS": 8,
                "PSH": 1.8,
                "PSD": 3.8,
                "PSA": 4.5,
                "AvgH": 1.75,
                "AvgD": 3.7,
                "AvgA": 4.4,
                "P>2.5": 1.7,
                "P<2.5": 2.2,
                "PSCH": 1.78,
                "PSCD": 3.9,
                "PSCA": 4.6,
                "AvgCH": 1.74,
                "AvgCD": 3.75,
                "AvgCA": 4.5,
            }
        )
    return _csv(rows)


def understat_payloads() -> tuple[dict[str, Any], dict[int, dict[str, Any]]]:
    us_team = {1: ("87", "Liverpool"), 2: ("73", "Bournemouth")}
    dates, matches = [], {}
    for fid, _rnd, kickoff, h, a, hg, ag in FIXTURES:
        mid = 30000 + fid
        dt = kickoff.replace("T", " ").replace("Z", "")
        dates.append(
            {
                "id": str(mid),
                "isResult": True,
                "h": {"id": us_team[h][0], "title": us_team[h][1], "short_title": "H"},
                "a": {"id": us_team[a][0], "title": us_team[a][1], "short_title": "A"},
                "goals": {"h": str(hg), "a": str(ag)},
                "xG": {"h": "1.5", "a": "0.9"},
                "datetime": dt,
            }
        )
        rosters: dict[str, dict[str, Any]] = {"h": {}, "a": {}}
        shots: dict[str, list[dict[str, Any]]] = {"h": [], "a": []}
        sid = mid * 100
        for el, _code, _f, _s, _w, _et, team, us_id, us_name in PLAYERS:
            side = "h" if team == h else "a"
            mins, goals, og, nshots = STATS[(fid, el)]
            rosters[side][str(el)] = {
                "id": str(el),
                "goals": str(goals),
                "own_goals": str(og),
                "shots": str(nshots),
                "xG": "0.3",
                "time": str(mins),
                "player_id": str(us_id),
                "team_id": us_team[team][0],
                "position": "M",
                "player": us_name,
                "h_a": side,
                "yellow_card": "0",
                "red_card": "0",
                "roster_in": "0",
                "roster_out": "0",
                "key_passes": "0",
                "assists": "0",
                "xA": "0",
            }
            for k in range(nshots):
                sid += 1
                shots[side].append(
                    {
                        "id": str(sid),
                        "minute": str(10 + k),
                        "result": "Goal" if k < goals else "MissedShots",
                        "X": "0.9",
                        "Y": "0.5",
                        "xG": "0.1",
                        "player": us_name,
                        "h_a": side,
                        "player_id": str(us_id),
                        "situation": "OpenPlay",
                        "shotType": "RightFoot",
                        "match_id": str(mid),
                        "date": dt,
                        "player_assisted": None,
                        "lastAction": "Pass",
                    }
                )
        matches[mid] = {"match_info": {"id": str(mid)}, "rostersData": rosters, "shotsData": shots}
    teams_data = {
        us_team[t][0]: {"id": us_team[t][0], "title": us_team[t][1], "history": []} for t in (1, 2)
    }
    players_data = [
        {
            "id": str(p[7]),
            "player_name": p[8],
            "games": "2",
            "time": "180",
            "goals": "0",
            "xG": "0",
            "assists": "0",
            "xA": "0",
            "shots": "0",
            "key_passes": "0",
            "position": "M",
            "team_title": TEAM_NAMES[p[6]],
            "npg": "0",
            "npxG": "0",
        }
        for p in PLAYERS
    ]
    return {"datesData": dates, "teamsData": teams_data, "playersData": players_data}, matches


def build_mini_lake(
    lake: Lake, *, fd_csv: bytes | None = None, gw_rows: list[dict[str, Any]] | None = None
) -> None:
    season = {"season": "2025_26"}
    _put(
        lake,
        "vaastav",
        "master_team_list",
        {"as_of_season": "2026_27"},
        _csv([{"season": "2025-26", "team": t, "team_name": n} for t, n in TEAM_NAMES.items()]),
    )
    _put(lake, "vaastav", "merged_gw", season, _csv(gw_rows or merged_gw_rows()))
    _put(
        lake,
        "vaastav",
        "players_raw",
        season,
        _csv(
            [
                {
                    "id": p[0],
                    "code": p[1],
                    "first_name": p[2],
                    "second_name": p[3],
                    "web_name": p[4],
                    "element_type": p[5],
                    "team": p[6],
                }
                for p in PLAYERS
            ]
        ),
    )
    _put(
        lake,
        "vaastav",
        "teams",
        season,
        _csv([{"id": t, "name": n} for t, n in TEAM_NAMES.items()]),
    )
    _put(lake, "football_data", "E0", {"season": "2526"}, fd_csv or football_data_csv())
    league, matches = understat_payloads()
    _put(
        lake,
        "understat",
        "league",
        {"league": "EPL", "season": "2025"},
        json.dumps(league).encode(),
    )
    for mid, payload in matches.items():
        _put(lake, "understat", "match", {"match": str(mid)}, json.dumps(payload).encode())
