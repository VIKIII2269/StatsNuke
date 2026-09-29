"""Understat bronze → ``us_match``, ``us_team_match``, ``us_player_season``,
``fact_shot`` and ``fact_player_match_understat``.

Understat timestamps are treated as UTC. Fixture linking tolerates ± 1 day, and linked
facts take their event time from the FPL kickoff in ``dim_fixture``.
"""

from __future__ import annotations

import html
import json
from datetime import timedelta
from typing import Any

import pandas as pd

from fplh.collectors.understat_schema import LeagueData, MatchData
from fplh.entities.teams import TeamResolver
from fplh.lake.silver.common import season_label

SOURCE = "understat"


def _f(x: Any) -> float:
    return float(x) if x not in (None, "") else float("nan")


def _i(x: Any) -> int:
    return int(float(x))


def normalise_league(
    payload: bytes,
    start_year: int,
    teams: TeamResolver,
    *,
    bronze_key: str = "",
    lag_hours: float = 24.0,
) -> dict[str, pd.DataFrame]:
    data = LeagueData.model_validate(json.loads(payload))
    season = season_label(start_year)
    names = {d.h.title for d in data.dates} | {d.a.title for d in data.dates}
    uid = teams.uids(names | {t.title for t in data.teams.values()})

    matches = pd.DataFrame(
        [
            {
                "season": season,
                "understat_match_id": int(d.id),
                "kickoff_at": pd.Timestamp(d.datetime, tz="UTC"),
                "home_team": uid[d.h.title],
                "away_team": uid[d.a.title],
                "understat_home_id": int(d.h.id),
                "understat_away_id": int(d.a.id),
                "is_result": d.isResult,
                "home_goals": _i(d.goals.h) if d.isResult and d.goals.h is not None else pd.NA,
                "away_goals": _i(d.goals.a) if d.isResult and d.goals.a is not None else pd.NA,
                "home_xg": _f(d.xG.h) if d.isResult else float("nan"),
                "away_xg": _f(d.xG.a) if d.isResult else float("nan"),
            }
            for d in data.dates
        ]
    )
    matches = matches.astype({"home_goals": "Int64", "away_goals": "Int64"})
    matches["event_at"] = matches["kickoff_at"]
    matches["observed_at"] = matches["kickoff_at"] + timedelta(hours=lag_hours)
    matches["source"] = SOURCE
    matches["bronze_key"] = bronze_key

    team_rows = []
    for t in data.teams.values():
        for h in t.history:
            team_rows.append(
                {
                    "season": season,
                    "team": uid[t.title],
                    "understat_team_id": int(t.id),
                    "kickoff_at": pd.Timestamp(h.date, tz="UTC"),
                    "venue": h.h_a,
                    "xg": h.xG,
                    "xga": h.xGA,
                    "npxg": h.npxG,
                    "npxga": h.npxGA,
                    "ppda_att": h.ppda.att,
                    "ppda_def": h.ppda.def_,
                    "ppda_allowed_att": h.ppda_allowed.att,
                    "ppda_allowed_def": h.ppda_allowed.def_,
                    "deep": h.deep,
                    "deep_allowed": h.deep_allowed,
                    "scored": h.scored,
                    "missed": h.missed,
                }
            )
    team_match = pd.DataFrame(team_rows)
    if not team_match.empty:
        team_match["event_at"] = team_match["kickoff_at"]
        team_match["observed_at"] = team_match["kickoff_at"] + timedelta(hours=lag_hours)
        team_match["source"] = SOURCE

    players = pd.DataFrame(
        [
            {
                "season": season,
                "understat_player_id": int(p.id),
                "player_name": html.unescape(p.player_name),
                "team_titles": p.team_title,
                "position": p.position,
                "games": _i(p.games),
                "minutes": _i(p.time),
                "goals": _i(p.goals),
                "xg": _f(p.xG),
                "assists": _i(p.assists),
                "xa": _f(p.xA),
                "shots": _i(p.shots),
                "key_passes": _i(p.key_passes),
            }
            for p in data.players
        ]
    )
    return {"us_match": matches, "us_team_match": team_match, "us_player_season": players}


def normalise_match(
    payload: bytes, match: pd.Series, *, bronze_key: str = "", lag_hours: float = 24.0
) -> dict[str, pd.DataFrame]:
    """``match`` is the ``us_match`` row (teams, kickoff) this payload belongs to."""
    data = MatchData.model_validate(json.loads(payload))
    mid = int(match["understat_match_id"])
    side_team = {"h": match["home_team"], "a": match["away_team"]}
    event_at = match["kickoff_at"]
    observed_at = event_at + timedelta(hours=lag_hours)

    shots = pd.DataFrame(
        [
            {
                "season": match["season"],
                "understat_match_id": mid,
                "shot_id": int(s.id),
                "minute": _i(s.minute),
                "side": s.h_a,
                "team": side_team[s.h_a],
                "understat_player_id": int(s.player_id),
                "player_name": html.unescape(s.player),
                "assisted_by": html.unescape(s.player_assisted) if s.player_assisted else None,
                "x": _f(s.X),
                "y": _f(s.Y),
                "xg": _f(s.xG),
                "result": s.result,
                "situation": s.situation,
                "shot_type": s.shotType,
                "last_action": s.lastAction,
            }
            for side in ("h", "a")
            for s in data.shots.get(side, [])
        ]
    )
    roster = pd.DataFrame(
        [
            {
                "season": match["season"],
                "understat_match_id": mid,
                "understat_player_id": int(r.player_id),
                "player_name": html.unescape(r.player),
                "side": r.h_a,
                "team": side_team[r.h_a],
                "position": r.position,
                "minutes": _i(r.time),
                "goals": _i(r.goals),
                "own_goals": _i(r.own_goals),
                "shots": _i(r.shots),
                "xg": _f(r.xG),
                "xa": _f(r.xA),
                "assists": _i(r.assists),
                "key_passes": _i(r.key_passes),
                "yellow_cards": _i(r.yellow_card),
                "red_cards": _i(r.red_card),
                "roster_in": _i(r.roster_in),
                "roster_out": _i(r.roster_out),
            }
            for side in ("h", "a")
            for r in data.rosters.get(side, {}).values()
        ]
    )
    for df in (shots, roster):
        if not df.empty:
            df["event_at"] = event_at
            df["observed_at"] = observed_at
            df["source"] = SOURCE
            df["bronze_key"] = bronze_key
    return {"fact_shot": shots, "fact_player_match_understat": roster}
