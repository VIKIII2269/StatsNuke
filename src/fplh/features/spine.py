"""Feature spine: one row per (player, fixture, deadline, horizon) (ARCHITECTURE.md §6.5)."""

from __future__ import annotations

from collections.abc import Sequence

import duckdb
import pandas as pd

from fplh.features.information_set import InformationSet

DEADLINE_BEFORE_KICKOFF = pd.Timedelta(minutes=90)
SPINE_KEYS = ["player_uid", "fixture_uid", "deadline_at", "horizon"]


def historical_deadlines(dim_fixture: pd.DataFrame, season: str) -> pd.DataFrame:
    """Per FPL round: first kickoff − 90 min (FPL's deadline rule)."""
    fx = dim_fixture[(dim_fixture["season"] == season) & dim_fixture["round"].notna()]
    out = fx.groupby("round")["kickoff_at"].min().reset_index()
    out["deadline_at"] = out["kickoff_at"] - DEADLINE_BEFORE_KICKOFF
    out["round"] = out["round"].astype("int64")
    result: pd.DataFrame = out[["round", "deadline_at"]].sort_values("round").reset_index(drop=True)
    return result


def target_fixtures(info: InformationSet, horizon: int) -> pd.DataFrame:
    """Scheduled fixtures in the next ``horizon`` rounds after the deadline."""
    fx = info.table("dim_fixture")
    fx = fx[(fx["kickoff_at"] > info.deadline) & fx["round"].notna()]
    if fx.empty:
        return fx.assign(horizon=pd.Series(dtype="int64"))
    # Only the season of the next kickoff (later seasons' round numbers restart at 1).
    fx = fx[fx["season"] == fx.sort_values("kickoff_at")["season"].iloc[0]]
    first = int(fx["round"].min())
    fx = fx[fx["round"] < first + horizon].copy()
    fx["horizon"] = (fx["round"] - first + 1).astype("int64")
    result: pd.DataFrame = fx.reset_index(drop=True)
    return result


def current_squads(info: InformationSet) -> pd.DataFrame:
    """player_uid → team and position as last observed (a player's latest club)."""
    pm = info.table("fact_player_match")
    frames = []
    if not pm.empty:
        frames.append(pm[["player_uid", "team", "position", "season", "observed_at", "kickoff_at"]])
    snap = info.table("snap_fpl_player")
    if not snap.empty:
        s = snap.assign(
            player_uid="fpl:" + snap["code"].astype(str), kickoff_at=snap["observed_at"]
        )
        frames.append(s[["player_uid", "team", "position", "season", "observed_at", "kickoff_at"]])
    if not frames:
        return pd.DataFrame(columns=["player_uid", "team", "position", "season"])
    allf = pd.concat(frames, ignore_index=True).sort_values(["kickoff_at", "observed_at"])
    latest = allf.drop_duplicates("player_uid", keep="last")
    return latest[["player_uid", "team", "position", "season"]].reset_index(drop=True)


def build_spine(info: InformationSet, horizon: int = 1) -> pd.DataFrame:
    fx = target_fixtures(info, horizon)
    squads = current_squads(info)
    if fx.empty or squads.empty:
        return pd.DataFrame(
            columns=[*SPINE_KEYS, "team", "opponent", "was_home", "position", "kickoff_at"]
        )
    season = fx["season"].iloc[0]
    squads = squads[squads["season"] == season]  # last season's squads don't carry over
    sides = pd.concat(
        [
            fx.assign(team=fx["home_team"], opponent=fx["away_team"], was_home=True),
            fx.assign(team=fx["away_team"], opponent=fx["home_team"], was_home=False),
        ]
    )
    spine = squads.merge(
        sides[["fixture_uid", "team", "opponent", "was_home", "kickoff_at", "horizon"]], on="team"
    )
    spine["deadline_at"] = info.deadline
    cols = [*SPINE_KEYS, "team", "opponent", "was_home", "position", "kickoff_at"]
    result: pd.DataFrame = spine[cols].sort_values(SPINE_KEYS[:2]).reset_index(drop=True)
    return result


def asof_join(
    spine: pd.DataFrame,
    facts: pd.DataFrame,
    by: Sequence[str],
    columns: Sequence[str],
    *,
    spine_time: str = "deadline_at",
    fact_time: str = "observed_at",
) -> pd.DataFrame:
    """Latest fact per ``by`` observed at or before each spine row's time (DuckDB ASOF)."""
    con = duckdb.connect()
    con.register("spine", spine.reset_index(drop=True).assign(_row=range(len(spine))))
    con.register("facts", facts[[*by, fact_time, *columns]])
    on = " AND ".join(f"s.{b} = f.{b}" for b in by)
    cols = ", ".join(f"f.{c}" for c in columns)
    out = con.execute(
        f"SELECT s.*, {cols} FROM spine s ASOF LEFT JOIN facts f "
        f"ON {on} AND s.{spine_time} >= f.{fact_time} ORDER BY s._row"
    ).df()
    result: pd.DataFrame = out.drop(columns=["_row"])
    return result
