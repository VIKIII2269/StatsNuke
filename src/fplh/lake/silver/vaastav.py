"""vaastav bronze → ``fpl_fixture``, ``fpl_player_season`` and ``fact_player_match``.

Team per player-fixture comes from the fixture, never from end-of-season
``players_raw.team`` (which is wrong for players transferred mid-season). Seasons
without ``fixtures.csv`` reconstruct home/away teams from the player rows: the home
team is the ``opponent_team`` of rows with ``was_home = False``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from fplh.entities.teams import TeamResolver
from fplh.lake.silver.common import (
    POSITION_BY_ELEMENT_TYPE,
    drop_exact_duplicates,
    latest_payload,
    observed_after,
    read_csv_bytes,
    season_label,
    utc,
)
from fplh.lake.storage import Lake

SOURCE = "vaastav"

STAT_COLUMNS: tuple[str, ...] = (
    "minutes",
    "goals_scored",
    "assists",
    "clean_sheets",
    "goals_conceded",
    "own_goals",
    "penalties_saved",
    "penalties_missed",
    "yellow_cards",
    "red_cards",
    "saves",
    "bonus",
    "bps",
    "total_points",
)
OPTIONAL_INT_COLUMNS: tuple[str, ...] = (
    "starts",
    "clearances_blocks_interceptions",
    "tackles",
    "recoveries",
    "defensive_contribution",
    "value",
    "selected",
    "transfers_in",
    "transfers_out",
)
OPTIONAL_FLOAT_COLUMNS: tuple[str, ...] = (
    "expected_goals",
    "expected_assists",
    "expected_goals_conceded",
    "influence",
    "creativity",
    "threat",
    "ict_index",
)
PLAYER_MATCH_COLUMNS: tuple[str, ...] = (
    "season",
    "element",
    "code",
    "position",
    "fpl_fixture_id",
    "round",
    "kickoff_at",
    "was_home",
    "team",
    "opponent",
    *STAT_COLUMNS,
    *OPTIONAL_INT_COLUMNS,
    *OPTIONAL_FLOAT_COLUMNS,
    "event_at",
    "observed_at",
    "source",
    "bronze_key",
)
FIXTURE_COLUMNS: tuple[str, ...] = (
    "season",
    "fpl_fixture_id",
    "round",
    "kickoff_at",
    "home_team",
    "away_team",
    "home_goals",
    "away_goals",
    "event_at",
    "observed_at",
    "source",
)


@dataclass
class VaastavSeason:
    season: str
    fixtures: pd.DataFrame
    players: pd.DataFrame
    player_match: pd.DataFrame
    notes: dict[str, int] = field(default_factory=dict)


def _team_names(
    master: pd.DataFrame, start_year: int, season_teams: pd.DataFrame | None
) -> dict[int, str]:
    """FPL team id → name: the season's own teams.csv, else master_team_list."""
    if season_teams is not None and {"id", "name"} <= set(season_teams.columns):
        return dict(
            zip(season_teams["id"].astype(int), season_teams["name"].astype(str), strict=True)
        )
    rows = master[master["season"] == season_label(start_year)]
    if rows.empty:
        raise ValueError(f"no team names for {season_label(start_year)}")
    return dict(zip(rows["team"].astype(int), rows["team_name"].astype(str), strict=True))


def _drop_postponed_placeholders(gw: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """A postponed fixture can appear twice: an unplayed placeholder in the original
    gameweek (no score) and the real row in the rescheduled one. Drop placeholders only
    where the played row exists (e.g. 2019/20 fixture 275, GW29 → GW39)."""
    unplayed = gw["team_h_score"].isna()
    played_keys = set(map(tuple, gw.loc[~unplayed, ["element", "fixture"]].to_numpy()))
    keys = list(map(tuple, gw[["element", "fixture"]].to_numpy()))
    placeholder = unplayed & pd.Series([k in played_keys for k in keys], index=gw.index)
    return gw[~placeholder].copy(), int(placeholder.sum())


def _fixture_frame(gw: pd.DataFrame, fixtures: pd.DataFrame | None) -> pd.DataFrame:
    """One row per FPL fixture id with FPL team ids, kickoff, round and score."""
    rows = gw.drop_duplicates("fixture")
    derived = pd.DataFrame(
        {
            "fpl_fixture_id": rows["fixture"].astype(int),
            "round": rows["round"].astype(int),
            "kickoff_time": rows["kickoff_time"],
            "home_goals": rows["team_h_score"],
            "away_goals": rows["team_a_score"],
        }
    )
    home = gw.loc[~gw["was_home"], ["fixture", "opponent_team"]].drop_duplicates("fixture")
    away = gw.loc[gw["was_home"], ["fixture", "opponent_team"]].drop_duplicates("fixture")
    derived = derived.merge(
        home.rename(columns={"fixture": "fpl_fixture_id", "opponent_team": "team_h"}),
        how="left",
    ).merge(
        away.rename(columns={"fixture": "fpl_fixture_id", "opponent_team": "team_a"}),
        how="left",
    )
    if fixtures is not None and {"id", "team_h", "team_a"} <= set(fixtures.columns):
        official = fixtures[["id", "team_h", "team_a"]].rename(columns={"id": "fpl_fixture_id"})
        derived = derived.drop(columns=["team_h", "team_a"]).merge(official, how="left")
    if derived[["team_h", "team_a"]].isna().any().any():
        missing = derived.loc[derived["team_h"].isna() | derived["team_a"].isna(), "fpl_fixture_id"]
        raise ValueError(f"cannot determine teams for fixtures {missing.tolist()[:10]}")
    return derived.astype({"team_h": int, "team_a": int})


def normalise_season(
    start_year: int,
    gw: pd.DataFrame,
    players_raw: pd.DataFrame,
    master: pd.DataFrame,
    teams: TeamResolver,
    *,
    fixtures: pd.DataFrame | None = None,
    season_teams: pd.DataFrame | None = None,
    bronze_key: str = "",
    lag_hours: float = 33.0,
) -> VaastavSeason:
    season = season_label(start_year)
    notes: dict[str, int] = {}
    gw = gw.drop(columns=["xP", "name"], errors="ignore")
    if "round" not in gw.columns:
        gw = gw.rename(columns={"GW": "round"})
    gw = gw.drop(columns=["GW"], errors="ignore")
    gw["was_home"] = gw["was_home"].astype(str).str.lower().isin(["true", "1"])

    # Position: per-row (2020/21+) or players_raw element_type; FPL positions are fixed
    # within a season. Assistant-manager rows (2024/25 chip, element_type 5 / "AM") are
    # not player events and are dropped.
    pos_by_element = players_raw.set_index("id")["element_type"].map(POSITION_BY_ELEMENT_TYPE)
    if "position" in gw.columns:
        gw["position"] = gw["position"].replace({"GKP": "GK"})
    else:
        gw["position"] = gw["element"].map(pos_by_element)
    is_manager = gw["position"].isna() | (gw["position"] == "AM")
    notes["manager_rows_dropped"] = int(is_manager.sum())
    gw = gw[~is_manager].copy()
    gw, notes["postponed_placeholders_dropped"] = _drop_postponed_placeholders(gw)
    gw, notes["exact_duplicates_dropped"] = drop_exact_duplicates(gw, ["element", "fixture"])

    fx = _fixture_frame(gw, fixtures)
    names = _team_names(master, start_year, season_teams)
    team_uid = teams.uids(names.values())
    fx["home_team"] = fx["team_h"].map(names).map(team_uid)
    fx["away_team"] = fx["team_a"].map(names).map(team_uid)
    fx["kickoff_at"] = utc(fx["kickoff_time"])
    fx["event_at"] = fx["kickoff_at"]
    fx["observed_at"] = observed_after(fx["event_at"], lag_hours)
    fx["season"] = season
    fx["source"] = SOURCE

    rows = gw.merge(
        fx[["fpl_fixture_id", "home_team", "away_team", "kickoff_at", "event_at", "observed_at"]],
        left_on="fixture",
        right_on="fpl_fixture_id",
        how="left",
        validate="many_to_one",
    )
    rows["team"] = np.where(rows["was_home"], rows["home_team"], rows["away_team"])
    rows["opponent"] = np.where(rows["was_home"], rows["away_team"], rows["home_team"])
    code = players_raw.set_index("id")["code"]
    rows["code"] = rows["element"].map(code)
    if rows["code"].isna().any():
        raise ValueError(
            f"{int(rows['code'].isna().sum())} rows with an element missing from players_raw"
        )
    rows["season"] = season
    rows["source"] = SOURCE
    rows["bronze_key"] = bronze_key
    for c in OPTIONAL_INT_COLUMNS:
        rows[c] = (
            pd.to_numeric(rows[c], errors="coerce").astype("Int64")
            if c in rows
            else pd.array([pd.NA] * len(rows), dtype="Int64")
        )
    for c in OPTIONAL_FLOAT_COLUMNS:
        rows[c] = pd.to_numeric(rows[c], errors="coerce").astype("float64") if c in rows else np.nan
    for c in STAT_COLUMNS:
        rows[c] = rows[c].astype("int64")
    rows = rows.astype(
        {"element": "int64", "code": "int64", "round": "int64", "fpl_fixture_id": "int64"}
    )
    player_match = rows[list(PLAYER_MATCH_COLUMNS)].sort_values(["fpl_fixture_id", "element"])

    players = pd.DataFrame(
        {
            "season": season,
            "element": players_raw["id"].astype("int64"),
            "code": players_raw["code"].astype("int64"),
            "first_name": players_raw["first_name"].astype(str),
            "second_name": players_raw["second_name"].astype(str),
            "web_name": players_raw["web_name"].astype(str),
            "position": players_raw["element_type"].map(POSITION_BY_ELEMENT_TYPE),
        }
    ).dropna(subset=["position"])
    fixtures_out = fx.astype({"home_goals": "Int64", "away_goals": "Int64"})[list(FIXTURE_COLUMNS)]
    return VaastavSeason(season, fixtures_out, players, player_match.reset_index(drop=True), notes)


def load_season(
    lake: Lake, start_year: int, teams: TeamResolver, lag_hours: float
) -> VaastavSeason:
    tag = f"{start_year}_{(start_year + 1) % 100:02d}"
    gw = latest_payload(lake, SOURCE, "merged_gw", season=tag)
    raw = latest_payload(lake, SOURCE, "players_raw", season=tag)
    master = latest_payload(lake, SOURCE, "master_team_list")
    if gw is None or raw is None or master is None:
        raise FileNotFoundError(f"vaastav {tag} not in bronze; run `fplh backfill vaastav`")
    fx = latest_payload(lake, SOURCE, "fixtures", season=tag)
    tm = latest_payload(lake, SOURCE, "teams", season=tag)
    return normalise_season(
        start_year,
        read_csv_bytes(gw[1]),
        read_csv_bytes(raw[1]),
        read_csv_bytes(master[1]),
        teams,
        fixtures=read_csv_bytes(fx[1]) if fx else None,
        season_teams=read_csv_bytes(tm[1]) if tm else None,
        bronze_key=gw[0],
        lag_hours=lag_hours,
    )
