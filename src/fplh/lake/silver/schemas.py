"""Pandera contracts for silver tables (ARCHITECTURE.md §6.6).

Contracts check types, ranges, non-null keys and key uniqueness. Tables may carry extra
columns (strict=False) so a new upstream field doesn't break the build; the columns that
downstream code depends on are required.
"""

from __future__ import annotations

import pandas as pd
import pandera.pandas as pa

POSITIONS = ["GK", "DEF", "MID", "FWD"]


def _ts(nullable: bool = False) -> pa.Column:
    return pa.Column(pd.DatetimeTZDtype(tz="UTC"), nullable=nullable, coerce=True)


def _nonneg(dtype: object = "int64", nullable: bool = False) -> pa.Column:
    return pa.Column(dtype, pa.Check.ge(0), nullable=nullable, coerce=True)


OBSERVED_NOT_BEFORE_EVENT = pa.Check(
    lambda df: df["observed_at"] >= df["event_at"],
    name="observed_at >= event_at",
    element_wise=False,
)

SCHEMAS: dict[str, pa.DataFrameSchema] = {
    "fact_player_match": pa.DataFrameSchema(
        {
            "season": pa.Column(str),
            "player_uid": pa.Column(str),
            "fixture_uid": pa.Column(str),
            "element": pa.Column("int64", pa.Check.ge(1), coerce=True),
            "code": pa.Column("int64", coerce=True),
            "position": pa.Column(str, pa.Check.isin(POSITIONS)),
            "team": pa.Column(str),
            "opponent": pa.Column(str),
            "minutes": pa.Column("int64", pa.Check.in_range(0, 130), coerce=True),
            "goals_scored": _nonneg(),
            "assists": _nonneg(),
            "own_goals": _nonneg(),
            "saves": _nonneg(),
            "total_points": pa.Column("int64", coerce=True),
            "event_at": _ts(),
            "observed_at": _ts(),
        },
        checks=[OBSERVED_NOT_BEFORE_EVENT],
        unique=["season", "element", "fpl_fixture_id"],
        strict=False,
    ),
    "fpl_fixture": pa.DataFrameSchema(
        {
            "fixture_uid": pa.Column(str, unique=True),
            "fpl_fixture_id": pa.Column("int64", coerce=True),
            "home_team": pa.Column(str),
            "away_team": pa.Column(str),
            "kickoff_at": _ts(),
            "home_goals": _nonneg("Int64", nullable=True),
            "away_goals": _nonneg("Int64", nullable=True),
        },
        unique=["season", "fpl_fixture_id"],
        strict=False,
    ),
    "fd_match": pa.DataFrameSchema(
        {
            "division": pa.Column(str, pa.Check.isin(["E0", "E1"])),
            "home_team": pa.Column(str),
            "away_team": pa.Column(str),
            "kickoff_at": _ts(),
            "home_goals": _nonneg("Int64", nullable=True),
            "away_goals": _nonneg("Int64", nullable=True),
            "event_at": _ts(),
            "observed_at": _ts(),
        },
        checks=[OBSERVED_NOT_BEFORE_EVENT],
        unique=["season", "division", "home_team", "away_team"],
        strict=False,
    ),
    "snap_odds": pa.DataFrameSchema(
        {
            "home_team": pa.Column(str),
            "away_team": pa.Column(str),
            "market": pa.Column(str, pa.Check.isin(["1x2", "total"])),
            "outcome": pa.Column(str, pa.Check.isin(["home", "draw", "away", "over", "under"])),
            "price": pa.Column(float, pa.Check.gt(1.0), coerce=True),
            "is_closing": pa.Column(bool),
            "kickoff_at": _ts(),
            "observed_at": _ts(),
        },
        checks=[
            pa.Check(
                lambda df: df["observed_at"] <= df["kickoff_at"], name="odds observed by kickoff"
            )
        ],
        strict=False,
    ),
    "us_match": pa.DataFrameSchema(
        {"understat_match_id": pa.Column("int64", unique=True, coerce=True), "kickoff_at": _ts()},
        strict=False,
    ),
    "fact_shot": pa.DataFrameSchema(
        {
            "shot_id": pa.Column("int64", unique=True, coerce=True),
            "understat_match_id": pa.Column("int64", coerce=True),
            "minute": pa.Column("int64", pa.Check.in_range(0, 130), coerce=True),
            "xg": pa.Column(float, pa.Check.in_range(0.0, 1.0), coerce=True),
            "event_at": _ts(),
            "observed_at": _ts(),
        },
        checks=[OBSERVED_NOT_BEFORE_EVENT],
        strict=False,
    ),
    "fact_player_match_understat": pa.DataFrameSchema(
        {
            "understat_match_id": pa.Column("int64", coerce=True),
            "understat_player_id": pa.Column("int64", coerce=True),
            "minutes": pa.Column("int64", pa.Check.in_range(0, 130), coerce=True),
            "event_at": _ts(),
            "observed_at": _ts(),
        },
        checks=[OBSERVED_NOT_BEFORE_EVENT],
        unique=["understat_match_id", "understat_player_id"],
        strict=False,
    ),
    "dim_fixture": pa.DataFrameSchema(
        {"fixture_uid": pa.Column(str, unique=True), "kickoff_at": _ts()}, strict=False
    ),
    "dim_player": pa.DataFrameSchema(
        {
            "player_uid": pa.Column(str, unique=True),
            "code": pa.Column("int64", unique=True, coerce=True),
        },
        strict=False,
    ),
    "snap_fpl_player": pa.DataFrameSchema(
        {"element": pa.Column("int64", coerce=True), "observed_at": _ts()},
        unique=["element", "observed_at"],
        strict=False,
    ),
}


def validate(table: str, df: pd.DataFrame) -> pd.DataFrame:
    schema = SCHEMAS.get(table)
    if schema is None or df.empty:
        return df
    return schema.validate(df, lazy=True)
