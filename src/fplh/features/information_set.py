"""The information set 𝓘(D) = {f : o_f ≤ D} (ARCHITECTURE.md §6.2).

Feature builders receive an :class:`InformationSet` and nothing else: no connection, no
path, no raw frame. Every fact and snapshot table it serves is pre-filtered to
``observed_at <= deadline``, and the latest observation time actually served is
recorded so a run can prove ``max_observed_at <= D``.

Dimension tables have no observation time, so only their schedule/identity columns are
exposed: final scores in ``dim_fixture`` or end-of-season player registrations would
otherwise leak the future. Outcomes always come from fact tables.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

import duckdb
import pandas as pd

from fplh.lake.parquet import read_table
from fplh.lake.storage import Lake

# Tables filtered by observed_at.
TIME_INDEXED = (
    "fact_player_match",
    "fact_player_match_understat",
    "fact_shot",
    "fd_match",
    "us_match",
    "us_team_match",
    "snap_odds",
    "snap_fpl_player",
    "fpl_event",
)
# Dimension columns that are knowable in advance (schedule and identity only).
DIMENSION_COLUMNS: dict[str, tuple[str, ...]] = {
    "dim_fixture": (
        "fixture_uid",
        "season",
        "home_team",
        "away_team",
        "kickoff_at",
        "round",
        "fpl_fixture_id",
    ),
    "dim_team": ("team_uid",),
    "dim_player": ("player_uid", "code", "first_name", "second_name", "web_name"),
}


class LeakageError(RuntimeError):
    pass


@dataclass
class SilverStore:
    """Silver tables, loaded lazily from a lake or given in memory (tests)."""

    lake: Lake | None = None
    frames: dict[str, pd.DataFrame] = field(default_factory=dict)

    def get(self, name: str) -> pd.DataFrame:
        if name not in self.frames:
            if self.lake is None:
                return pd.DataFrame()
            self.frames[name] = read_table(self.lake, f"silver/{name}")
        return self.frames[name]

    @classmethod
    def from_frames(cls, frames: Mapping[str, pd.DataFrame]) -> SilverStore:
        return cls(None, dict(frames))


class InformationSet:
    def __init__(self, deadline: pd.Timestamp, store: SilverStore) -> None:
        if deadline.tzinfo is None:
            raise ValueError("deadline must be timezone-aware")
        self._deadline = deadline.tz_convert("UTC")
        self._store = store
        self._cache: dict[str, pd.DataFrame] = {}
        self._max_observed: pd.Timestamp | None = None
        self._con: duckdb.DuckDBPyConnection | None = None

    @classmethod
    def at(cls, deadline: pd.Timestamp, store: SilverStore) -> InformationSet:
        return cls(deadline, store)

    @property
    def deadline(self) -> pd.Timestamp:
        return self._deadline

    @property
    def max_observed_at(self) -> pd.Timestamp | None:
        """Latest ``observed_at`` of any row served so far (None if none)."""
        return self._max_observed

    def table(self, name: str) -> pd.DataFrame:
        """A copy of ``name`` restricted to what was knowable at the deadline."""
        if name not in self._cache:
            self._cache[name] = self._load(name)
        return self._cache[name].copy()

    def _load(self, name: str) -> pd.DataFrame:
        df = self._store.get(name)
        if name in DIMENSION_COLUMNS:
            cols = [c for c in DIMENSION_COLUMNS[name] if c in df.columns]
            return (
                df[cols].copy()
                if not df.empty
                else pd.DataFrame(columns=list(DIMENSION_COLUMNS[name]))
            )
        if name not in TIME_INDEXED:
            raise LeakageError(f"{name!r} has no observation time and is not exposed")
        if df.empty:
            return df
        visible = df[df["observed_at"] <= self._deadline].copy()
        if not visible.empty:
            latest = visible["observed_at"].max()
            if self._max_observed is None or latest > self._max_observed:
                self._max_observed = latest
        return visible

    def sql(self, query: str) -> pd.DataFrame:
        """Run SQL against the filtered views (every exposed table, as its own name)."""
        if self._con is None:
            self._con = duckdb.connect()
            for name in (*TIME_INDEXED, *DIMENSION_COLUMNS):
                frame = self.table(name)
                if not frame.empty:
                    self._con.register(name, frame)
        result: pd.DataFrame = self._con.execute(query).df()
        return result

    def assert_no_leakage(self) -> None:
        if self._max_observed is not None and self._max_observed > self._deadline:
            raise LeakageError(
                f"served a fact observed at {self._max_observed} > deadline {self._deadline}"
            )
