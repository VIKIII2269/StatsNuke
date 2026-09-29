"""Helpers shared by the silver normalisers."""

from __future__ import annotations

import io
from collections.abc import Sequence
from datetime import timedelta

import pandas as pd

from fplh.lake.bronze import latest_by_params, parse_key, read_bronze
from fplh.lake.storage import Lake

POSITION_BY_ELEMENT_TYPE = {1: "GK", 2: "DEF", 3: "MID", 4: "FWD"}


def season_label(start_year: int) -> str:
    """2025 → ``2025-26`` (the silver partition label)."""
    return f"{start_year}-{(start_year + 1) % 100:02d}"


def read_csv_bytes(payload: bytes) -> pd.DataFrame:
    """CSV with a UTF-8 → cp1252 fallback (older football files are Windows-encoded)."""
    for encoding in ("utf-8-sig", "cp1252"):
        try:
            return pd.read_csv(io.BytesIO(payload), encoding=encoding, low_memory=False)
        except UnicodeDecodeError:
            continue
    raise ValueError("CSV is neither UTF-8 nor cp1252")


def latest_payload(
    lake: Lake, source: str, endpoint: str, **params: str
) -> tuple[str, bytes] | None:
    """(bronze key, payload) of the latest object whose params include ``params``."""
    match = None
    for p, key in latest_by_params(lake, source, endpoint).items():
        if params.items() <= dict(p).items():
            match = (
                key
                if match is None or parse_key(key).observed_at > parse_key(match).observed_at
                else match
            )
    if match is None:
        return None
    return match, read_bronze(lake, match)[1]


def utc(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, utc=True)


def observed_after(event_at: pd.Series, lag_hours: float) -> pd.Series:
    """Backfilled facts: o_f = e_f + ℓ_s (ARCHITECTURE.md §6.2)."""
    return event_at + timedelta(hours=lag_hours)


def drop_exact_duplicates(df: pd.DataFrame, key: Sequence[str]) -> tuple[pd.DataFrame, int]:
    """Drop exact duplicate rows; raise if rows share ``key`` but differ elsewhere."""
    before = len(df)
    df = df.drop_duplicates()
    conflicting = df.duplicated(list(key), keep=False)
    if conflicting.any():
        sample = df.loc[conflicting, list(key)].head(6).to_dict("records")
        raise ValueError(f"{int(conflicting.sum())} conflicting duplicate rows on {key}: {sample}")
    return df.reset_index(drop=True), before - len(df)
