"""football-data.co.uk CSV → ``fd_match`` (results, stats, referee) and ``snap_odds``.

Bookmaker columns are discovered, not hard-coded: a 1X2 book is any prefix ``P`` with
``PH``, ``PD`` and ``PA`` columns; an over/under 2.5 book is any ``P>2.5`` / ``P<2.5``
pair. A trailing ``C`` marks closing prices (``PSCH``, ``AvgC>2.5``). Observation times
(ARCHITECTURE.md §6.2) follow football-data's collection practice: pre-match odds on the
Friday afternoon before weekend fixtures and the Tuesday afternoon before midweek ones
(16:00 UK); closing odds at kickoff; results 48 h after kickoff.
"""

from __future__ import annotations

import csv
import io
from datetime import datetime, time, timedelta
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from fplh.entities.teams import TeamResolver
from fplh.lake.silver.common import season_label

SOURCE = "football_data"
UK = ZoneInfo("Europe/London")
DEFAULT_KICKOFF = time(15, 0)
PREMATCH_COLLECTION = time(16, 0)

BOOK_NAMES = {
    "PS": "pinnacle",
    "P": "pinnacle",
    "B365": "bet365",
    "Max": "market_max",
    "BbMx": "market_max",
    "Avg": "market_avg",
    "BbAv": "market_avg",
}
# Bookmaker codes whose closing variant is written ``<code>C…`` (e.g. PSCH, B365C>2.5).
CLOSING_BASES = frozenset(BOOK_NAMES) | {
    "BW",
    "IW",
    "WH",
    "VC",
    "BFE",
    "1XB",
    "BMGM",
    "BV",
    "CL",
    "LB",
    "BFD",
    "BF",
}
MATCH_STATS = {
    "FTHG": "home_goals",
    "FTAG": "away_goals",
    "HTHG": "home_goals_ht",
    "HTAG": "away_goals_ht",
    "HS": "home_shots",
    "AS": "away_shots",
    "HST": "home_shots_on_target",
    "AST": "away_shots_on_target",
    "HC": "home_corners",
    "AC": "away_corners",
    "HF": "home_fouls",
    "AF": "away_fouls",
    "HY": "home_yellows",
    "AY": "away_yellows",
    "HR": "home_reds",
    "AR": "away_reds",
}
MATCH_COLUMNS: tuple[str, ...] = (
    "season",
    "division",
    "kickoff_at",
    "kickoff_time_imputed",
    "home_team",
    "away_team",
    *MATCH_STATS.values(),
    "referee",
    "event_at",
    "observed_at",
    "source",
    "bronze_key",
)
ODDS_COLUMNS: tuple[str, ...] = (
    "season",
    "division",
    "kickoff_at",
    "home_team",
    "away_team",
    "bookmaker",
    "book_code",
    "market",
    "line",
    "outcome",
    "price",
    "is_closing",
    "observed_at",
    "source",
    "bronze_key",
)


def read_football_data_csv(payload: bytes) -> tuple[pd.DataFrame, int]:
    """Parse a football-data CSV robustly. Returns (frame, ragged rows padded).

    Some files have rows with more fields than the header, trailing blank lines and
    Windows encodings; rows are padded rather than dropped.
    """
    for enc in ("utf-8-sig", "cp1252"):
        try:
            text = payload.decode(enc)
            break
        except UnicodeDecodeError:
            continue
    else:  # pragma: no cover
        raise ValueError("undecodable CSV")
    rows = [r for r in csv.reader(io.StringIO(text)) if any(c.strip() for c in r)]
    if not rows:
        return pd.DataFrame(), 0
    header = [h.strip() for h in rows[0]]
    width = max(len(r) for r in rows)
    ragged = sum(1 for r in rows[1:] if len(r) != len(header))
    header += [f"_extra_{i}" for i in range(width - len(header))]
    body = [r + [""] * (width - len(r)) for r in rows[1:]]
    df = pd.DataFrame(body, columns=header).replace("", np.nan)
    df = df.loc[
        :, [c for c in df.columns if c and not (c.startswith("_extra_") and df[c].isna().all())]
    ]
    return df.dropna(subset=[c for c in ("HomeTeam", "AwayTeam") if c in df.columns]), ragged


def _kickoffs(df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    dates = pd.to_datetime(df["Date"], dayfirst=True, format="mixed")
    has_time = "Time" in df.columns
    times = df["Time"] if has_time else pd.Series(np.nan, index=df.index)
    imputed = times.isna()
    local = [
        datetime.combine(d.date(), DEFAULT_KICKOFF if pd.isna(t) else time.fromisoformat(str(t)))
        for d, t in zip(dates, times, strict=True)
    ]
    kickoff = pd.Series([pd.Timestamp(x, tz=UK).tz_convert("UTC") for x in local], index=df.index)
    return kickoff, imputed


def prematch_observed_at(kickoff_utc: pd.Timestamp) -> pd.Timestamp:
    """16:00 UK on the Friday (weekend round: Fri–Mon) or Tuesday (midweek: Tue–Thu)
    on or before kickoff. If that isn't strictly before kickoff, the price is treated as
    known only at kickoff (unusable before any deadline)."""
    local = kickoff_utc.tz_convert(UK)
    wd = local.weekday()  # Mon=0
    anchor = 4 if wd in (4, 5, 6, 0) else 1  # Friday or Tuesday
    back = (wd - anchor) % 7
    day = (local - timedelta(days=back)).date()
    t = pd.Timestamp(datetime.combine(day, PREMATCH_COLLECTION), tz=UK).tz_convert("UTC")
    return t if t < kickoff_utc else kickoff_utc


def _book(prefix: str) -> tuple[str, str, bool]:
    """(bookmaker name, code, is_closing) for a column prefix."""
    closing = False
    code = prefix
    if prefix.endswith("C") and prefix[:-1] in CLOSING_BASES:
        code, closing = prefix[:-1], True
    return BOOK_NAMES.get(code, code.lower()), code, closing


def odds_long(df: pd.DataFrame) -> pd.DataFrame:
    cols = set(df.columns)
    parts: list[pd.DataFrame] = []
    triples = sorted(
        p[:-1] for p in cols if p.endswith("H") and p[:-1] and {p[:-1] + "D", p[:-1] + "A"} <= cols
    )
    for prefix in triples:
        name, code, closing = _book(prefix)
        for suffix, outcome in (("H", "home"), ("D", "draw"), ("A", "away")):
            parts.append(
                pd.DataFrame(
                    {
                        "row": df.index,
                        "bookmaker": name,
                        "book_code": code,
                        "market": "1x2",
                        "line": np.nan,
                        "outcome": outcome,
                        "price": pd.to_numeric(df[prefix + suffix], errors="coerce"),
                        "is_closing": closing,
                    }
                )
            )
    for col in sorted(c for c in cols if c.endswith(">2.5") and c[:-4] + "<2.5" in cols):
        prefix = col[:-4]
        name, code, closing = _book(prefix)
        for c, outcome in ((col, "over"), (prefix + "<2.5", "under")):
            parts.append(
                pd.DataFrame(
                    {
                        "row": df.index,
                        "bookmaker": name,
                        "book_code": code,
                        "market": "total",
                        "line": 2.5,
                        "outcome": outcome,
                        "price": pd.to_numeric(df[c], errors="coerce"),
                        "is_closing": closing,
                    }
                )
            )
    if not parts:
        return pd.DataFrame(
            columns=[
                "row",
                "bookmaker",
                "book_code",
                "market",
                "line",
                "outcome",
                "price",
                "is_closing",
            ]
        )
    out = pd.concat(parts, ignore_index=True)
    return out[out["price"] > 1.0]


def normalise_file(
    payload: bytes,
    *,
    start_year: int,
    division: str,
    teams: TeamResolver,
    bronze_key: str = "",
    result_lag_hours: float = 48.0,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, int]]:
    """(fd_match, snap_odds, notes) for one division-season file."""
    df, ragged = read_football_data_csv(payload)
    notes = {"ragged_rows_padded": ragged, "rows": len(df)}
    if df.empty:
        return (
            pd.DataFrame(columns=list(MATCH_COLUMNS)),
            pd.DataFrame(columns=list(ODDS_COLUMNS)),
            notes,
        )
    teams.add_canonical(pd.concat([df["HomeTeam"], df["AwayTeam"]]).astype(str).str.strip())
    kickoff, imputed = _kickoffs(df)
    notes["kickoff_time_imputed"] = int(imputed.sum())
    match = pd.DataFrame(
        {
            "season": season_label(start_year),
            "division": division,
            "kickoff_at": kickoff,
            "kickoff_time_imputed": imputed.to_numpy(),
            "home_team": df["HomeTeam"].astype(str).str.strip().map(teams.uid),
            "away_team": df["AwayTeam"].astype(str).str.strip().map(teams.uid),
        },
        index=df.index,
    )
    for src, dst in MATCH_STATS.items():
        match[dst] = (
            pd.to_numeric(df[src], errors="coerce").astype("Int64")
            if src in df
            else pd.array([pd.NA] * len(df), dtype="Int64")
        )
    match["referee"] = (
        df["Referee"].astype("string")
        if "Referee" in df
        else pd.Series(pd.NA, index=df.index, dtype="string")
    )
    match["event_at"] = kickoff
    match["observed_at"] = kickoff + timedelta(hours=result_lag_hours)
    match["source"] = SOURCE
    match["bronze_key"] = bronze_key

    odds = odds_long(df)
    odds = odds.join(
        match[["season", "division", "kickoff_at", "home_team", "away_team"]], on="row"
    )
    prematch_obs = match["kickoff_at"].map(prematch_observed_at)
    odds["observed_at"] = np.where(
        odds["is_closing"], odds["kickoff_at"], odds["row"].map(prematch_obs)
    )
    odds["observed_at"] = pd.to_datetime(odds["observed_at"], utc=True)
    odds["source"] = SOURCE
    odds["bronze_key"] = bronze_key
    return (
        match[list(MATCH_COLUMNS)].reset_index(drop=True),
        odds[list(ODDS_COLUMNS)].reset_index(drop=True),
        notes,
    )


FAMILIES = {
    "pinnacle_closing_1x2": ("PSCH", "PSCD", "PSCA"),
    "market_avg_closing_1x2": ("AvgCH", "AvgCD", "AvgCA"),
    "bet365_closing_1x2": ("B365CH", "B365CD", "B365CA"),
    "pinnacle_prematch_1x2": ("PSH", "PSD", "PSA"),
    "market_avg_prematch_1x2": ("AvgH", "AvgD", "AvgA"),
    "pinnacle_closing_ou25": ("PC>2.5", "PC<2.5"),
    "market_avg_closing_ou25": ("AvgC>2.5", "AvgC<2.5"),
    "market_avg_prematch_ou25": ("Avg>2.5", "Avg<2.5"),
}


def odds_column_report(files: dict[str, bytes]) -> pd.DataFrame:
    """Share of matches with a complete price set per odds family, per season code."""
    rows = []
    for season, payload in sorted(files.items()):
        df, _ = read_football_data_csv(payload)
        row: dict[str, object] = {"season": season, "matches": len(df)}
        for family, cols in FAMILIES.items():
            if len(df) and all(c in df.columns for c in cols):
                present = df[list(cols)].apply(pd.to_numeric, errors="coerce").notna().all(axis=1)
                row[family] = round(float(present.mean()), 3)
            else:
                row[family] = 0.0
        rows.append(row)
    return pd.DataFrame(rows)
