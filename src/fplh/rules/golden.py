"""Golden tests: official event statistics in → official ``total_points`` out, exactly.

Two input routes produce the same normalised frame (one row per element × fixture):

* :func:`load_vaastav` for vaastav/Fantasy-Premier-League ``merged_gw.csv`` (2025/26);
* :func:`history_from_fpl` for our own ``bootstrap-static`` + ``element-summary`` bronze
  pulls (2026/27 onward).

ARCHITECTURE.md §9.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from fplh.rules.bonus import assign_bonus
from fplh.rules.config import Rules, season_slug
from fplh.rules.engine import COMPONENTS, DEFENSIVE_COLUMNS, EVENT_COLUMNS, defensive_count
from fplh.rules.engine import score as score_points

VAASTAV_BASE = "https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master/data"
KEY = ["element", "fixture"]
OPTIONAL_CHECK_COLUMNS = ("bps", "clean_sheets", "defensive_contribution")


def vaastav_url(season: str, base: str = VAASTAV_BASE) -> str:
    return f"{base.rstrip('/')}/{season_slug(season).replace('_', '-')}/gws/merged_gw.csv"


def golden_cache_path(cache_dir: Path, season: str) -> Path:
    return cache_dir / "golden" / season_slug(season) / "merged_gw.csv"


@dataclass
class GoldenFrame:
    season: str
    frame: pd.DataFrame
    duplicates_dropped: int = 0


def _normalise(season: str, df: pd.DataFrame) -> GoldenFrame:
    if "round" not in df.columns and "GW" in df.columns:
        df = df.rename(columns={"GW": "round"})
    required = [*KEY, "round", "position", "total_points", *EVENT_COLUMNS]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"golden frame is missing columns: {missing}")
    df = df.copy()
    # Some sources carry an "AM" style position in later seasons; fail loudly instead.
    df["position"] = df["position"].astype(str).str.upper().replace({"GKP": "GK"})
    # Exact duplicate element × fixture rows are a known scrape artefact in vaastav
    # 2025/26 (10 rows). Drop exact duplicates only; conflicting duplicates are an error.
    before = len(df)
    df = df.drop_duplicates()
    conflicting = df.duplicated(KEY, keep=False)
    if conflicting.any():
        raise ValueError(
            f"{int(conflicting.sum())} conflicting duplicate element×fixture rows, e.g.\n"
            f"{df.loc[conflicting, [*KEY, 'total_points']].head()}"
        )
    return GoldenFrame(season, df.reset_index(drop=True), before - len(df))


def load_vaastav(season: str, path: Path) -> GoldenFrame:
    return _normalise(season, pd.read_csv(path))


def history_from_fpl(
    season: str, bootstrap: dict[str, Any], summaries: Iterable[dict[str, Any]]
) -> GoldenFrame:
    """Flatten ``element-summary`` history rows, taking positions from ``bootstrap``."""
    pos_names = {t["id"]: t["singular_name_short"] for t in bootstrap["element_types"]}
    position = {e["id"]: pos_names[e["element_type"]] for e in bootstrap["elements"]}
    rows = [row for s in summaries for row in s.get("history", [])]
    df = pd.DataFrame(rows)
    if df.empty:
        return GoldenFrame(season, df)
    df["position"] = df["element"].map(position)
    if df["position"].isna().any():
        raise ValueError("history rows for elements missing from bootstrap-static")
    return _normalise(season, df)


def from_silver(season: str, player_match: pd.DataFrame) -> GoldenFrame:
    """Golden frame from silver ``fact_player_match`` (every season, any source)."""
    df = player_match[player_match["season"] == season].rename(
        columns={"fpl_fixture_id": "fixture"}
    )
    if df["defensive_contribution"].isna().all():
        df = df.drop(columns=["defensive_contribution"])  # not recorded that season
    for c in DEFENSIVE_COLUMNS:
        df[c] = df[c].fillna(0).astype("int64")
    return GoldenFrame(season, df.reset_index(drop=True))


def load_known_exceptions(path: Path, season: str) -> set[tuple[int, int]]:
    """``(element, fixture)`` pairs excused from the exact-match gate, with a reason each."""
    if not path.exists():
        return set()
    entries = yaml.safe_load(path.read_text()) or []
    out: set[tuple[int, int]] = set()
    for e in entries:
        if not e.get("reason"):
            raise ValueError(f"known exception without a reason: {e}")
        if season_slug(str(e["season"])) == season_slug(season):
            out.add((int(e["element"]), int(e["fixture"])))
    return out


@dataclass
class GoldenReport:
    season: str
    rows: int
    duplicates_dropped: int
    excused: int
    points_mismatches: pd.DataFrame
    checks: dict[str, int] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.points_mismatches.empty and not any(self.checks.values())

    def summary(self) -> str:
        lines = [
            f"season {self.season}: {self.rows} player-fixture rows "
            f"({self.duplicates_dropped} exact duplicates dropped, {self.excused} excused)",
            f"  total_points mismatches: {len(self.points_mismatches)}",
        ]
        lines += [f"  {name} mismatches: {n}" for name, n in self.checks.items()]
        return "\n".join(lines)


def check(
    golden: GoldenFrame,
    rules: Rules,
    excused: set[tuple[int, int]] | None = None,
    *,
    bonus_from_bps: bool = True,
) -> GoldenReport:
    """Score ``golden`` and compare with official points and derived flags.

    ``bonus_from_bps`` re-derives bonus from BPS; it needs every player of each fixture,
    so disable it for partial frames.
    """
    df = golden.frame
    excused = excused or set()
    keep = ~pd.Series(list(zip(df["element"], df["fixture"], strict=True))).isin(excused)
    n_excused = int((~keep).sum())
    df = df[keep.to_numpy()].reset_index(drop=True)

    parts = score_points(df, rules)
    bad = parts["total"] != df["total_points"]
    mismatches = pd.concat(
        [df.loc[bad, [*KEY, "round", "position", "total_points"]], parts.loc[bad]], axis=1
    )

    checks: dict[str, int] = {}
    played = df["minutes"] > 0
    if "clean_sheets" in df.columns:
        cs_rule = rules.clean_sheet
        ours = played & (df["minutes"] >= cs_rule.min_minutes) & (df["goals_conceded"] == 0)
        checks["clean_sheet_flag"] = int((ours.astype(int) != df["clean_sheets"]).sum())
    if "defensive_contribution" in df.columns and all(c in df for c in DEFENSIVE_COLUMNS):
        dc = rules.defensive_contribution
        outfield = (df["position"] != "GK").to_numpy()
        count = np.where(
            (df["position"] == "DEF").to_numpy(),
            defensive_count(dc.DEF, df),
            defensive_count(dc.MID_FWD, df),
        )
        official = df["defensive_contribution"].to_numpy()
        checks["defensive_count"] = int((outfield & (count != official)).sum())
    if bonus_from_bps and "bps" in df.columns:
        ours_bonus = assign_bonus(df["bps"], df["fixture"], rules.bonus.ranks, eligible=played)
        checks["bonus_from_bps"] = int((ours_bonus != df["bonus"]).sum())

    return GoldenReport(
        season=golden.season,
        rows=len(df),
        duplicates_dropped=golden.duplicates_dropped,
        excused=n_excused,
        points_mismatches=mismatches,
        checks=checks,
    )


__all__ = [
    "COMPONENTS",
    "GoldenFrame",
    "GoldenReport",
    "check",
    "history_from_fpl",
    "load_known_exceptions",
    "load_vaastav",
    "vaastav_url",
]
