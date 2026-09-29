"""Understat league and match data (EPL, 2014/15 onward).

Understat serves JSON to its own pages from two endpoints (both need the
``X-Requested-With: XMLHttpRequest`` header):

* ``/getLeagueData/EPL/{season_start_year}`` → ``dates`` (fixtures, results, xG),
  ``teams`` (per-match team history incl. PPDA) and ``players`` (season totals);
* ``/getMatchData/{match_id}`` → ``rosters`` and ``shots`` (plus rendered ``tmpl`` HTML, ignored).

Stage 1 refreshes league data; stage 2 fetches every finished match not yet in bronze.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any

from fplh.collectors.backfill import FileSpec
from fplh.lake.bronze import latest_by_params, read_bronze
from fplh.lake.storage import Lake

SOURCE = "understat"
FIRST_SEASON = 2014
LEAGUE = "EPL"
HEADERS = {"X-Requested-With": "XMLHttpRequest"}
DEFAULT_BASE = "https://understat.com"


def league_specs(
    seasons: list[int], current_season: int, base: str = DEFAULT_BASE
) -> list[FileSpec]:
    return [
        FileSpec(
            "league",
            f"{base.rstrip('/')}/getLeagueData/{LEAGUE}/{year}",
            {"league": LEAGUE, "season": str(year)},
            refresh=year == current_season,
            headers=HEADERS,
        )
        for year in seasons
    ]


def finished_match_ids(league_payload: dict[str, Any], since: datetime | None = None) -> list[int]:
    out = []
    for m in league_payload["dates"]:
        if not m.get("isResult"):
            continue
        played = (
            datetime.fromisoformat(m["datetime"]).replace(tzinfo=UTC) if "datetime" in m else None
        )
        if since is None or played is None or played >= since:
            out.append(int(m["id"]))
    return sorted(out)


def match_specs(
    lake: Lake, base: str = DEFAULT_BASE, *, since: datetime | None = None
) -> list[FileSpec]:
    """One spec per finished match in the latest league payloads (skip handled later).

    ``since`` limits it to matches played after that time (the daily incremental run,
    whose ephemeral lake can't see what earlier runs stored)."""
    ids: set[int] = set()
    for key in latest_by_params(lake, SOURCE, "league").values():
        _, payload = read_bronze(lake, key)
        ids.update(finished_match_ids(json.loads(payload), since))
    return [
        FileSpec(
            "match",
            f"{base.rstrip('/')}/getMatchData/{mid}",
            {"match": str(mid)},
            headers=HEADERS,
        )
        for mid in sorted(ids)
    ]
