"""Understat league and match data (EPL, 2014/15 onward).

Understat serves JSON to its own pages from two endpoints (both need the
``X-Requested-With: XMLHttpRequest`` header):

* ``/getLeagueData/EPL/{season_start_year}`` → ``datesData`` (fixtures, results, xG),
  ``teamsData`` (per-match team history incl. PPDA) and ``playersData`` (season totals);
* ``/getMatchData/{match_id}`` → ``match_info``, ``rostersData`` and ``shotsData``.

Stage 1 refreshes league data; stage 2 fetches every finished match not yet in bronze.
"""

from __future__ import annotations

import json
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


def finished_match_ids(league_payload: dict[str, Any]) -> list[int]:
    return sorted(int(m["id"]) for m in league_payload["datesData"] if m.get("isResult"))


def match_specs(lake: Lake, base: str = DEFAULT_BASE) -> list[FileSpec]:
    """One spec per finished match in the latest league payloads (skip handled later)."""
    ids: set[int] = set()
    for key in latest_by_params(lake, SOURCE, "league").values():
        _, payload = read_bronze(lake, key)
        ids.update(finished_match_ids(json.loads(payload)))
    return [
        FileSpec(
            "match",
            f"{base.rstrip('/')}/getMatchData/{mid}",
            {"match": str(mid)},
            headers=HEADERS,
        )
        for mid in sorted(ids)
    ]
