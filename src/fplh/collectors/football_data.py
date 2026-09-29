"""football-data.co.uk results, match stats and odds (EPL = E0, Championship = E1).

Column meanings: https://www.football-data.co.uk/notes.txt. Pre-match odds are collected
on Friday afternoons for weekend games and Tuesday afternoons for midweek games; closing
odds (``…C…`` columns) are at kickoff (used to set ``observed_at`` in silver).
"""

from __future__ import annotations

from fplh.collectors.backfill import FileSpec

SOURCE = "football_data"
DEFAULT_BASE = "https://www.football-data.co.uk/mmz4281"
DIVISIONS = ("E0", "E1")
FIRST_SEASON = 1993

# Closing-odds column families checked by the availability report (spec gap 3).
CLOSING_FAMILIES: dict[str, tuple[str, str, str]] = {
    "pinnacle_closing": ("PSCH", "PSCD", "PSCA"),
    "market_avg_closing": ("AvgCH", "AvgCD", "AvgCA"),
    "market_max_closing": ("MaxCH", "MaxCD", "MaxCA"),
    "bet365_closing": ("B365CH", "B365CD", "B365CA"),
    "pinnacle_prematch": ("PSH", "PSD", "PSA"),
    "market_avg_prematch": ("AvgH", "AvgD", "AvgA"),
    "bet365_prematch": ("B365H", "B365D", "B365A"),
}


def season_code(start_year: int) -> str:
    """1993 → ``9394``; 2026 → ``2627``."""
    return f"{start_year % 100:02d}{(start_year + 1) % 100:02d}"


def football_data_specs(
    seasons: list[int],
    current_season: int,
    divisions: tuple[str, ...] = DIVISIONS,
    base: str = DEFAULT_BASE,
) -> list[FileSpec]:
    base = base.rstrip("/")
    return [
        FileSpec(
            division,
            f"{base}/{season_code(year)}/{division}.csv",
            {"season": season_code(year)},
            refresh=year == current_season,
            optional=year == current_season,  # the new season's file may not exist yet
        )
        for year in seasons
        for division in divisions
    ]
