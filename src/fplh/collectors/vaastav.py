"""vaastav/Fantasy-Premier-League history (2016/17 onward) into bronze.

Per season: ``gws/merged_gw.csv`` (per player-fixture stats), ``players_raw.csv``
(positions, teams, stable ``code``), and ``fixtures.csv`` / ``teams.csv`` where the
season has them; plus the cross-season ``master_team_list.csv``. The ``xP`` column may
contain post-match information and is dropped in silver (ARCHITECTURE.md §5.3.1).
"""

from __future__ import annotations

from fplh.collectors.backfill import FileSpec, season_label

SOURCE = "vaastav"
FIRST_SEASON = 2016
DEFAULT_BASE = "https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master/data"

# endpoint name → (path inside the season folder, optional?)
SEASON_FILES: dict[str, tuple[str, bool]] = {
    "merged_gw": ("gws/merged_gw.csv", False),
    "players_raw": ("players_raw.csv", False),
    "fixtures": ("fixtures.csv", True),
    "teams": ("teams.csv", True),
}


def vaastav_specs(
    seasons: list[int], current_season: int, base: str = DEFAULT_BASE
) -> list[FileSpec]:
    base = base.rstrip("/")
    specs = [
        FileSpec(
            "master_team_list",
            f"{base}/master_team_list.csv",
            {"as_of_season": season_label(current_season, "_")},
        )
    ]
    for year in seasons:
        label = season_label(year)
        for endpoint, (path, optional) in SEASON_FILES.items():
            specs.append(
                FileSpec(
                    endpoint,
                    f"{base}/{label}/{path}",
                    {"season": season_label(year, "_")},
                    refresh=year == current_season,
                    optional=optional,
                )
            )
    return specs
