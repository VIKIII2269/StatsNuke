"""Official points reproduced exactly (ARCHITECTURE.md §9).

The offline samples in ``data/`` always run. The full-season checks run when the vaastav
CSVs are cached (``uv run fplh golden fetch --season 2025/26 --season 2026/27``) and are
skipped otherwise; CI fetches them.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from fplh.cli import KNOWN_EXCEPTIONS
from fplh.rules import load_rules
from fplh.rules.golden import check, golden_cache_path, load_known_exceptions, load_vaastav
from fplh.settings import get_settings

DATA = Path(__file__).parent / "data"

SAMPLES = [
    ("2025/26", DATA / "2025_26_sample.csv"),
    ("2026/27", DATA / "2026_27_gw1.csv"),
]


def _assert_exact(season: str, path: Path) -> None:
    report = check(
        load_vaastav(season, path),
        load_rules(season),
        load_known_exceptions(KNOWN_EXCEPTIONS, season),
    )
    assert report.ok, report.summary() + "\n" + report.points_mismatches.head(20).to_string()


@pytest.mark.parametrize(("season", "path"), SAMPLES, ids=[s for s, _ in SAMPLES])
def test_offline_sample_reproduces_official_points(season: str, path: Path) -> None:
    _assert_exact(season, path)


def test_sample_covers_edge_cases() -> None:
    df = load_vaastav("2025/26", DATA / "2025_26_sample.csv").frame
    assert (df["penalties_saved"] > 0).any()
    assert (df["penalties_missed"] > 0).any()
    assert (df["own_goals"] > 0).any()
    assert (df["red_cards"] > 0).any()
    assert ((df["position"] == "MID") & (df["clean_sheets"] == 1)).any()
    assert ((df["minutes"] > 0) & (df["minutes"] < 60)).any()
    assert (df.groupby(["element", "round"])["fixture"].nunique() > 1).any(), "no DGW player"
    assert (df["defensive_contribution"] >= 12).any()


def test_duplicate_artefact_is_dropped() -> None:
    golden = load_vaastav("2025/26", DATA / "2025_26_sample.csv")
    assert golden.duplicates_dropped > 0
    assert not golden.frame.duplicated(["element", "fixture"]).any()


@pytest.mark.golden
@pytest.mark.parametrize("season", ["2025/26", "2026/27"])
def test_full_season(season: str) -> None:
    path = golden_cache_path(get_settings().cache_dir, season)
    if not path.exists():
        pytest.skip(f"no cached data; run `fplh golden fetch --season {season}`")
    _assert_exact(season, path)
