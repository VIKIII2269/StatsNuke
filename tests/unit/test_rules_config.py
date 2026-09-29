from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from fplh.rules.config import load_rules, parse_rules, rules_path, season_slug
from fplh.settings import get_settings


def _raw(season: str = "2026/27") -> dict:
    return yaml.safe_load(rules_path(season).read_text())


@pytest.mark.parametrize("season", ["2025/26", "2026/27", "2026-27"])
def test_season_configs_load(season: str) -> None:
    rules = load_rules(season)
    assert season_slug(rules.season) == season_slug(season)
    assert set(rules.goal) == {"GK", "DEF", "MID", "FWD"}


def test_placeholder_rejected_with_path() -> None:
    raw = _raw()
    raw["goal"]["GK"] = "VERIFY"
    with pytest.raises(ValidationError, match=r"goal\.GK"):
        parse_rules(raw)


def test_unknown_key_rejected() -> None:
    raw = _raw()
    raw["assists"] = 3  # typo of "assist"
    with pytest.raises(ValidationError):
        parse_rules(raw)


def test_partial_cbi_rejected() -> None:
    raw = _raw()
    raw["defensive_contribution"]["DEF"]["actions"] = ["clearance", "tackle"]
    with pytest.raises(ValidationError, match="together"):
        parse_rules(raw)


def test_missing_position_rejected() -> None:
    raw = _raw()
    del raw["goal"]["FWD"]
    with pytest.raises(ValidationError, match="FWD"):
        parse_rules(raw)


def test_season_mismatch_rejected(tmp_path: Path) -> None:
    (tmp_path / "rules").mkdir()
    raw = _raw()
    raw["season"] = "2025/26"
    (tmp_path / "rules" / "fpl_2026_27.yaml").write_text(yaml.safe_dump(raw))
    with pytest.raises(ValueError, match="declares season"):
        load_rules("2026/27", tmp_path)


@pytest.mark.parametrize(("given", "slug"), [("2026/27", "2026_27"), ("2025-26", "2025_26")])
def test_season_slug(given: str, slug: str) -> None:
    assert season_slug(given) == slug


def test_bad_season_string() -> None:
    with pytest.raises(ValueError, match="season"):
        season_slug("26/27")


def test_configs_dir_setting_points_at_repo() -> None:
    assert (get_settings().configs_dir / "sources.yaml").exists()
