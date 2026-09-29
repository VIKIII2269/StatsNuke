"""Typed, strict model of a season's scoring-rules YAML (ARCHITECTURE.md §5.4)."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from fplh.settings import get_settings

POSITIONS: tuple[str, ...] = ("GK", "DEF", "MID", "FWD")
Position = Literal["GK", "DEF", "MID", "FWD"]
Action = Literal["clearance", "block", "interception", "tackle", "recovery"]
Chip = Literal["wildcard", "free_hit", "triple_captain", "bench_boost"]

PLACEHOLDER = "VERIFY"


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class Appearance(_Strict):
    lt_60: int
    gte_60: int


class PerN(_Strict):
    per: int = Field(gt=0)
    points: int


class CleanSheet(_Strict):
    GK: int
    DEF: int
    MID: int
    FWD: int
    min_minutes: int = Field(ge=0)


class DefensiveGroup(_Strict):
    actions: list[Action]
    threshold: int = Field(gt=0)
    points: int

    @model_validator(mode="after")
    def _cbi_together(self) -> DefensiveGroup:
        # FPL publishes clearances, blocks and interceptions as one combined count.
        cbi = {"clearance", "block", "interception"}
        if 0 < len(cbi & set(self.actions)) < 3:
            raise ValueError("clearance, block and interception must be listed together")
        return self


class DefensiveContribution(_Strict):
    DEF: DefensiveGroup
    MID_FWD: DefensiveGroup
    cap_per_match: int = Field(ge=0)


class Bonus(_Strict):
    ranks: list[int]
    tie_rule: Literal["official"]


class Chips(_Strict):
    sets: int = Field(ge=1)
    per_set: list[Chip]
    first_set_deadline_gw: int


class Game(_Strict):
    squad: dict[Position, int]
    xi_min: dict[Position, int]
    max_per_club: int
    free_transfer_bank_max: int
    hit_cost: int
    free_transfers_preserved_on: list[Chip]
    chips: Chips
    lockdown: str


class Rules(_Strict):
    season: str
    appearance: Appearance
    goal: dict[Position, int]
    assist: int
    clean_sheet: CleanSheet
    goals_conceded: dict[Position, PerN]
    saves: PerN
    penalty_save: int
    penalty_miss: int
    yellow_card: int
    red_card: int
    own_goal: int
    defensive_contribution: DefensiveContribution
    bonus: Bonus
    bps: dict[str, Any]
    game: Game

    @model_validator(mode="before")
    @classmethod
    def _no_placeholders(cls, data: Any) -> Any:
        found = list(_find_placeholders(data, ""))
        if found:
            raise ValueError(
                f"unverified rule values ({PLACEHOLDER}) at: {', '.join(found)}; "
                "confirm against the official rules page and the golden tests"
            )
        return data

    @model_validator(mode="after")
    def _complete(self) -> Rules:
        missing = set(POSITIONS) - set(self.goal)
        if missing:
            raise ValueError(f"goal points missing for {sorted(missing)}")
        return self


def _find_placeholders(node: Any, path: str) -> Any:
    if isinstance(node, dict):
        for k, v in node.items():
            yield from _find_placeholders(v, f"{path}.{k}" if path else str(k))
    elif isinstance(node, list):
        for i, v in enumerate(node):
            yield from _find_placeholders(v, f"{path}[{i}]")
    elif isinstance(node, str) and node.strip().upper() == PLACEHOLDER:
        yield path


_SEASON = re.compile(r"^(\d{4})[/\-_](\d{2})$")


def season_slug(season: str) -> str:
    """``"2026/27"`` / ``"2026-27"`` → ``"2026_27"``."""
    m = _SEASON.match(season.strip())
    if not m:
        raise ValueError(f"season must look like 2026/27, got {season!r}")
    return f"{m.group(1)}_{m.group(2)}"


def rules_path(season: str, configs_dir: Path | None = None) -> Path:
    base = configs_dir or get_settings().configs_dir
    return base / "rules" / f"fpl_{season_slug(season)}.yaml"


def parse_rules(data: Any) -> Rules:
    return Rules.model_validate(data)


def load_rules(season: str, configs_dir: Path | None = None) -> Rules:
    path = rules_path(season, configs_dir)
    with path.open() as fh:
        rules = parse_rules(yaml.safe_load(fh))
    if season_slug(rules.season) != season_slug(season):
        raise ValueError(f"{path} declares season {rules.season}, expected {season}")
    return rules
