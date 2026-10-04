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


class BpsPerN(_Strict):
    per: int = Field(gt=0)
    bps: int


class Bps(_Strict):
    """Bonus Points System weights for the events the simulator generates (M10). The
    official table also scores actions it does not generate (passes, key passes, shots
    on target, big chances, recoveries, …); those stay in M10's residual."""

    minutes_lt_60: int
    minutes_gte_60: int
    goal: dict[Position, int]
    assist: int
    clean_sheet: dict[Literal["GK", "DEF"], int]  # 60+ minutes, like FPL clean sheets
    goal_conceded: dict[Literal["GK", "DEF"], int]  # per goal conceded while on the pitch
    save: int
    penalty_save: int
    penalty_miss: int
    yellow_card: int
    red_card: int
    own_goal: int
    # Season-specific detail the simulator does not generate (documented for M10).
    tackled_penalty: int | None = None
    cbi: BpsPerN | None = None
    gk_save_inside_box: int | None = None
    gk_save_other: int | None = None
    gk_big_chance_saved: int | None = None


class Chips(_Strict):
    """A season's chips, in one of two forms:

    * ``sets`` copies of the ``per_set`` chips; the first set is playable through gameweek
      ``first_set_deadline_gw`` (inclusive), the second after it (2025/26 on);
    * ``windows``: per chip, one inclusive gameweek range per copy (e.g. two wildcards, one
      per half, and one free hit for the season, as before 2025/26).
    """

    sets: int | None = Field(default=None, ge=1)
    per_set: list[Chip] | None = None
    first_set_deadline_gw: int | None = None
    windows: dict[Chip, list[tuple[int, int]]] | None = None

    @model_validator(mode="after")
    def _one_form(self) -> Chips:
        by_sets = self.sets is not None and self.per_set is not None
        if by_sets == (self.windows is not None):
            raise ValueError("give either sets/per_set/first_set_deadline_gw or windows")
        if self.sets is not None and self.sets > 1 and self.first_set_deadline_gw is None:
            raise ValueError("two or more sets need first_set_deadline_gw")
        for chip, ranges in (self.windows or {}).items():
            for lo, hi in ranges:
                if not 1 <= lo <= hi <= 38:
                    raise ValueError(f"{chip}: bad gameweek window ({lo}, {hi})")
        return self

    def allowance(self, last_gw: int = 38) -> dict[str, list[tuple[int, int]]]:
        """Per chip, the inclusive gameweek window of each copy."""
        if self.windows is not None:
            return {str(c): [(int(a), int(b)) for a, b in w] for c, w in self.windows.items()}
        assert self.sets is not None
        assert self.per_set is not None
        if self.sets == 1:
            spans = [(1, last_gw)]
        else:
            d = int(self.first_set_deadline_gw or last_gw // 2)
            spans = [(1, d), (d + 1, last_gw)]
        return {str(c): list(spans) for c in self.per_set}


class Game(_Strict):
    squad: dict[Position, int]
    xi_min: dict[Position, int]
    max_per_club: int
    free_transfer_bank_max: int
    hit_cost: int
    free_transfers_preserved_on: list[Chip]
    chips: Chips
    lockdown: str
    # gameweek → free transfers available that week regardless of the bank (top-ups such as
    # the 2022/23 World Cup break, unlimited ≈ 15, or 2025/26's AFCON top-up to 5)
    special_free_transfers: dict[int, int] = Field(default_factory=dict)


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
    bps: Bps
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
