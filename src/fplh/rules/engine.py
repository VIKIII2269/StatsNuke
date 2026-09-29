"""FPL points from match events: a pure, vectorised function of (events, rules).

Input columns use the official FPL API names (``element-summary`` history), one row per
player × fixture. Output has one column per scoring component plus ``total`` so errors
can be attributed term by term. The same core (:func:`score_arrays`) is used by the match
simulator on arrays of any shape (e.g. simulations × players). ARCHITECTURE.md §9.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd

from fplh.rules.config import POSITIONS, DefensiveGroup, Rules

IntArray = npt.NDArray[np.int64]

EVENT_COLUMNS: tuple[str, ...] = (
    "minutes",
    "goals_scored",
    "assists",
    "goals_conceded",  # conceded while the player was on the pitch
    "own_goals",
    "penalties_saved",
    "penalties_missed",
    "yellow_cards",
    "red_cards",
    "saves",
    "bonus",
)
DEFENSIVE_COLUMNS: tuple[str, ...] = ("clearances_blocks_interceptions", "tackles", "recoveries")

COMPONENTS: tuple[str, ...] = (
    "appearance",
    "goals",
    "assists",
    "clean_sheet",
    "goals_conceded",
    "saves",
    "penalty_saves",
    "penalty_misses",
    "yellow_cards",
    "red_cards",
    "own_goals",
    "defensive_contribution",
    "bonus",
)

_ACTION_COLUMN = {
    "clearance": "clearances_blocks_interceptions",
    "block": "clearances_blocks_interceptions",
    "interception": "clearances_blocks_interceptions",
    "tackle": "tackles",
    "recovery": "recoveries",
}


def _by_position(position: npt.NDArray[np.str_], values: Mapping[Any, int]) -> IntArray:
    out = np.zeros(position.shape, dtype=np.int64)
    for pos, v in values.items():
        out[position == pos] = v
    return out


def defensive_count(
    group: DefensiveGroup, events: Mapping[str, npt.ArrayLike] | pd.DataFrame
) -> IntArray:
    cols = dict.fromkeys(_ACTION_COLUMN[a] for a in group.actions)  # ordered, de-duplicated
    total: IntArray | None = None
    for col in cols:
        v = np.asarray(events[col], dtype=np.int64)
        total = v if total is None else total + v
    assert total is not None
    return total


def score_arrays(
    events: Mapping[str, npt.ArrayLike], position: npt.ArrayLike, rules: Rules
) -> dict[str, IntArray]:
    """Points per scoring component for broadcast-compatible event arrays."""
    pos = np.asarray(position).astype(str)
    unknown = set(np.unique(pos)) - set(POSITIONS)
    if unknown:
        raise ValueError(f"unknown positions: {sorted(unknown)}")
    missing = [c for c in EVENT_COLUMNS if c not in events]
    if missing:
        raise KeyError(f"missing event columns: {missing}")

    def col(name: str) -> IntArray:
        return np.broadcast_to(np.asarray(events[name], dtype=np.int64), pos.shape)

    minutes = col("minutes")
    played = minutes > 0
    conceded = col("goals_conceded")

    cs = rules.clean_sheet
    cs_points = {"GK": cs.GK, "DEF": cs.DEF, "MID": cs.MID, "FWD": cs.FWD}
    kept_clean = (minutes >= cs.min_minutes) & (conceded == 0) & played

    gc = np.zeros(pos.shape, dtype=np.int64)
    for p, rule in rules.goals_conceded.items():
        mask = pos == p
        gc[mask] = (conceded[mask] // rule.per) * rule.points

    dc_rules = rules.defensive_contribution
    dc = np.zeros(pos.shape, dtype=np.int64)
    if dc_rules.DEF.points or dc_rules.MID_FWD.points:
        missing_dc = [c for c in DEFENSIVE_COLUMNS if c not in events]
        if missing_dc:
            raise KeyError(f"missing defensive-contribution columns: {missing_dc}")
        for group, mask in (
            (dc_rules.DEF, pos == "DEF"),
            (dc_rules.MID_FWD, (pos == "MID") | (pos == "FWD")),
        ):
            count = np.broadcast_to(defensive_count(group, events), pos.shape)
            hit = mask & (count >= group.threshold)
            dc[hit] = min(group.points, dc_rules.cap_per_match)

    saves = rules.saves
    return {
        "appearance": np.where(
            minutes >= 60,
            rules.appearance.gte_60,
            np.where(played, rules.appearance.lt_60, 0),
        ).astype(np.int64),
        "goals": col("goals_scored") * _by_position(pos, rules.goal),
        "assists": col("assists") * rules.assist,
        "clean_sheet": np.where(kept_clean, _by_position(pos, cs_points), 0).astype(np.int64),
        "goals_conceded": gc,
        "saves": (col("saves") // saves.per) * saves.points,
        "penalty_saves": col("penalties_saved") * rules.penalty_save,
        "penalty_misses": col("penalties_missed") * rules.penalty_miss,
        "yellow_cards": col("yellow_cards") * rules.yellow_card,
        "red_cards": col("red_cards") * rules.red_card,
        "own_goals": col("own_goals") * rules.own_goal,
        "defensive_contribution": dc,
        "bonus": col("bonus"),
    }


def score(events: pd.DataFrame, rules: Rules) -> pd.DataFrame:
    """Points breakdown for a player × fixture frame (needs a ``position`` column)."""
    arrays = {c: events[c].to_numpy() for c in events.columns if c != "position"}
    parts = score_arrays(arrays, events["position"].to_numpy(), rules)
    out = pd.DataFrame({c: parts[c] for c in COMPONENTS}, index=events.index)
    out["total"] = out[list(COMPONENTS)].sum(axis=1)
    return out
