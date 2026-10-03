"""Features of the OpenFPL method (Groos 2025, arXiv:2508.09992), re-implemented on our
silver tables: trailing means over the last 1, 3, 5, 10 and 38 matches of

* the player's FPL record (points, minutes, ICT, event counts, BPS, price);
* "relevant" points: points in matches the player actually played;
* the player's Understat record (shots, xG, xA, key passes, xGChain, xGBuildup);
* the player's team and the opponent (Understat team xG, xGA, goals, deep, …);

plus position and venue. Each row is computed at its own ``deadline_at`` from rows
observed by then (``trailing_means``), so the same builder serves prediction at D and
training sets spanning many past deadlines.
"""

from __future__ import annotations

import pandas as pd

from fplh.features.information_set import InformationSet
from fplh.features.trailing import trailing_means

WINDOWS = (1, 3, 5, 10, 38)
FPL_VALUES = (
    "total_points",
    "minutes",
    "influence",
    "creativity",
    "threat",
    "goals_scored",
    "assists",
    "clean_sheets",
    "goals_conceded",
    "own_goals",
    "penalties_saved",
    "penalties_missed",
    "yellow_cards",
    "red_cards",
    "saves",
    "bonus",
    "bps",
)
US_VALUES = ("shots", "xg", "xa", "key_passes", "xg_chain", "xg_buildup")
TEAM_VALUES = ("xg", "xga", "npxg", "npxga", "scored", "missed", "deep", "deep_allowed")
POSITIONS = ("GK", "DEF", "MID", "FWD")
ROW_COLUMNS = ("player_uid", "team", "opponent", "was_home", "position", "deadline_at")


def _with(df: pd.DataFrame, cols: tuple[str, ...]) -> pd.DataFrame:
    missing = [c for c in cols if c not in df]
    return df.assign(**dict.fromkeys(missing, float("nan"))) if missing else df


def openfpl_features(info: InformationSet, rows: pd.DataFrame) -> pd.DataFrame:
    """Feature frame aligned with ``rows`` (columns ROW_COLUMNS, any index)."""
    pm = _with(info.table("fact_player_match"), (*FPL_VALUES, "value"))
    us = _with(info.table("fact_player_match_understat"), US_VALUES)
    if "player_uid" in us:
        us = us[us["player_uid"].notna()]
    tm = _with(info.table("us_team_match"), TEAM_VALUES)
    parts = [
        trailing_means(pm, "player_uid", FPL_VALUES, WINDOWS, rows, prefix="p_"),
        trailing_means(pm, "player_uid", ("value",), (1,), rows, prefix="price_"),
        trailing_means(
            pm[pm["minutes"] > 0], "player_uid", ("total_points",), WINDOWS, rows, prefix="app_"
        ),
        trailing_means(us, "player_uid", US_VALUES, WINDOWS, rows, prefix="u_"),
        trailing_means(tm, "team", TEAM_VALUES, WINDOWS, rows, prefix="t_"),
        trailing_means(
            tm, "team", TEAM_VALUES, WINDOWS, rows.assign(team=rows["opponent"]), prefix="o_"
        ),
    ]
    context = pd.DataFrame(
        {
            "home": rows["was_home"].astype(float),
            **{f"pos_{p}": (rows["position"] == p).astype(float) for p in POSITIONS},
        },
        index=rows.index,
    )
    return pd.concat([*parts, context], axis=1).drop(columns=["price_n"])
