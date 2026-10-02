"""Official BPS weights (rules YAML ``bps``) against weights estimated from the data.

Least squares of official ``bps`` on the counts of the events the simulator generates,
controlling for Understat key passes and shots. The estimates are *effective* weights:
a goal usually also brings a shot on target, a penalty miss a missed big chance, so they
differ from the official table by the BPS of correlated actions the table scores
separately. M10 (Phase 3) reconstructs BPS with the effective weights; this report shows
both side by side and the minutes boundary (the 60+ award applies from exactly 60).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from fplh.rules.config import Rules

COMPONENTS = [
    "minutes_lt_60",
    "minutes_gte_60",
    "goal_GK",
    "goal_DEF",
    "goal_MID",
    "goal_FWD",
    "assist",
    "clean_sheet",
    "goal_conceded",
    "save",
    "penalty_save",
    "penalty_miss",
    "yellow_card",
    "red_card",
    "own_goal",
]
CONTROLS = ["key_passes", "shots"]


def design(pm: pd.DataFrame) -> pd.DataFrame:
    """Event-count design matrix for player-fixtures with minutes > 0. ``pm`` has the
    ``fact_player_match`` columns plus (optionally) Understat ``key_passes``/``shots``."""
    d = pm[pm["minutes"] > 0]
    gk_def = d["position"].isin(["GK", "DEF"]).astype(int)
    x = pd.DataFrame(
        {
            "minutes_lt_60": (d["minutes"] < 60).astype(int),
            "minutes_gte_60": (d["minutes"] >= 60).astype(int),
            **{
                f"goal_{p}": d["goals_scored"] * (d["position"] == p)
                for p in ("GK", "DEF", "MID", "FWD")
            },
            "assist": d["assists"],
            "clean_sheet": d["clean_sheets"] * gk_def,
            "goal_conceded": d["goals_conceded"] * gk_def,
            "save": d["saves"],
            "penalty_save": d["penalties_saved"],
            "penalty_miss": d["penalties_missed"],
            "yellow_card": d["yellow_cards"],
            "red_card": d["red_cards"],
            "own_goal": d["own_goals"],
        },
        index=d.index,
    )
    for c in CONTROLS:
        x[c] = d[c].fillna(0) if c in d else 0
    return x.astype(float)


def official_weights(rules: Rules) -> dict[str, float]:
    b = rules.bps
    return {
        "minutes_lt_60": b.minutes_lt_60,
        "minutes_gte_60": b.minutes_gte_60,
        **{f"goal_{p}": v for p, v in b.goal.items()},
        "assist": b.assist,
        "clean_sheet": b.clean_sheet["DEF"],
        "goal_conceded": b.goal_conceded["DEF"],
        "save": b.save,
        "penalty_save": b.penalty_save,
        "penalty_miss": b.penalty_miss,
        "yellow_card": b.yellow_card,
        "red_card": b.red_card,
        "own_goal": b.own_goal,
    }


def estimate_bps_weights(pm: pd.DataFrame, rules: Rules) -> pd.DataFrame:
    """component, official, estimated (+ R² and n as frame attrs)."""
    x = design(pm)
    y = pm.loc[x.index, "bps"].astype(float).to_numpy()
    beta, *_ = np.linalg.lstsq(x.to_numpy(), y, rcond=None)
    resid = y - x.to_numpy() @ beta
    est = dict(zip(x.columns, beta, strict=True))
    off = official_weights(rules)
    out = pd.DataFrame(
        {
            "component": COMPONENTS + CONTROLS,
            "official": [off.get(c, np.nan) for c in COMPONENTS + CONTROLS],
            "estimated": [round(float(est[c]), 2) for c in COMPONENTS + CONTROLS],
        }
    )
    out.attrs["r2"] = float(1 - resid.var() / y.var()) if len(y) else float("nan")
    out.attrs["n"] = len(y)
    return out
