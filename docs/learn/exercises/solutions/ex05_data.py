"""Reference solutions for exercise 05."""

from __future__ import annotations

import pandas as pd

from fplh.features.information_set import InformationSet
from fplh.features.spine import SPINE_KEYS


def observation_time(event_at: pd.Series, lag_hours: float) -> pd.Series:
    return event_at + pd.Timedelta(hours=lag_hours)


def visible(df: pd.DataFrame, deadline: pd.Timestamp) -> pd.DataFrame:
    return df[df["observed_at"] <= deadline]


def goals_so_far(info: InformationSet, spine: pd.DataFrame) -> pd.DataFrame:
    pm = info.table("fact_player_match")
    out = spine[SPINE_KEYS].copy()
    if pm.empty:
        return out.assign(goals_so_far=0)
    goals = pm.groupby("player_uid")["goals_scored"].sum().rename("goals_so_far")
    out = out.merge(goals, left_on="player_uid", right_index=True, how="left")
    out["goals_so_far"] = out["goals_so_far"].fillna(0)
    return out


def latest_price(spine: pd.DataFrame, snaps: pd.DataFrame) -> pd.Series:
    left = spine.reset_index(drop=True).assign(_row=lambda d: range(len(d)))
    merged = pd.merge_asof(
        left.sort_values("deadline_at"),
        snaps.sort_values("observed_at"),
        left_on="deadline_at",
        right_on="observed_at",
        by="player_uid",
        direction="backward",
    )
    return merged.sort_values("_row")["price"].reset_index(drop=True)


def link_decision(score: float, overlap: float, shared: int) -> str:
    if score >= 92 and overlap >= 0.8:
        return "auto"
    if 80 <= score < 92 and overlap >= 0.95 and shared >= 3:
        return "auto"
    if score >= 80:
        return "review"
    return "none"
