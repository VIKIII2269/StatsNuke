"""Baseline feature builders. Each is ``(InformationSet, spine) -> frame keyed by the
spine``; they see only what the information set serves (ARCHITECTURE.md P4).
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd

from fplh.features.information_set import InformationSet
from fplh.features.spine import SPINE_KEYS, asof_join

FeatureBuilder = Callable[[InformationSet, pd.DataFrame], pd.DataFrame]

EVENT_TOTALS = (
    "minutes",
    "goals_scored",
    "assists",
    "saves",
    "bonus",
    "yellow_cards",
    "red_cards",
    "own_goals",
    "penalties_missed",
    "penalties_saved",
    "goals_conceded",
)


def naive_minutes(info: InformationSet, spine: pd.DataFrame, n: int = 3) -> pd.DataFrame:
    """Mean minutes and shares of 60+ / 1–59 over the player's last ``n`` registered
    fixtures (this season). No appearances yet → NULL, never imputed."""
    pm = info.table("fact_player_match")
    out = spine[SPINE_KEYS].copy()
    if pm.empty:
        return out.assign(mins_mean=np.nan, p60=np.nan, p1_59=np.nan, n_recent=0)
    season = spine["fixture_uid"].str.split(":").str[0].iloc[0] if len(spine) else None
    recent = (
        pm[pm["season"] == season]
        .sort_values("kickoff_at")
        .groupby("player_uid")
        .tail(n)
        .assign(ge60=lambda d: d["minutes"] >= 60, sub=lambda d: d["minutes"].between(1, 59))
        .groupby("player_uid")
        .agg(
            mins_mean=("minutes", "mean"),
            p60=("ge60", "mean"),
            p1_59=("sub", "mean"),
            n_recent=("minutes", "size"),
        )
    )
    out = out.merge(recent, left_on="player_uid", right_index=True, how="left")
    out["n_recent"] = out["n_recent"].fillna(0).astype("int64")
    return out


def season_totals(info: InformationSet, spine: pd.DataFrame) -> pd.DataFrame:
    """Season-to-date event totals, plus defensive-contribution threshold hits."""
    pm = info.table("fact_player_match")
    out = spine[SPINE_KEYS].copy()
    empty_cols = {f"s_{c}": np.nan for c in (*EVENT_TOTALS, "dc_hits", "dc_known", "apps")}
    season = spine["fixture_uid"].str.split(":").str[0].iloc[0] if len(spine) else None
    if pm.empty or season not in set(pm["season"]):
        return out.assign(**empty_cols)  # no history yet this season: NULL, not 0
    pm = pm[pm["season"] == season].copy()
    cbit = pm["clearances_blocks_interceptions"].astype("float") + pm["tackles"].astype("float")
    thresh = np.where(
        pm["position"] == "DEF", cbit >= 10, (cbit + pm["recoveries"].astype("float")) >= 12
    )
    pm["dc_hits"] = np.where(cbit.isna(), np.nan, thresh.astype(float))
    pm["team_cs"] = ((pm["goals_conceded"] == 0) & (pm["minutes"] >= 60)).astype(float)
    totals = pm.groupby("player_uid").agg(
        **{c: (c, "sum") for c in EVENT_TOTALS},
        dc_hits=("dc_hits", "sum"),
        dc_known=("dc_hits", "count"),
        apps=("minutes", lambda m: int((m > 0).sum())),
    )
    totals["dc_hits"] = totals["dc_hits"].where(totals["dc_known"] > 0)  # unknown ≠ zero
    return out.merge(totals.add_prefix("s_"), left_on="player_uid", right_index=True, how="left")


def snapshot_status(info: InformationSet, spine: pd.DataFrame) -> pd.DataFrame:
    """Latest captured FPL status at the deadline; NULL before capture began (gap 1)."""
    snap = info.table("snap_fpl_player")
    cols = ["status", "chance_of_playing_next_round", "now_cost", "selected_by_percent", "ep_next"]
    if snap.empty:
        return spine[SPINE_KEYS].assign(**{c: np.nan for c in cols})
    snap = snap.assign(player_uid="fpl:" + snap["code"].astype(str))
    return asof_join(spine[SPINE_KEYS], snap, ["player_uid"], cols)


BUILDERS: dict[str, FeatureBuilder] = {
    "naive_minutes": naive_minutes,
    "season_totals": season_totals,
    "snapshot_status": snapshot_status,
}


def build_features(info: InformationSet, spine: pd.DataFrame) -> pd.DataFrame:
    out = spine.copy()
    for builder in BUILDERS.values():
        out = out.merge(builder(info, spine), on=SPINE_KEYS, how="left")
    info.assert_no_leakage()
    return out
