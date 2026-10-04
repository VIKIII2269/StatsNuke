"""Minutes-model features (M4, ARCHITECTURE.md §7.6), point-in-time per row.

Each row (player, fixture, ``deadline_at``) gets, from what was observable at its own
deadline:

* trailing shares over the player's last 1/3/5/10 registered fixtures: started (from the
  Understat lineups), appeared, played 60+, and mean minutes;
* depth: the summed recent start share of team-mates at the same position registered
  for the fixture, and the player's rank among them;
* rest days before the fixture and days since the player's last appearance;
* position, price and the latest snapshot ``chance_of_playing_next_round`` (missing,
  never imputed, before our own captures began: gap 1).

``started`` labels come from Understat rosters (2014/15+, every FPL appearance is
linked); FPL's own ``starts`` exists only from 2022/23 and agrees wherever it is set.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from fplh.features.information_set import InformationSet
from fplh.features.spine import asof_join
from fplh.features.trailing import trailing_means

WINDOWS = (1, 3, 5, 10)
POSITIONS = ("GK", "DEF", "MID", "FWD")


def player_history(info: InformationSet) -> pd.DataFrame:
    """FPL player-fixtures observable at D with the Understat start flag."""
    pm = info.table("fact_player_match")
    us = info.table("fact_player_match_understat")
    cols = ["player_uid", "fixture_uid", "started"]
    if not us.empty and "started" in us:
        us = us[us["player_uid"].notna()][cols].drop_duplicates(["player_uid", "fixture_uid"])
        pm = pm.merge(us, on=["player_uid", "fixture_uid"], how="left")
    else:
        pm = pm.assign(started=np.nan)
    pm["started"] = pm["started"].astype("boolean").fillna(False).astype(float)
    if "value" not in pm:
        pm["value"] = np.nan
    pm["appeared"] = (pm["minutes"] > 0).astype(float)
    pm["full"] = (pm["minutes"] >= 60).astype(float)
    pm["full_if_started"] = pm["full"].where(pm["started"] > 0)
    return pm


def _rest_days(info: InformationSet, rows: pd.DataFrame) -> pd.Series:
    """Days between the team's previous scheduled fixture and this one (schedule is public)."""
    dim = info.table("dim_fixture")
    sides = pd.concat(
        [
            dim.assign(team=dim["home_team"])[["team", "fixture_uid", "kickoff_at"]],
            dim.assign(team=dim["away_team"])[["team", "fixture_uid", "kickoff_at"]],
        ]
    ).sort_values(["team", "kickoff_at"])
    sides["prev"] = sides.groupby("team")["kickoff_at"].shift()
    gap = (sides["kickoff_at"] - sides["prev"]).dt.total_seconds() / 86400
    lookup = pd.Series(
        gap.to_numpy(), index=pd.MultiIndex.from_frame(sides[["team", "fixture_uid"]])
    )
    keys = pd.MultiIndex.from_frame(rows[["team", "fixture_uid"]])
    out = pd.Series(lookup.reindex(keys).to_numpy(), index=rows.index).clip(upper=30)
    return out


def minutes_features(info: InformationSet, rows: pd.DataFrame) -> pd.DataFrame:
    """``rows``: player_uid, fixture_uid, team, position, kickoff_at, deadline_at."""
    hist = player_history(info)
    parts = [
        trailing_means(
            hist,
            "player_uid",
            ("started", "appeared", "full", "minutes"),
            WINDOWS,
            rows,
            prefix="h_",
        ),
        trailing_means(hist, "player_uid", ("full_if_started",), (5, 10), rows, prefix="hs_"),
        trailing_means(hist, "player_uid", ("value",), (1,), rows, prefix="price_"),
    ]
    out = pd.concat(parts, axis=1).drop(columns=["hs_n", "price_n"])
    # depth at position among the players registered for the same fixture and team
    rate = out["h_started_5"].fillna(0.0)
    grp = [rows["fixture_uid"], rows["team"], rows["position"]]
    total = rate.groupby(grp).transform("sum")
    out["depth_others"] = total - rate
    out["depth_rank"] = rate.groupby(grp).rank(ascending=False, method="min")
    out["rest_days"] = _rest_days(info, rows)
    last = trailing_means(
        hist[hist["minutes"] > 0].assign(t=lambda d: d["kickoff_at"].astype("int64") / 8.64e13),
        "player_uid",
        ("t",),
        (1,),
        rows,
    )["t_1"]
    out["days_since_appearance"] = (rows["kickoff_at"].astype("int64") / 8.64e13 - last).clip(
        upper=365
    )
    snap = info.table("snap_fpl_player")
    if not snap.empty and "chance_of_playing_next_round" in snap:
        s = snap.assign(player_uid="fpl:" + snap["code"].astype(str))
        j = asof_join(
            rows[["player_uid", "deadline_at"]].reset_index(drop=True),
            s,
            ["player_uid"],
            ["chance_of_playing_next_round"],
        )
        out["chance_of_playing"] = j["chance_of_playing_next_round"].to_numpy(dtype=float)
    else:
        out["chance_of_playing"] = np.nan
    for p in POSITIONS:
        out[f"pos_{p}"] = (rows["position"] == p).astype(float)
    return out
