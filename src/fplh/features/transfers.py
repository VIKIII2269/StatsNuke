"""FPL transfer activity per round, a crowd signal of team news (model v2).

FPL's per-round history (``fact_player_match``: ``transfers_in``, ``transfers_out``,
``selected``) counts the transfers made *before* that round's deadline: GW1 rows are 0,
and ownership is measured at the deadline. The counts are therefore public at the
round's deadline, though the row itself is only collected after the match. The derived
view ``fpl_round_transfers`` re-times them: one row per (season, round, player) with
``observed_at`` = the round's deadline (first kickoff − 90 min), so the information set
serves round g's transfers at D_g and never earlier.

Owners selling a player en masse is the market's reaction to injury and rotation news
(among players given P(start) ≈ 0.87 by M4, the most-sold 1 % started 35 % of the time).
The view is derived on read from silver tables already in the lake, so adding it
changes no silver file and no cached run's data hash.

Live season: the round being played next has no history row yet. Our bootstrap captures
(``snap_fpl_player``) carry the same counts for the open transfer window
(``transfers_in_event``, ``transfers_out_event``, ``selected_by_percent``,
``total_players``), observed at capture time, so at a deadline the latest capture stands
in until the round's history row (observed at that deadline) exists.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

COLUMNS = ["tr_sell", "tr_buy", "tr_own"]
# Columns of a fact table that are public before the row's own observation time: FPL's
# transfer counts and ownership are final at the round's deadline. The leakage check
# perturbs them by that availability time, not by the row's ``observed_at``.
PRE_DEADLINE: dict[str, tuple[str, ...]] = {
    "fact_player_match": ("transfers_in", "transfers_out", "selected")
}
DEADLINE_BEFORE_KICKOFF = pd.Timedelta(minutes=90)
MIN_OWNERS = 1000.0  # damps the ratios of barely owned players


OUT = ["season", "round", "player_uid", *COLUMNS, "event_at", "observed_at"]
SNAP_NEED = {"transfers_in_event", "transfers_out_event", "selected_by_percent", "total_players"}


def _ratios(
    t_in: pd.Series, t_out: pd.Series, owners: pd.Series, share: pd.Series
) -> dict[str, pd.Series]:
    owners_before = (owners - t_in + t_out).clip(lower=0)  # ownership at the last deadline
    # clipped: early seasons' ownership counts are not always consistent with the transfers
    return {
        "tr_sell": (t_out / (owners_before + MIN_OWNERS)).clip(0, 1),
        "tr_buy": (t_in / (owners_before + MIN_OWNERS)).pipe(np.log1p),
        "tr_own": share.clip(0, 1),
    }


def _deadlines(dim: pd.DataFrame) -> pd.Series:
    fx = dim[dim["round"].notna()]
    deadline = fx.groupby(["season", fx["round"].astype("int64")])["kickoff_at"].min()
    out: pd.Series = (deadline - DEADLINE_BEFORE_KICKOFF).rename("observed_at")
    return out


def available_at(pm: pd.DataFrame, dim: pd.DataFrame) -> pd.Series:
    """When each ``fact_player_match`` row's PRE_DEADLINE columns became public: its
    round's deadline (the row's ``observed_at`` if the round is unknown)."""
    if dim.empty or "fixture_uid" not in pm:
        out: pd.Series = pm["observed_at"]
        return out
    keys = pd.MultiIndex.from_arrays([pm["season"], _round_of(pm, dim)])
    when = _deadlines(dim).reindex(keys).to_numpy()
    out = pd.Series(when, index=pm.index).fillna(pm["observed_at"])
    return out


def _round_of(pm: pd.DataFrame, dim: pd.DataFrame) -> pd.Series:
    """Each row's round from the schedule (public in advance), not from the fact row."""
    rounds = dim.dropna(subset=["round"]).set_index("fixture_uid")["round"]
    out: pd.Series = pm["fixture_uid"].map(rounds).astype("Int64")
    return out


def _from_snapshots(snap: pd.DataFrame, deadlines: pd.Series) -> pd.DataFrame:
    """The open window's counts from each capture; round = the next deadline's."""
    s = snap.dropna(subset=list(SNAP_NEED)).copy()
    if s.empty:
        return pd.DataFrame(columns=OUT)
    s["player_uid"] = "fpl:" + s["code"].astype("int64").astype(str)
    s = s.sort_values("observed_at")
    d = deadlines.reset_index().rename(columns={"observed_at": "deadline"})
    d = d.sort_values("deadline")
    s = pd.merge_asof(
        s, d, left_on="observed_at", right_on="deadline", by="season", direction="forward"
    ).dropna(subset=["round"])
    managers = s["total_players"].astype(float)
    share = s["selected_by_percent"].astype(float) / 100
    s = s.assign(
        **_ratios(
            s["transfers_in_event"].astype(float),
            s["transfers_out_event"].astype(float),
            share * managers,
            share,
        )
    )
    s["round"] = s["round"].astype("int64")
    s["event_at"] = s["observed_at"]
    out: pd.DataFrame = s[OUT]
    return out


def fpl_round_transfers(
    pm: pd.DataFrame, dim: pd.DataFrame, snap: pd.DataFrame | None = None
) -> pd.DataFrame:
    """season, round, player_uid, the transfer ratios, observed_at (the round deadline,
    or the capture time for live snapshots)."""
    need = {"transfers_in", "transfers_out", "selected", "fixture_uid", "player_uid"}
    if dim.empty:
        return pd.DataFrame(columns=OUT)
    deadlines = _deadlines(dim)
    live = (
        _from_snapshots(snap, deadlines)
        if snap is not None and not snap.empty and set(snap.columns) >= SNAP_NEED
        else pd.DataFrame(columns=OUT)
    )
    if pm.empty or not need <= set(pm.columns):
        return live.reset_index(drop=True)
    r = pm.assign(round=_round_of(pm, dim)).dropna(subset=["round", "player_uid", "selected"])
    r = r.sort_values("kickoff_at").drop_duplicates(["season", "round", "player_uid"])
    r = r[["season", "round", "player_uid", "transfers_in", "transfers_out", "selected"]].copy()
    r["round"] = r["round"].astype("int64")
    r = r.join(deadlines, on=["season", "round"], how="inner")
    sel = r["selected"].astype(float)
    managers = sel.groupby([r["season"], r["round"]]).transform("sum") / 15
    r = r.assign(
        **_ratios(
            r["transfers_in"].astype(float),
            r["transfers_out"].astype(float),
            sel,
            sel / managers.where(managers > 0),
        )
    )
    first = r["round"] == 1  # no transfers before GW1: the ratios carry no news
    r.loc[first, ["tr_sell", "tr_buy"]] = np.nan
    r["event_at"] = r["observed_at"]
    frames = [f for f in (r[OUT], live) if not f.empty]
    out: pd.DataFrame = pd.concat(frames, ignore_index=True) if frames else live
    return out
