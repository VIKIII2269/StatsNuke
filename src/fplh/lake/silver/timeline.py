"""Match timelines from Understat rosters and shots: lineups and ``fact_match_event``.

FBref is not reachable, so substitution and red-card timing come from Understat rosters
(Phase 3 decision, ARCHITECTURE.md §15 open question 2):

* a starter's ``roster_in`` is the roster ``id`` of the substitute who replaced them, and
  the substitute's ``roster_out`` points back;
* Understat ``time`` (``minutes``) is minutes played, capped at 90, so a player replaced
  at minute *m* shows *m* and the substitute 90 − *m*. When the replaced player shows 90
  (substituted in stoppage time) the substitution minute is 90 − the substitute's minutes;
* a sent-off player is never replaced, so the red-card minute is the minute they left;
  a red shown at minute 90 or later is flagged ``red_late``;
* yellow-card minutes are not recorded anywhere we can reach.

Goal times come from ``fact_shot`` (``result`` Goal / OwnGoal; an own-goal row carries
the scorer's side and counts for the opponent).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

FULL_TIME = 90
LINEUP_COLUMNS = [
    "started",
    "on_minute",
    "off_minute",
    "off_reason",
    "replaced_roster_id",
    "replaced_by_roster_id",
    "red_minute",
    "red_late",
]
EVENT_KINDS = ("goal", "own_goal", "sub", "red", "unreplaced_off")
EVENT_COLUMNS = [
    "season",
    "understat_match_id",
    "fixture_uid",
    "side",
    "team",
    "minute",
    "kind",
    "seq",
    "is_penalty",
    "player_uid",
    "understat_player_id",
    "other_player_uid",
    "score_home_after",
    "score_away_after",
    "reds_home_after",
    "reds_away_after",
    "event_at",
    "observed_at",
]
OTHER_SIDE = {"h": "a", "a": "h"}


def _side_lineup(g: pd.DataFrame, notes: dict[str, int]) -> dict[int, dict[str, Any]]:
    rows = {int(r["roster_id"]): r for r in g.to_dict("records")}
    out: dict[int, dict[str, Any]] = {}

    def resolve(rid: int, depth: int = 0) -> dict[str, Any]:
        if rid in out:
            return out[rid]
        r = rows[rid]
        minutes = int(r["minutes"])
        started = r["position"] != "Sub"
        partner_out = int(r["roster_out"] or 0)  # the player this one replaced
        if started:
            on = 0
        elif partner_out in rows and depth < 10:
            on = int(resolve(partner_out, depth + 1)["off_minute"])
        else:
            notes["unpaired_subs"] += 1
            on = max(FULL_TIME - minutes, 0)
        replaced_by = int(r["roster_in"] or 0)
        end = on + minutes
        if replaced_by in rows:
            sub = rows[replaced_by]
            # 90 − the substitute's minutes is the substitution minute only if the
            # substitute then plays to the end (not replaced again, not sent off).
            to_end = int(sub["roster_in"] or 0) not in rows and int(sub["red_cards"]) == 0
            implied = FULL_TIME - int(sub["minutes"])
            if end < FULL_TIME:
                if to_end:
                    notes["sub_pairs"] += 1
                    if abs(end - implied) > 1:
                        notes["sub_minute_disagreements"] += 1
                off = end
            else:
                off = implied if to_end else end
            off, reason = int(np.clip(off, on, FULL_TIME)), "sub"
        else:
            off = min(end, FULL_TIME)
            if int(r["red_cards"]) > 0:
                reason = "red"
            else:
                reason = "full" if off >= FULL_TIME else "unreplaced"
        red = int(r["red_cards"]) > 0
        out[rid] = {
            "started": started,
            "on_minute": on,
            "off_minute": off,
            "off_reason": reason,
            "replaced_roster_id": partner_out if partner_out in rows else 0,
            "replaced_by_roster_id": replaced_by if replaced_by in rows else 0,
            "red_minute": off if red else pd.NA,
            "red_late": bool(red and end >= FULL_TIME),
        }
        return out[rid]

    for rid in rows:
        resolve(rid)
    return out


def derive_lineups(roster: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    """``fact_player_match_understat`` rows plus LINEUP_COLUMNS, and build notes."""
    notes = {"sub_pairs": 0, "sub_minute_disagreements": 0, "unpaired_subs": 0}
    out = roster.copy()
    if out.empty:
        for c in LINEUP_COLUMNS:
            out[c] = pd.Series(dtype="object")
        return out, notes
    found: dict[tuple[int, int], dict[str, Any]] = {}
    for (mid, _side), g in out.groupby(["understat_match_id", "side"], sort=False):
        for rid, v in _side_lineup(g, notes).items():
            found[(int(str(mid)), rid)] = v
    keys = zip(out["understat_match_id"].astype(int), out["roster_id"].astype(int), strict=True)
    cols = pd.DataFrame([found[k] for k in keys], index=out.index)
    for c in LINEUP_COLUMNS:
        out[c] = cols[c]
    out["started"] = out["started"].astype(bool)
    out["red_late"] = out["red_late"].astype(bool)
    for c in ("on_minute", "off_minute", "replaced_roster_id", "replaced_by_roster_id"):
        out[c] = out[c].astype("int64")
    out["red_minute"] = out["red_minute"].astype("Int64")
    return out, notes


def _player_events(lineups: pd.DataFrame) -> pd.DataFrame:
    by_rid = lineups.set_index(["understat_match_id", "roster_id"])
    subs = lineups[~lineups["started"]]
    replaced = by_rid.reindex(
        pd.MultiIndex.from_arrays([subs["understat_match_id"], subs["replaced_roster_id"]])
    )
    reds = lineups[lineups["red_minute"].notna()]
    gone = lineups[lineups["off_reason"] == "unreplaced"]
    parts = [
        subs.assign(
            kind="sub", minute=subs["on_minute"], other_player_uid=replaced["player_uid"].to_numpy()
        ),
        reds.assign(kind="red", minute=reds["red_minute"].astype("int64")),
        gone.assign(kind="unreplaced_off", minute=gone["off_minute"]),
    ]
    out = pd.concat(parts, ignore_index=True)
    return out.assign(is_penalty=False, seq=0)


def _goal_events(shots: pd.DataFrame) -> pd.DataFrame:
    g = shots[shots["result"].isin(["Goal", "OwnGoal"])].copy()
    own = g["result"] == "OwnGoal"
    g["kind"] = np.where(own, "own_goal", "goal")
    g["side"] = np.where(own, g["side"].map(OTHER_SIDE), g["side"])  # side credited
    g["is_penalty"] = g["situation"] == "Penalty"
    g["other_player_uid"] = g.get("assister_uid", pd.NA)
    g["seq"] = g["shot_id"]
    return g


def match_events(
    lineups: pd.DataFrame, shots: pd.DataFrame, us_match: pd.DataFrame
) -> pd.DataFrame:
    """One row per goal, own goal, substitution, red card and unreplaced exit, in match
    order, with the score and red cards *after* each event. ``side`` is the side credited
    with a goal (the opponent of an own-goal scorer) and otherwise the player's side."""
    if lineups.empty or us_match.empty:
        return pd.DataFrame(columns=EVENT_COLUMNS)
    teams = us_match.set_index("understat_match_id")[["home_team", "away_team"]]
    ev = pd.concat([_goal_events(shots), _player_events(lineups)], ignore_index=True)
    order = {k: i for i, k in enumerate(EVENT_KINDS)}
    ev["kind_order"] = ev["kind"].map(order)
    ev = ev.sort_values(["understat_match_id", "minute", "kind_order", "seq"], kind="stable")
    goal = ev["kind"].isin(["goal", "own_goal"])
    red = ev["kind"] == "red"
    grp = ev.groupby("understat_match_id", sort=False)
    for side, name in (("h", "home"), ("a", "away")):
        ev[f"_g_{side}"] = (goal & (ev["side"] == side)).astype(int)
        ev[f"_r_{side}"] = (red & (ev["side"] == side)).astype(int)
        ev[f"score_{name}_after"] = grp[f"_g_{side}"].cumsum()
        ev[f"reds_{name}_after"] = grp[f"_r_{side}"].cumsum()
    home = ev["understat_match_id"].map(teams["home_team"])
    away = ev["understat_match_id"].map(teams["away_team"])
    ev["team"] = np.where(ev["side"] == "h", home, away)
    ev["minute"] = ev["minute"].astype("int64")
    ev["seq"] = ev["seq"].astype("int64")
    ev["is_penalty"] = ev["is_penalty"].astype(bool)
    return ev[EVENT_COLUMNS].reset_index(drop=True)
