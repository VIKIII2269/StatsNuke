"""The simulator against FPL's own ``ep_next`` on live gameweeks (gap 2).

``ep_next`` is FPL's expected points for the next gameweek, published in the bootstrap
data we capture (``snap_fpl_player``, from 2026-09-30). For each deadline D, the last
capture observed at or before D, and no older than ``max_age``, gives the player's
``ep_next``; the simulator's expected points are summed over the player's fixtures in
that gameweek (double gameweeks), as are the actual points. Both are scored by squared
error on the same player-gameweeks, compared with the gameweek-block bootstrap.

Historical seasons have no captures, so this runs only on live gameweeks.
"""

from __future__ import annotations

import pandas as pd

from fplh.evaluate.bootstrap import compare
from fplh.features.information_set import SilverStore
from fplh.features.spine import asof_join

MAX_AGE = pd.Timedelta(days=4)


def ep_next_at(store: SilverStore, deadlines: list[pd.Timestamp]) -> pd.DataFrame:
    """player_uid, deadline_at, ep_next: the last capture observed at or before D."""
    snap = store.get("snap_fpl_player")
    if snap.empty or "ep_next" not in snap:
        return pd.DataFrame(columns=["player_uid", "deadline_at", "ep_next"])
    s = snap.assign(player_uid="fpl:" + snap["code"].astype(str), captured_at=snap["observed_at"])[
        ["player_uid", "observed_at", "captured_at", "ep_next"]
    ]
    players = s["player_uid"].unique()
    queries = pd.DataFrame(
        [(p, d) for d in deadlines for p in players], columns=["player_uid", "deadline_at"]
    )
    j = asof_join(queries, s, ["player_uid"], ["captured_at", "ep_next"])
    fresh = (j["deadline_at"] - j["captured_at"]) <= MAX_AGE
    out: pd.DataFrame = j.loc[
        fresh & j["ep_next"].notna(), ["player_uid", "deadline_at", "ep_next"]
    ]
    return out.reset_index(drop=True)


def compare_ep_next(
    pred: pd.DataFrame, store: SilverStore, n_boot: int = 2000
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """``pred``: player_uid, fixture_uid, deadline_at, expected_points (walk-forward)."""
    pm = store.get("fact_player_match")[["player_uid", "fixture_uid", "total_points"]]
    per_fixture = pred[["player_uid", "fixture_uid", "deadline_at", "expected_points"]].merge(
        pm, on=["player_uid", "fixture_uid"]
    )
    gw = per_fixture.groupby(["player_uid", "deadline_at"], as_index=False)[
        ["expected_points", "total_points"]
    ].sum()
    deadlines = sorted(gw["deadline_at"].unique())
    rows = gw.merge(
        ep_next_at(store, [pd.Timestamp(d) for d in deadlines]),
        on=[
            "player_uid",
            "deadline_at",
        ],
    )
    if rows.empty:
        return rows, pd.DataFrame(
            [{"n": 0, "note": "no live gameweek with both captures and results"}]
        )
    rows["se_simulator"] = (rows["expected_points"] - rows["total_points"]) ** 2
    rows["se_ep_next"] = (rows["ep_next"] - rows["total_points"]) ** 2
    block = rows["deadline_at"].astype(str)
    c = compare(rows["se_simulator"], rows["se_ep_next"], block, n_boot=n_boot)
    summary = pd.DataFrame(
        [
            {
                "n": len(rows),
                "gameweeks": int(block.nunique()),
                "mse_simulator": float(rows["se_simulator"].mean()),
                "mse_ep_next": float(rows["se_ep_next"].mean()),
                "diff": c.mean_diff,
                "ci_low": c.ci_low,
                "ci_high": c.ci_high,
            }
        ]
    )
    return rows, summary
