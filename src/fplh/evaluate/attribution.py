"""Error attribution for the player simulator (ARCHITECTURE.md §11.5, ticket 3.8).

For each player-fixture, three simulations with the same inputs and seed:

* ŷ⁰: the forecast (everything simulated);
* ŷ¹: the actual minutes imposed (who started, when players came on and off, red cards);
* ŷ²: the actual minutes and the actual goal events (goals, assists, own goals, goals
  conceded while on the pitch, missed penalties); saves, cards, defensive actions and
  bonus stay simulated given them.

The error splits exactly into three parts that sum to it on every row:

    y − ŷ⁰ = (ŷ¹ − ŷ⁰)  [minutes]  +  (ŷ² − ŷ¹)  [goal events]  +  (y − ŷ²)  [the rest]
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd

from fplh.features.information_set import InformationSet, SilverStore
from fplh.features.spine import build_spine, historical_deadlines
from fplh.lake.storage import Lake
from fplh.models.player_sim import PlayerSimulator, Prepared
from fplh.rules.config import Rules
from fplh.sim.simulator import FULL, NEVER, FixtureInputs, ForcedSide

GOAL_EVENTS = ("goals_scored", "assists", "own_goals", "goals_conceded", "penalties_missed")
ACTUAL_COLUMNS = ("minutes", "starts", "red_cards", *GOAL_EVENTS, "total_points")


def forced_side(uids: np.ndarray, actual: pd.DataFrame, goals: bool) -> ForcedSide:
    """``actual``: index player_uid with ``ACTUAL_COLUMNS``; absent players did not play."""
    a = actual.reindex(uids)
    minutes = a["minutes"].fillna(0).to_numpy(dtype=np.int64)
    starts = a["starts"].to_numpy(dtype=float)
    started = np.where(np.isnan(starts), minutes >= 45, starts > 0) & (minutes > 0)
    on = np.where(minutes == 0, NEVER, np.where(started, 0, FULL - minutes))
    off = np.where(started, np.where(minutes >= FULL, FULL, minutes), FULL)
    events = {c: a[c].fillna(0).to_numpy(dtype=np.int64) for c in GOAL_EVENTS} if goals else None
    return ForcedSide(
        on.astype(np.int64),
        off.astype(np.int64),
        a["red_cards"].fillna(0).to_numpy(dtype=np.int64),
        events,
    )


def attribute_fixture(
    sim: PlayerSimulator,
    prep: Prepared,
    fx: FixtureInputs,
    rules: Rules,
    actual: pd.DataFrame,
) -> pd.DataFrame:
    """ŷ⁰, ŷ¹, ŷ² and the parts for every player of one fixture with an outcome."""
    preds = []
    for forced in (None, False, True):
        f = fx
        if forced is not None:
            home, away = (forced_side(s.player_uid, actual, forced) for s in fx.sides)
            f = replace(fx, forced=(home, away))
        res = sim.simulate(prep, f, rules)
        preds.append(np.concatenate([s.points.mean(axis=0) for s in res.sides]))
    uids = np.concatenate([s.player_uid for s in fx.sides])
    pos = np.concatenate([s.position for s in fx.sides])
    out = pd.DataFrame(
        {
            "player_uid": uids,
            "fixture_uid": fx.fixture_uid,
            "position": pos,
            "y0": preds[0],
            "y1": preds[1],
            "y2": preds[2],
        }
    )
    out = out[out["player_uid"].isin(actual.index)].reset_index(drop=True)
    out["y"] = actual.loc[out["player_uid"], "total_points"].to_numpy(dtype=float)
    return decompose(out)


def decompose(df: pd.DataFrame) -> pd.DataFrame:
    out = df.assign(
        error=df["y"] - df["y0"],
        minutes_part=df["y1"] - df["y0"],
        goals_part=df["y2"] - df["y1"],
        rest_part=df["y"] - df["y2"],
    )
    parts = out["minutes_part"] + out["goals_part"] + out["rest_part"]
    if not np.allclose(parts.to_numpy(), out["error"].to_numpy()):  # pragma: no cover
        raise AssertionError("attribution parts do not sum to the error")
    return out


def evaluate_attribution(
    lake: Lake, seasons: list[str], sim: PlayerSimulator, every: int = 1
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per-row attribution and a summary: MSE of each forecast and mean |part|."""
    store = SilverStore(lake)
    dim = store.get("dim_fixture")
    pm = store.get("fact_player_match")
    frames = []
    for season in seasons:
        for _, deadline in list(historical_deadlines(dim, season).itertuples(index=False))[::every]:
            info = InformationSet.at(deadline, store)
            spine = build_spine(info, 1)
            if spine.empty:  # no fixture in the next round (as in the walk-forward runner)
                continue
            prep = sim.prepare(info, spine)
            for fx, rules in prep.fixtures:
                actual = pm[pm["fixture_uid"] == fx.fixture_uid].drop_duplicates("player_uid")
                actual = actual.set_index("player_uid")[list(ACTUAL_COLUMNS)]
                frames.append(
                    attribute_fixture(sim, prep, fx, rules, actual).assign(
                        season=season, deadline_at=deadline
                    )
                )
    rows = pd.concat(frames, ignore_index=True)
    summary = pd.DataFrame(
        [
            {
                "position": pos,
                "n": len(g),
                "mse_forecast": float(((g["y"] - g["y0"]) ** 2).mean()),
                "mse_actual_minutes": float(((g["y"] - g["y1"]) ** 2).mean()),
                "mse_actual_minutes_goals": float(((g["y"] - g["y2"]) ** 2).mean()),
                "mean_abs_minutes_part": float(g["minutes_part"].abs().mean()),
                "mean_abs_goals_part": float(g["goals_part"].abs().mean()),
                "mean_abs_rest_part": float(g["rest_part"].abs().mean()),
            }
            for pos, g in [("all", rows), *rows.groupby("position")]
        ]
    )
    return rows, summary
