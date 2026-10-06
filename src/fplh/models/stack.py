"""M11 v1: a walk-forward stacking layer over the base forecasters (model v2).

The simulator and the OpenFPL replica are built differently and their errors are only
partly correlated; FPL's crowd (ownership, transfers) and price carry information neither
uses. The stack predicts a player-fixture's points from

* the simulator's summary (expected points, P(start), P(60+), expected goals, assists,
  bonus, P(clean sheet), the points sd) and the replica's expected points;
* the crowd at the deadline (``features.transfers``: owners selling, buying, ownership);
* the latest price observed at the deadline, position and the forecast horizon.

Gradient-boosted trees (shallow, heavily regularised) start from the simulator's forecast
(``base_margin``; ``StackConfig.blend`` mixes in the replica), so with little data the
stack is the simulator. Walk-forward:
at each deadline D the trees are trained on earlier deadlines' rows whose outcome was
observed by D (refit every ``refit_every`` deadlines) and predict the rows at D. The base
forecasts are themselves walk-forward, so no row ever sees its own outcome.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import xgboost as xgb

from fplh.features.spine import asof_join
from fplh.features.transfers import COLUMNS as CROWD

KEYS = ["player_uid", "fixture_uid", "deadline_at"]
SIM = {
    "expected_points": "sim_ep",
    "sd_points": "sim_sd",
    "p_start": "sim_p_start",
    "p_60": "sim_p_60",
    "e_goals": "sim_goals",
    "e_assists": "sim_assists",
    "e_bonus": "sim_bonus",
    "p_clean_sheet": "sim_cs",
}
POSITIONS = ("GK", "DEF", "MID", "FWD")
FEATURES = [
    *SIM.values(),
    "rep_ep",
    *CROWD,
    "price",
    "horizon",
    *(f"pos_{p}" for p in POSITIONS),
]
PARAMS: dict[str, Any] = {
    "objective": "reg:squarederror",
    "eta": 0.03,
    "max_depth": 3,
    "min_child_weight": 200,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "lambda": 10.0,
    "tree_method": "hist",
    "seed": 0,
    "nthread": 4,
}


@dataclass(frozen=True)
class StackConfig:
    rounds: int = 300
    refit_every: int = 4
    # weight of the simulator in the starting blend: the news-aware simulator alone beats
    # any blend with the replica on 2021/22, so the trees start from it
    blend: float = 1.0
    min_rows: int = 5000  # below this the stack is the blend


def stack_frame(
    sim: pd.DataFrame, rep: pd.DataFrame, crowd: pd.DataFrame, prices: pd.DataFrame
) -> pd.DataFrame:
    """One row per (player, fixture, deadline) both forecasters scored, with features.

    ``crowd``: the ``fpl_round_transfers`` view; ``prices``: player_uid, value,
    observed_at (``fact_player_match``)."""
    sim = sim if "horizon" in sim else sim.assign(horizon=1)
    s = sim[[*KEYS, "position", "horizon", *SIM]].rename(columns=SIM)
    r = rep[[*KEYS, "expected_points"]].rename(columns={"expected_points": "rep_ep"})
    f = s.merge(r, on=KEYS, how="inner").dropna(subset=["sim_ep", "rep_ep"])
    f = f.sort_values(KEYS).reset_index(drop=True)
    j = asof_join(f[["player_uid", "deadline_at"]], crowd, ["player_uid"], list(CROWD))
    for c in CROWD:
        f[c] = j[c].to_numpy(dtype=float)
    pr = prices.dropna(subset=["value"])
    jp = asof_join(f[["player_uid", "deadline_at"]], pr, ["player_uid"], ["value"])
    f["price"] = jp["value"].to_numpy(dtype=float) / 10
    for p in POSITIONS:
        f[f"pos_{p}"] = (f["position"] == p).astype(float)
    return f


def repeat_forecasts(rep: pd.DataFrame, rows: pd.DataFrame) -> pd.DataFrame:
    """A horizon-1 forecaster's per-fixture forecast at D carried to every later fixture
    of the player in ``rows`` (player_uid, fixture_uid, deadline_at), as in the replay."""
    nxt = rep.groupby(["player_uid", "deadline_at"])["expected_points"].mean().reset_index()
    out: pd.DataFrame = (
        rows[KEYS].drop_duplicates().merge(nxt, on=["player_uid", "deadline_at"], how="inner")
    )
    return out


def _margin(f: pd.DataFrame, w: float) -> np.ndarray:
    out: np.ndarray = (w * f["sim_ep"] + (1 - w) * f["rep_ep"]).to_numpy(float)
    return out


def walk_forward_stack(
    frame: pd.DataFrame,
    outcomes: pd.DataFrame,
    deadlines: Sequence[pd.Timestamp],
    config: StackConfig | None = None,
    targets: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Stacked expected points for the rows at each of ``deadlines``.

    Trains on ``frame``; predicts the rows of ``targets`` (default ``frame``), e.g. a
    horizon-5 frame stacked by trees trained on horizon-1 forecasts.
    ``outcomes``: player_uid, fixture_uid, total_points, observed_at."""
    cfg = config or StackConfig()
    y = outcomes[["player_uid", "fixture_uid", "total_points", "observed_at"]].rename(
        columns={"observed_at": "y_at"}
    )
    f = frame.merge(y, on=["player_uid", "fixture_uid"], how="left")
    rows = f if targets is None else targets
    out = []
    booster: xgb.Booster | None = None
    for i, d in enumerate(sorted(deadlines)):
        now = rows[rows["deadline_at"] == d]
        if now.empty:
            continue
        if booster is None or i % cfg.refit_every == 0:
            train = f[(f["deadline_at"] < d) & (f["y_at"] <= d) & f["total_points"].notna()]
            booster = None
            if len(train) >= cfg.min_rows:
                dm = xgb.DMatrix(
                    train[FEATURES],
                    label=train["total_points"],
                    base_margin=_margin(train, cfg.blend),
                )
                booster = xgb.train(PARAMS, dm, num_boost_round=cfg.rounds)
        base = _margin(now, cfg.blend)
        if booster is None:
            pred = base
        else:
            pred = np.asarray(
                booster.predict(xgb.DMatrix(now[FEATURES], base_margin=base)), dtype=float
            )
        out.append(now[[*KEYS, "position"]].assign(expected_points=np.clip(pred, -1.0, None)))
    if not out:
        return pd.DataFrame(columns=[*KEYS, "position", "expected_points"])
    return pd.concat(out, ignore_index=True)
