"""OpenFPL re-implemented as a walk-forward benchmark (Phase 3 exit gate).

OpenFPL (Groos 2025) predicts a player's points from trailing FPL and Understat
statistics with gradient-boosted trees, one model per position. We re-implement it on
our silver tables instead of loading its published models (they were trained on our
evaluation seasons and are pickles): ``features.openfpl`` builds its feature set and
``OpenFPLReplica`` trains XGBoost walk-forward. At deadline D the training set is every
player-fixture whose outcome is observable at D, featurised at *its own* round deadline,
so the model never sees anything a forecaster at that deadline could not.

Hyper-parameters are fixed (tuned on 2021/22, before the evaluation seasons).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
import xgboost as xgb

from fplh.features.information_set import InformationSet
from fplh.features.openfpl import POSITIONS, openfpl_features
from fplh.features.spine import SPINE_KEYS, historical_deadlines
from fplh.features.trailing import trailing_means

DEFAULT_PARAMS: dict[str, Any] = {
    "eta": 0.03,
    "max_depth": 5,
    "min_child_weight": 10,
    "subsample": 0.8,
    "colsample_bytree": 0.6,
    "lambda": 1.0,
    "tree_method": "hist",
    "objective": "reg:squarederror",
    "seed": 0,
    "nthread": 4,
}
# Boosting rounds per position: early stopping on 2021/22 with training on 2016/17-2020/21.
DEFAULT_ROUNDS = {"GK": 135, "DEF": 166, "MID": 114, "FWD": 80}


def round_deadlines(info: InformationSet) -> pd.Series:
    """fixture_uid → its round's deadline (the schedule is public in advance)."""
    dim = info.table("dim_fixture").dropna(subset=["round"])
    frames = []
    for season in sorted(dim["season"].unique()):
        d = historical_deadlines(dim, season)
        frames.append(dim[dim["season"] == season][["fixture_uid", "round"]].merge(d, on="round"))
    if not frames:
        return pd.Series(dtype="datetime64[ns, UTC]")
    out: pd.Series = pd.concat(frames).set_index("fixture_uid")["deadline_at"]
    return out


def training_rows(info: InformationSet, first_season: str = "2016-17") -> pd.DataFrame:
    """Player-fixtures with an observable outcome at D, as of their own round deadline,
    for players already seen in that season before it (as ``build_spine`` requires)."""
    pm = info.table("fact_player_match")
    pm = pm[pm["season"] >= first_season]
    if pm.empty:
        return pd.DataFrame()
    keys = ["player_uid", "fixture_uid", "season", "team", "opponent", "was_home", "position"]
    rows = pm[[*keys, "total_points", "minutes", "kickoff_at", "observed_at"]].copy()
    rows["deadline_at"] = rows["fixture_uid"].map(round_deadlines(info))
    rows = rows.dropna(subset=["deadline_at"])
    seen = pm.assign(_ps=pm["player_uid"] + "|" + pm["season"], _one=1.0)
    count = trailing_means(
        seen, "_ps", ("_one",), (1,), rows.assign(_ps=rows["player_uid"] + "|" + rows["season"])
    )
    return rows[count["n"].to_numpy() > 0].reset_index(drop=True)


@dataclass
class OpenFPLReplica:
    """Walk-forward predictor: one XGBoost regressor per position, refit every
    ``refit_every`` deadlines on all history observable at the deadline."""

    refit_every: int = 4
    first_season: str = "2016-17"
    params: dict[str, Any] = field(default_factory=lambda: dict(DEFAULT_PARAMS))
    rounds: dict[str, int] = field(default_factory=lambda: dict(DEFAULT_ROUNDS))
    name: str = "openfpl_replica"
    version: str = "1"
    _models: dict[str, xgb.Booster] = field(default_factory=dict, repr=False)
    _fallback: float = field(default=0.0, repr=False)
    _calls: int = field(default=0, repr=False)

    def fit(self, info: InformationSet) -> OpenFPLReplica:
        rows = training_rows(info, self.first_season)
        self._models = {}
        if rows.empty:
            return self
        self._fallback = float(rows["total_points"].mean())  # positions too rare to model
        x = openfpl_features(info, rows)
        for pos in POSITIONS:
            mask = (rows["position"] == pos).to_numpy()
            if mask.sum() < 50:
                continue
            train = xgb.DMatrix(x[mask], label=rows.loc[mask, "total_points"].astype(float))
            self._models[pos] = xgb.train(self.params, train, num_boost_round=self.rounds[pos])
        return self

    def predict(self, info: InformationSet, spine: pd.DataFrame) -> pd.DataFrame:
        if self._calls % self.refit_every == 0:
            self.fit(info)
        self._calls += 1
        x = openfpl_features(info, spine)
        pred = np.full(len(spine), self._fallback)
        for pos, model in self._models.items():
            mask = (spine["position"] == pos).to_numpy()
            if mask.any():
                pred[mask] = model.predict(xgb.DMatrix(x[mask]))
        out = spine[[*SPINE_KEYS, "position"]].copy()
        out["expected_points"] = pred
        return out
