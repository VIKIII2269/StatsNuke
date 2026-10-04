"""M4 minutes model (ARCHITECTURE.md §7.6): staged start / 60+ / substitute probabilities.

* π^S = P(start), π^60 = P(60+ minutes | start), π^B = P(appearance | not started);
* gradient-boosted trees (XGBoost, native API) with monotone constraints (more recent
  starts never lower π^S), then isotonic calibration fitted on the chronologically last
  20 % of the training rows by a model trained on the first 80 %;
* trained walk-forward on every player-fixture observable at the deadline, each
  featurised at its own round deadline (``features.minutes``), refit every 4 deadlines.

Team consistency (applied after calibration, per team and fixture): Σ π^S over the
registered players is scaled to 11, and the expected substitute appearances
Σ (1 − π^S)·π^B to the substitutes a side actually uses under the fixture's limit,
estimated from the lineups observable at the deadline (2.75 under three substitutes,
3.76 under five). Trees cannot extrapolate the 2022/23 move to five substitutes from
a history of three; the scaling makes π^B calibrated across it.

The news overlay of gap 1 is a no-op until snapshot captures exist: ``chance_of_playing``
is a feature that is missing (never imputed) before our own captures began. Horizon
drift and the small-sample prior blend are not needed at horizon 1 (the trees already
see how many matches a player has, ``h_n``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd
import xgboost as xgb
from scipy.optimize import isotonic_regression

from fplh.features.information_set import InformationSet
from fplh.features.minutes import minutes_features, player_history
from fplh.features.spine import SPINE_KEYS
from fplh.models.openfpl import training_rows
from fplh.rules.football import load_substitution_rules

Array = npt.NDArray[np.float64]
PARAMS: dict[str, Any] = {
    "objective": "binary:logistic",
    "eta": 0.05,
    "max_depth": 5,
    "min_child_weight": 20,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "tree_method": "hist",
    "seed": 0,
    "nthread": 4,
}
ROUNDS = 250
STAGES = ("start", "full", "sub")
MONOTONE_UP = {"h_started_1", "h_started_3", "h_started_5", "h_started_10", "chance_of_playing"}


@dataclass
class Isotonic:
    """Monotone map from raw to calibrated probabilities (piecewise linear between knots)."""

    x: Array
    y: Array

    @classmethod
    def fit(cls, p: Array, y: Array) -> Isotonic:
        order = np.argsort(p, kind="mergesort")
        fitted = isotonic_regression(y[order].astype(float)).x
        return cls(p[order], np.asarray(fitted, dtype=float))

    def __call__(self, p: Array) -> Array:
        out: Array = np.interp(p, self.x, self.y)
        return out


def labelled_rows(info: InformationSet, first_season: str = "2016-17") -> pd.DataFrame:
    rows = training_rows(info, first_season)
    if rows.empty:
        return rows
    hist = player_history(info)[["player_uid", "fixture_uid", "started"]]
    rows = rows.merge(hist, on=["player_uid", "fixture_uid"], how="left")
    rows["y_start"] = rows["started"].fillna(0.0)
    rows["y_full"] = (rows["minutes"] >= 60).astype(float)
    rows["y_sub"] = (rows["minutes"] > 0).astype(float)
    return rows.sort_values(["deadline_at", "player_uid", "fixture_uid"]).reset_index(drop=True)


@dataclass
class _Stage:
    booster: xgb.Booster
    calibration: Isotonic

    def predict(self, x: pd.DataFrame) -> Array:
        raw = self.booster.predict(xgb.DMatrix(x))
        return self.calibration(np.asarray(raw, dtype=float))


def _train(x: pd.DataFrame, y: Array, rounds: int) -> xgb.Booster:
    cons = tuple(1 if c in MONOTONE_UP else 0 for c in x.columns)
    params = {**PARAMS, "monotone_constraints": "(" + ",".join(map(str, cons)) + ")"}
    return xgb.train(params, xgb.DMatrix(x, label=y), num_boost_round=rounds)


def fit_stage(x: pd.DataFrame, y: Array, rounds: int = ROUNDS) -> _Stage:
    cut = int(len(x) * 0.8)  # rows are in deadline order
    if len(x) - cut < 50 or len(np.unique(y[cut:])) < 2:  # too little to calibrate on
        return _Stage(_train(x, y, rounds), Isotonic(np.array([0.0, 1.0]), np.array([0.0, 1.0])))
    early = _train(x.iloc[:cut], y[:cut], rounds)
    raw = np.asarray(early.predict(xgb.DMatrix(x.iloc[cut:])), dtype=float).ravel()
    calibration = Isotonic.fit(raw, y[cut:])
    return _Stage(_train(x, y, rounds), calibration)


def subs_used_per_side(info: InformationSet) -> dict[int, float]:
    """Mean substitutes used per side by the era's substitution limit (lineups at D)."""
    us = info.table("fact_player_match_understat")
    matches = info.table("us_match")
    if us.empty or "started" not in us or matches.empty:
        return {}
    rules = load_substitution_rules()
    kickoff = matches.set_index("understat_match_id")["kickoff_at"].to_dict()
    sides = us.groupby(["understat_match_id", "side"]).size().index
    used = us[~us["started"].astype(bool)].groupby(["understat_match_id", "side"]).size()
    used = used.reindex(sides, fill_value=0)
    limit = [rules.limit(pd.Timestamp(kickoff[m])) for m, _ in used.index]
    by_limit = pd.Series(used.to_numpy(), index=limit).groupby(level=0).mean()
    return {int(str(k)): float(v) for k, v in by_limit.items()}


@dataclass
class MinutesModel:
    stages: dict[str, _Stage] = field(default_factory=dict)
    subs_per_side: dict[int, float] = field(default_factory=dict)

    @classmethod
    def fit(cls, info: InformationSet, rounds: int = ROUNDS) -> MinutesModel:
        rows = labelled_rows(info)
        x = minutes_features(info, rows)
        started = rows["y_start"].to_numpy() > 0
        stages = {
            "start": fit_stage(x, rows["y_start"].to_numpy(), rounds),
            "full": fit_stage(x[started], rows.loc[started, "y_full"].to_numpy(), rounds),
            "sub": fit_stage(x[~started], rows.loc[~started, "y_sub"].to_numpy(), rounds),
        }
        return cls(stages, subs_used_per_side(info))

    def predict(self, info: InformationSet, rows: pd.DataFrame) -> pd.DataFrame:
        """``rows``: player_uid, fixture_uid, team, position, kickoff_at, deadline_at."""
        x = minutes_features(info, rows)
        p = pd.DataFrame({f"p_{s}": self.stages[s].predict(x) for s in STAGES}, index=rows.index)
        grp = [rows["fixture_uid"], rows["team"]]
        total = p["p_start"].groupby(grp).transform("sum")
        p["p_start"] = (p["p_start"] * 11 / total.where(total > 0, 11)).clip(0.0, 1.0)
        if self.subs_per_side:
            rules = load_substitution_rules()
            fallback = self.subs_per_side[max(self.subs_per_side)]  # unseen limit
            target = np.array(
                [
                    self.subs_per_side.get(rules.limit(pd.Timestamp(k)), fallback)
                    for k in rows["kickoff_at"]
                ]
            )
            expected = ((1 - p["p_start"]) * p["p_sub"]).groupby(grp).transform("sum")
            scale = target / expected.where(expected > 0, np.nan).to_numpy()
            p["p_sub"] = np.clip(p["p_sub"] * np.nan_to_num(scale, nan=1.0), 0.0, 0.99)
        return p


@dataclass
class MinutesPredictor:
    """Walk-forward predictor: π^S, π^60 and π^B per spine row (M4)."""

    refit_every: int = 4
    rounds: int = ROUNDS
    name: str = "minutes"
    version: str = "1"
    _model: MinutesModel | None = field(default=None, repr=False)
    _calls: int = field(default=0, repr=False)

    def predict(self, info: InformationSet, spine: pd.DataFrame) -> pd.DataFrame:
        if self._model is None or self._calls % self.refit_every == 0:
            self._model = MinutesModel.fit(info, self.rounds)
        self._calls += 1
        p = self._model.predict(info, spine)
        out = spine[[*SPINE_KEYS, "position"]].copy()
        for c in p.columns:
            out[c] = p[c].to_numpy()
        return out
