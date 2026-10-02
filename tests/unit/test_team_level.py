from __future__ import annotations

import pandas as pd

from fplh.evaluate.team_level import (
    GRID_COLS,
    FusedPredictor,
    M1Predictor,
    MarketPredictor,
    fixture_losses,
    summarise,
)
from fplh.evaluate.walk_forward import run_walk_forward
from fplh.features.information_set import SilverStore
from fplh.models.fusion import Fusion
from fplh.models.team_strength import TeamStrengthParams
from tests.synthetic_silver import deadlines, make
from tests.unit.test_walk_forward import _odds


def test_m1_walk_forward_is_leak_free_and_coherent() -> None:
    frames = make()
    res = run_walk_forward(
        SilverStore.from_frames(frames),
        M1Predictor(TeamStrengthParams()),
        deadlines(frames),
        unit="fixture",
        write=False,
    )
    p = res.predictions
    assert len(p) == len(frames["dim_fixture"])  # every fixture forecast once at horizon 1
    assert ((p[["p_home", "p_draw", "p_away"]].sum(axis=1) - 1).abs() < 1e-9).all()
    assert (p[GRID_COLS].sum(axis=1) <= 1 + 1e-9).all()
    assert pd.Timestamp(res.manifest.max_observed_at) <= max(deadlines(frames))
    losses = fixture_losses(
        p, frames["dim_fixture"][["fixture_uid", "home_goals", "away_goals", "round", "season"]]
    )
    s = summarise(losses)
    assert s["n"] == len(p) and 0 < s["rps"] < 0.5


def test_market_and_fused_predictors() -> None:
    frames = make()
    frames["snap_odds"] = _odds(frames)
    store = SilverStore.from_frames(frames)
    ds = deadlines(frames)
    market = run_walk_forward(store, MarketPredictor(), ds, unit="fixture", write=False).predictions
    assert (market["lambda_home"] > market["lambda_away"]).all()  # prices favour the home side
    fused = run_walk_forward(
        store,
        FusedPredictor(M1Predictor(TeamStrengthParams()), Fusion((10.0, 0.0, 0.0))),
        ds,
        unit="fixture",
        write=False,
    ).predictions
    assert (fused["market_weight"] > 0.99).all()
    merged = fused.merge(market, on="fixture_uid", suffixes=("", "_mkt"))
    assert ((merged["lambda_home"] - merged["lambda_home_mkt"]).abs() < 1e-3).all()
