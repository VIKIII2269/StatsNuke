from __future__ import annotations

import numpy as np
import pandas as pd

from fplh.evaluate.phase2 import closing_devig_calibration
from fplh.evaluate.team_level import (
    GRID_COLS,
    FusedPredictor,
    M1Predictor,
    MarketPredictor,
    fixture_losses,
    market_rates,
    summarise,
)
from fplh.evaluate.walk_forward import run_walk_forward
from fplh.features.information_set import InformationSet, SilverStore
from fplh.models.fusion import Fusion
from fplh.models.goal_benchmarks import GoalModel, benchmarks
from fplh.models.market import devig
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


def test_market_rates_reproduce_the_prices_under_the_grid_model() -> None:
    frames = make()
    frames["snap_odds"] = _odds(frames)
    g1 = benchmarks()["G1"]
    g1 = GoalModel(g1.name, g1.grid_fn, {"rho": -0.12}, g1.bounds)
    store = SilverStore.from_frames(frames)
    ds = deadlines(frames)
    for method in ("multiplicative", "shin"):
        pred = run_walk_forward(
            store, MarketPredictor(g1, devig_method=method), ds, unit="fixture", write=False
        ).predictions
        target = devig([2.0, 3.4, 4.0], method)
        got = pred[["p_home", "p_draw", "p_away"]].to_numpy(float)
        assert len(pred) and np.abs(got - target).max() < 1e-4


def test_market_maximum_is_not_a_book() -> None:
    frames = make()
    odds = _odds(frames)
    odds["bookmaker"] = "market_max"
    frames["snap_odds"] = odds
    fx = frames["dim_fixture"].head(3)
    info = InformationSet.at(max(deadlines(frames)), SilverStore.from_frames(frames))
    assert market_rates(info, fx, benchmarks()["G0"])["lam_mkt_home"].isna().all()


def test_closing_calibration_uses_only_observable_closing_prices() -> None:
    frames = make()
    dim = frames["dim_fixture"]
    odds = _odds(frames).assign(is_closing=True, division="E0")
    odds["observed_at"] = odds["kickoff_at"]
    frames["snap_odds"] = odds
    frames["fd_match"] = dim[["fixture_uid", "home_goals", "away_goals", "kickoff_at"]].assign(
        division="E0", observed_at=dim["kickoff_at"] + pd.Timedelta(hours=48)
    )
    store = SilverStore.from_frames(frames)
    mid = dim["kickoff_at"].sort_values().iloc[len(dim) // 2]
    table = closing_devig_calibration(store, mid)
    assert list(table["bookmaker"]) == ["market_avg"]
    assert 0 < table["n"].iloc[0] < len(dim)  # later matches are not observable yet
    assert set(table.columns) >= {"multiplicative", "power", "shin"}
