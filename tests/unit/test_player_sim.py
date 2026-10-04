"""PR 6: the player simulator as a walk-forward predictor, attribution and ep_next."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fplh.evaluate.attribution import decompose, forced_side
from fplh.evaluate.ep_next import compare_ep_next, ep_next_at
from fplh.evaluate.phase2 import load_m1_params
from fplh.evaluate.phase3 import crps_rows
from fplh.evaluate.walk_forward import run_walk_forward
from fplh.features.information_set import InformationSet, SilverStore
from fplh.features.spine import build_spine, target_fixtures
from fplh.models.player_sim import PMF_RANGE, PlayerSimulator
from fplh.rules.config import load_rules
from fplh.sim.emulator import Emulator
from fplh.sim.simulator import (
    NEVER,
    FixtureInputs,
    ForcedSide,
    League,
    TimingModel,
    simulate_fixture,
)
from tests.property.test_simulator_properties import PARAMS, side
from tests.synthetic_silver import deadlines, make, with_understat


@pytest.fixture(scope="module")
def emulator() -> Emulator:
    return Emulator.build(PARAMS, grid=6, n_sims=3000)


def synthetic_rates(store: SilverStore, ds: list[pd.Timestamp]) -> pd.DataFrame:
    frames = []
    for d in ds:
        fx = target_fixtures(InformationSet.at(d, store), 1)
        frames.append(fx[["fixture_uid"]].assign(deadline_at=d, mu_home=1.5, mu_away=1.1))
    return pd.concat(frames, ignore_index=True)


def test_player_simulator_walk_forward(emulator: Emulator) -> None:
    f = with_understat(make())
    store = SilverStore.from_frames(f)
    ds = deadlines(f)[2:]
    rates = synthetic_rates(store, ds)
    for variant in ({}, {"minutes": "naive", "attack": "raw"}):
        sim = PlayerSimulator(rates, emulator, PARAMS, load_m1_params(), n_sims=300, **variant)  # type: ignore[arg-type]
        res = run_walk_forward(store, sim, ds, write=False)
        p = res.predictions
        assert len(p) and p["expected_points"].notna().all()
        lo, hi = PMF_RANGE
        pmf = p[[f"pmf_{k}" for k in range(lo, hi + 1)]].sum(axis=1)
        assert np.allclose(pmf, 1.0)
        assert p["p_60"].between(0, 1).all() and (p["p_60"] <= p["p_play"] + 1e-12).all()
        # the pmf's mean is the expected points wherever no mass is folded into its ends
        mean = sum(k * p[f"pmf_{k}"] for k in range(lo, hi + 1))
        inner = (p[f"pmf_{lo}"] == 0) & (p[f"pmf_{hi}"] == 0)
        assert inner.any()
        assert np.allclose(mean[inner], p.loc[inner, "expected_points"])
        crps = crps_rows(p, pd.Series(np.full(len(p), 2)))
        assert (crps >= 0).all()


def test_forced_minutes_and_goals_are_imposed() -> None:
    h, a = side("h", 1), side("a", 2)
    n = len(h)
    on = np.full(n, NEVER)
    on[[0, *range(2, 12)]] = 0  # GK + ten outfield start
    on[12] = 70
    off = np.full(n, 90)
    off[5] = 70  # replaced by player 12
    forced_h = ForcedSide(on, off, np.zeros(n, dtype=np.int64))
    goals = {
        c: np.zeros(len(a), dtype=np.int64)
        for c in ("goals_scored", "assists", "own_goals", "goals_conceded", "penalties_missed")
    }
    goals["goals_scored"][3] = 2
    on_a = np.where(np.arange(len(a)) < 11, 0, NEVER)
    forced_a = ForcedSide(on_a, np.full(len(a), 90), np.zeros(len(a), dtype=np.int64), goals)
    fx = FixtureInputs("s:h:a", (1.4, 1.1), (h, a), 5, forced=(forced_h, forced_a))
    r = simulate_fixture(fx, PARAMS, load_rules("2024/25"), League(), TimingModel(), 200, 1)
    mins_h = r.sides[0].events["minutes"]
    expected = np.where(on == NEVER, 0, np.minimum(off, 90) - np.minimum(on, 90))
    assert (mins_h == expected[None, :]).all()
    assert (r.sides[1].events["goals_scored"][:, 3] == 2).all()
    assert r.sides[1].events["goals_scored"].sum(1).max() == 2


def test_forced_side_from_actuals_and_decomposition() -> None:
    actual = pd.DataFrame(
        {
            "minutes": [90, 60, 20, 0],
            "starts": [1, 1, 0, 0],
            "red_cards": [0, 0, 0, 0],
            "goals_scored": [1, 0, 0, 0],
            "assists": [0, 1, 0, 0],
            "own_goals": [0, 0, 0, 0],
            "goals_conceded": [1, 1, 0, 0],
            "penalties_missed": [0, 0, 0, 0],
            "total_points": [8, 4, 1, 0],
        },
        index=["p0", "p1", "p2", "p3"],
    )
    f = forced_side(np.array(["p0", "p1", "p2", "p3", "absent"]), actual, goals=True)
    assert f.on.tolist() == [0, 0, 70, NEVER, NEVER]
    assert f.off.tolist()[:3] == [90, 60, 90]
    assert f.goals is not None and f.goals["goals_scored"].tolist() == [1, 0, 0, 0, 0]
    rng = np.random.default_rng(0)
    df = pd.DataFrame({c: rng.normal(2, 1, 50) for c in ("y0", "y1", "y2", "y")})
    out = decompose(df)
    total = out["minutes_part"] + out["goals_part"] + out["rest_part"]
    assert np.allclose(total, out["error"])


def test_ep_next_comparison_on_captured_gameweeks() -> None:
    f = make()
    store = SilverStore.from_frames(f)
    ds = deadlines(f)
    eps = ep_next_at(store, ds)
    assert len(eps) and eps["ep_next"].notna().all()
    snap = f["snap_fpl_player"]
    # the value is the last capture at or before the deadline
    d = ds[3]
    uid = eps.loc[eps["deadline_at"] == d, "player_uid"].iloc[0]
    code = int(uid.split(":")[1])
    s = snap[(snap["code"] == code) & (snap["observed_at"] <= d)].sort_values("observed_at")
    got = eps.loc[(eps["deadline_at"] == d) & (eps["player_uid"] == uid), "ep_next"].iloc[0]
    assert got == s["ep_next"].iloc[-1]
    pred = pd.concat(
        [build_spine(InformationSet.at(dd, store), 1).assign(expected_points=2.0) for dd in ds]
    )
    rows, summary = compare_ep_next(pred, store, n_boot=200)
    assert len(rows) and summary["n"].iloc[0] == len(rows)
    se = (rows["expected_points"] - rows["total_points"]) ** 2  # summed over the gameweek
    assert np.isclose(summary["mse_simulator"].iloc[0], se.mean())
    late = replace_capture_times(f, pd.Timedelta(days=-30))
    assert ep_next_at(SilverStore.from_frames(late), ds).empty  # stale captures are ignored


def replace_capture_times(
    f: dict[str, pd.DataFrame], shift: pd.Timedelta
) -> dict[str, pd.DataFrame]:
    snap = f["snap_fpl_player"].copy()
    snap["observed_at"] = snap["observed_at"].min() + shift
    return {**f, "snap_fpl_player": snap}
