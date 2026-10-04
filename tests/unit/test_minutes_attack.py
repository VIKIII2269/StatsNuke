from __future__ import annotations

import numpy as np
import pandas as pd

from fplh.evaluate.walk_forward import run_walk_forward
from fplh.features.information_set import InformationSet, SilverStore
from fplh.features.leakage import check_leakage
from fplh.features.minutes import minutes_features
from fplh.features.spine import SPINE_KEYS, build_spine
from fplh.models.attack import fit_attack
from fplh.models.minutes import Isotonic, MinutesModel, MinutesPredictor, subs_used_per_side
from tests.synthetic_silver import deadlines, make, with_understat


def frames() -> dict[str, pd.DataFrame]:
    return with_understat(make())


def keyed(info: InformationSet, spine: pd.DataFrame) -> pd.DataFrame:
    return pd.concat([spine[SPINE_KEYS], minutes_features(info, spine)], axis=1)


def test_minutes_features_are_leak_free() -> None:
    f = frames()
    assert check_leakage(f, deadlines(f), {"minutes": keyed}, horizon=1) == []


def test_isotonic_is_monotone_and_bounded() -> None:
    rng = np.random.default_rng(0)
    p = rng.random(2000)
    y = (rng.random(2000) < p**2).astype(float)
    iso = Isotonic.fit(p, y)
    out = iso(np.linspace(0, 1, 101))
    assert (np.diff(out) >= -1e-12).all() and out.min() >= 0 and out.max() <= 1


def test_minutes_walk_forward_and_era_shift() -> None:
    f = frames()
    store = SilverStore.from_frames(f)
    ds = deadlines(f)
    res = run_walk_forward(store, MinutesPredictor(refit_every=2, rounds=20), ds[2:], write=False)
    p = res.predictions
    for c in ("p_start", "p_full", "p_sub"):
        assert p[c].between(0, 1).all()
    info = InformationSet.at(ds[-1], store)
    model = MinutesModel.fit(info, rounds=20)
    spine = build_spine(info, 1)
    q = model.predict(info, spine)
    starts = q["p_start"].groupby([spine["fixture_uid"], spine["team"]]).sum()
    assert (starts <= 11 + 1e-9).all()  # five synthetic players per team: all capped at 1
    assert subs_used_per_side(info)  # lineups give substitutes used per side by limit
    assert model.sub_shift and all(np.isfinite(list(model.sub_shift.values())))


def test_attack_rates_and_fallbacks() -> None:
    f = frames()
    info = InformationSet.at(deadlines(f)[-1], SilverStore.from_frames(f))
    rates = fit_attack(info)
    assert len(rates.players) > 0
    assert (rates.players["goal_rate"] >= 0).all() and (rates.players["assist_rate"] >= 0).all()
    assert 0 <= rates.league["penalty_share"] <= 1 and 0 <= rates.league["own_goal_share"] <= 1
    out = rates.rates_for(pd.Series(["us:alpha:1", "nobody"]), pd.Series(["MID", "FWD"]))
    assert out["goal_rate"].notna().all() and len(out) == 2
