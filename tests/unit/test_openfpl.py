from __future__ import annotations

import numpy as np
import pandas as pd

from fplh.evaluate.player_level import align, outcomes, player_losses, summary, top_k_precision
from fplh.evaluate.walk_forward import output_digest, run_walk_forward
from fplh.features.information_set import InformationSet, SilverStore
from fplh.features.leakage import check_leakage
from fplh.features.openfpl import openfpl_features
from fplh.features.spine import SPINE_KEYS
from fplh.models.baselines import NaiveLast5
from fplh.models.openfpl import OpenFPLReplica, training_rows
from tests.synthetic_silver import SEASON, deadlines, make, with_understat


def frames() -> dict[str, pd.DataFrame]:
    f = with_understat(make())
    tm = f["us_match"]
    team_rows = pd.concat(
        [
            tm.assign(
                team=tm["home_team"],
                xg=tm["home_xg"],
                xga=tm["away_xg"],
                scored=tm["home_goals"],
                missed=tm["away_goals"],
            ),
            tm.assign(
                team=tm["away_team"],
                xg=tm["away_xg"],
                xga=tm["home_xg"],
                scored=tm["away_goals"],
                missed=tm["home_goals"],
            ),
        ]
    )
    return {**f, "us_team_match": team_rows}


def keyed(info: InformationSet, spine: pd.DataFrame) -> pd.DataFrame:
    return pd.concat([spine[SPINE_KEYS], openfpl_features(info, spine)], axis=1)


def test_openfpl_features_are_leak_free() -> None:
    f = frames()
    assert check_leakage(f, deadlines(f), {"openfpl": keyed}, horizon=1) == []


def test_training_features_equal_features_at_each_deadline() -> None:
    f = frames()
    store = SilverStore.from_frames(f)
    info = InformationSet.at(deadlines(f)[-1] + pd.Timedelta(days=30), store)
    rows = training_rows(info)
    assert len(rows) > 0 and rows["deadline_at"].nunique() > 2
    batch = openfpl_features(info, rows)
    for d, idx in rows.groupby("deadline_at").groups.items():
        one = openfpl_features(info.restrict(pd.Timestamp(d)), rows.loc[idx])
        pd.testing.assert_frame_equal(batch.loc[idx], one, check_dtype=False)


def test_replica_and_naive_walk_forward() -> None:
    f = frames()
    store = SilverStore.from_frames(f)
    ds = deadlines(f)
    replica = run_walk_forward(store, OpenFPLReplica(refit_every=2), ds, write=False)
    again = run_walk_forward(store, OpenFPLReplica(refit_every=2), ds, write=False)
    assert output_digest(replica) == output_digest(again)
    naive = run_walk_forward(store, NaiveLast5(), ds, write=False)
    for res in (replica, naive):
        p = res.predictions
        assert p["expected_points"].notna().all()
        assert pd.Timestamp(res.manifest.max_observed_at) <= max(ds)
    # every synthetic player scores 2 points per fixture: the naive mean is exactly 2 once
    # one of the player's fixtures is observable, else 0 (squads come from snapshots)
    p = naive.predictions
    pm = f["fact_player_match"]
    seen = [
        bool(((pm["player_uid"] == u) & (pm["observed_at"] <= d)).any())
        for u, d in zip(p["player_uid"], p["deadline_at"], strict=True)
    ]
    assert np.allclose(p["expected_points"], np.where(seen, 2.0, 0.0))


def test_player_scoring_and_alignment() -> None:
    f = frames()
    store = SilverStore.from_frames(f)
    ds = deadlines(f)
    naive = run_walk_forward(store, NaiveLast5(), ds, write=False).predictions
    actual = outcomes(store)
    a = player_losses(naive, actual)
    b = player_losses(naive.iloc[: len(naive) // 2], actual)
    aligned = align({"a": a, "b": b})
    assert len(aligned["a"]) == len(aligned["b"]) == len(b)
    s = summary(a[a["expected_points"] > 0])
    assert s["mse"] == 0.0
    assert (a["block"].str.startswith(SEASON)).all()


def test_top_k_precision_counts_ties() -> None:
    g = pd.DataFrame(
        {
            "block": "s:1",
            "position": "MID",
            "expected_points": [5.0, 4.0, 3.0, 2.0, 1.0],
            "total_points": [10, 1, 7, 7, 0],
        }
    )
    # top-2 predicted = first two rows; actual top 2 = {10, 7, 7} → 1 of 2 hit
    assert top_k_precision(g, k=2) == 0.5
