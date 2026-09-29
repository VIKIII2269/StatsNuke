from __future__ import annotations

from pathlib import Path

import pandas as pd

from fplh.evaluate.manifest import Manifest
from fplh.evaluate.tracking import leaderboard, log_metrics
from fplh.evaluate.walk_forward import output_digest, run_walk_forward
from fplh.features.information_set import SilverStore
from fplh.lake.storage import Lake
from fplh.models.baselines import A0Match, A0Player
from fplh.rules import load_rules
from tests.synthetic_silver import deadlines, make


def _odds(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for f in frames["dim_fixture"].itertuples():
        for outcome, price in (("home", 2.0), ("draw", 3.4), ("away", 4.0)):
            rows.append(
                {
                    "fixture_uid": f.fixture_uid,
                    "bookmaker": "market_avg",
                    "market": "1x2",
                    "line": float("nan"),
                    "outcome": outcome,
                    "price": price,
                    "is_closing": False,
                    "kickoff_at": f.kickoff_at,
                    "observed_at": f.kickoff_at - pd.Timedelta(days=1),
                    "home_team": f.home_team,
                    "away_team": f.away_team,
                }
            )
    return pd.DataFrame(rows)


def test_player_walk_forward_replays_bit_identically(tmp_path: Path) -> None:
    frames = make()
    frames["snap_odds"] = _odds(frames)
    store = SilverStore.from_frames(frames)
    lake = Lake(str(tmp_path / "lake"))
    ds = deadlines(frames)
    first = run_walk_forward(
        store, A0Player(load_rules("2025/26")), ds, lake=lake, seeds={"sim": 1}
    )
    second = run_walk_forward(
        store, A0Player(load_rules("2025/26")), ds, lake=lake, seeds={"sim": 1}
    )
    assert first.run_id == second.run_id
    assert output_digest(first) == output_digest(second)
    assert first.manifest.max_observed_at is not None
    assert all(pd.Timestamp(first.manifest.max_observed_at) <= max(ds) for _ in [0])
    stored = Manifest.from_json(lake.get_bytes(first.written[1]).decode())
    assert stored.run_id == first.run_id
    preds = first.predictions
    assert {"expected_points", "p_60", "market_available"} <= set(preds.columns)
    assert preds["market_available"].all()
    assert (preds["expected_points"] >= -1).all()


def test_changing_inputs_changes_run_id() -> None:
    frames = make()
    store = SilverStore.from_frames(frames)
    ds = deadlines(frames)
    a = run_walk_forward(store, A0Player(load_rules("2025/26")), ds, write=False)
    b = run_walk_forward(store, A0Player(load_rules("2025/26")), ds[:-1], write=False)
    assert a.run_id != b.run_id


def test_fixture_walk_forward_devigs_market() -> None:
    frames = make()
    frames["snap_odds"] = _odds(frames)
    res = run_walk_forward(
        SilverStore.from_frames(frames), A0Match(), deadlines(frames), unit="fixture", write=False
    )
    p = res.predictions
    assert (p[["p_home", "p_draw", "p_away"]].sum(axis=1).round(9) == 1).all()
    assert (p["p_home"] > p["p_away"]).all()
    assert (p["lambda_home"] > p["lambda_away"]).all()


def test_leaderboard_upserts(tmp_path: Path) -> None:
    lake = Lake(str(tmp_path / "lake"))
    log_metrics(lake, "run1", {"mae": 1.0}, scope="2024-25")
    log_metrics(lake, "run1", {"mae": 0.9, "rmse": 2.0}, scope="2024-25")
    board = leaderboard(lake)
    assert len(board) == 2
    assert board.set_index("metric").loc["mae", "value"] == 0.9
