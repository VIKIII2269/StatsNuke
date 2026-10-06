from __future__ import annotations

import numpy as np
import pandas as pd

from fplh.models.stack import FEATURES, StackConfig, stack_frame, walk_forward_stack


def base(n_players: int = 60, n_rounds: int = 12, seed: int = 0) -> tuple[pd.DataFrame, ...]:
    rng = np.random.default_rng(seed)
    t0 = pd.Timestamp("2030-08-01", tz="UTC")
    rows, ys = [], []
    for g in range(n_rounds):
        d = t0 + pd.Timedelta(days=7 * g)
        for p in range(n_players):
            skill = (p % 10) / 3
            rows.append(
                {
                    "player_uid": f"fpl:{p}",
                    "fixture_uid": f"f{g}:{p // 2}",
                    "deadline_at": d,
                    "position": ("GK", "DEF", "MID", "FWD")[p % 4],
                    "expected_points": skill + rng.normal(0, 0.3),
                    "sd_points": 2.0,
                    "p_start": 0.8,
                    "p_60": 0.7,
                    "e_goals": 0.1,
                    "e_assists": 0.1,
                    "e_bonus": 0.2,
                    "p_clean_sheet": 0.3,
                }
            )
            ys.append(
                {
                    "player_uid": f"fpl:{p}",
                    "fixture_uid": f"f{g}:{p // 2}",
                    "total_points": skill + rng.normal(0, 1),
                    "observed_at": d + pd.Timedelta(days=3),
                    "value": 50 + p,
                }
            )
    sim = pd.DataFrame(rows)
    rep = sim[["player_uid", "fixture_uid", "deadline_at"]].assign(
        expected_points=sim["expected_points"] + rng.normal(0, 0.3, len(sim))
    )
    y = pd.DataFrame(ys)
    crowd = pd.DataFrame(
        {
            "player_uid": [f"fpl:{p}" for p in range(n_players)],
            "tr_sell": 0.05,
            "tr_buy": 0.05,
            "tr_own": 0.1,
            "observed_at": t0 - pd.Timedelta(days=1),
        }
    )
    return sim, rep, y, crowd


def test_frame_has_every_feature() -> None:
    sim, rep, y, crowd = base()
    f = stack_frame(sim, rep, crowd, y[["player_uid", "value", "observed_at"]])
    assert set(FEATURES) <= set(f.columns)
    assert len(f) == len(sim)
    # the price is the latest observed at the deadline: none before the first outcome
    first = f[f["deadline_at"] == f["deadline_at"].min()]
    assert first["price"].isna().all()


def test_stack_is_the_blend_without_enough_history() -> None:
    sim, rep, y, crowd = base()
    f = stack_frame(sim, rep, crowd, y[["player_uid", "value", "observed_at"]])
    ds = sorted(f["deadline_at"].unique())
    out = walk_forward_stack(f, y, ds, StackConfig(min_rows=10**9, blend=0.5))
    blend = 0.5 * f["sim_ep"] + 0.5 * f["rep_ep"]
    np.testing.assert_allclose(
        out.sort_values(["deadline_at", "player_uid"])["expected_points"].to_numpy(),
        f.sort_values(["deadline_at", "player_uid"]).pipe(lambda d: blend[d.index]).to_numpy(),
    )


def test_stack_never_trains_on_unobserved_outcomes() -> None:
    sim, rep, y, crowd = base()
    f = stack_frame(sim, rep, crowd, y[["player_uid", "value", "observed_at"]])
    ds = sorted(f["deadline_at"].unique())
    cfg = StackConfig(rounds=20, refit_every=1, min_rows=50)
    a = walk_forward_stack(f, y, ds, cfg)
    # scrambling every outcome observed after a deadline leaves that deadline's forecasts
    d = ds[6]
    y2 = y.copy()
    later = y2["observed_at"] > d
    y2.loc[later, "total_points"] = y2.loc[later, "total_points"].to_numpy()[::-1] + 5
    b = walk_forward_stack(f, y2, ds, cfg)
    pd.testing.assert_frame_equal(
        a[a["deadline_at"] <= d].reset_index(drop=True),
        b[b["deadline_at"] <= d].reset_index(drop=True),
    )
    assert not np.allclose(
        a[a["deadline_at"] > d]["expected_points"], b[b["deadline_at"] > d]["expected_points"]
    )
