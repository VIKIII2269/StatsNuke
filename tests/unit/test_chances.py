from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fplh.features.information_set import SilverStore
from fplh.models import chances
from fplh.models.chances import (
    ASSIST_POINTS,
    AssistModel,
    adjust,
    appearances,
    attach,
    player_states,
    team_states,
)

T0 = pd.Timestamp("2025-08-16 14:00", tz="UTC")


def frames() -> dict[str, pd.DataFrame]:
    fixtures, us, pm = [], [], []
    for k in range(4):  # a v b at home, then away, alternating; one week apart
        home, away = ("a", "b") if k % 2 == 0 else ("b", "a")
        fid = f"2025-26:{home}:{away}:{k}"
        ko = T0 + pd.Timedelta(days=7 * k)
        fixtures.append(
            {
                "fixture_uid": fid,
                "season": "2025-26",
                "home_team": home,
                "away_team": away,
                "kickoff_at": ko,
                "round": k + 1,
            }
        )
        if k < 3:  # the last fixture is the one being forecast
            for side, team in (("h", home), ("a", away)):
                uid = f"fpl:{team}1"
                us.append(
                    {
                        "player_uid": uid,
                        "fixture_uid": fid,
                        "side": side,
                        "minutes": 90,
                        "xg": 0.1 * (k + 1),
                        "xa": 0.2 * (k + 1),
                        "shots": 2,
                        "key_passes": 1,
                        "xg_chain": 0.3,
                        "started": True,
                    }
                )
                pm.append(
                    {
                        "player_uid": uid,
                        "fixture_uid": fid,
                        "team": team,
                        "kickoff_at": ko,
                        "assists": 1 if k == 1 else 0,
                        "total_points": 3,
                    }
                )
    return {
        "dim_fixture": pd.DataFrame(fixtures),
        "fact_player_match_understat": pd.DataFrame(us),
        "fact_player_match": pd.DataFrame(pm),
        "snap_fpl_player": pd.DataFrame(columns=["code", "team", "observed_at"]),
    }


def test_features_use_only_matches_strictly_before_the_forecast_time() -> None:
    store = SilverStore.from_frames(frames())
    u = appearances(store)
    second = T0 + pd.Timedelta(days=7)
    rows = pd.DataFrame(
        {"player_uid": ["fpl:a1"], "team": ["a"], "opp": ["b"], "home": [0.0], "at": [second]}
    )
    x = attach(rows, player_states(u), team_states(u))
    assert x["xa90_3"].iloc[0] == pytest.approx(0.2)  # only the first match (xA 0.2 in 90')
    assert x["team_xg_6"].iloc[0] == pytest.approx(0.1)
    late = rows.assign(at=T0 + pd.Timedelta(days=15))
    x = attach(late, player_states(u), team_states(u))
    assert x["xa90_3"].iloc[0] == pytest.approx((0.2 + 0.4 + 0.6) / 3)
    assert x["mins_3"].iloc[0] == pytest.approx(90)


class Constant(AssistModel):
    def rate(self, x: pd.DataFrame) -> np.ndarray:
        return np.full(len(x), 0.4)


def test_adjust_blends_assists_into_expected_points() -> None:
    store = SilverStore.from_frames(frames())
    fid = frames()["dim_fixture"]["fixture_uid"].iloc[3]
    pred = pd.DataFrame(
        {
            "player_uid": ["fpl:a1", "fpl:nobody"],
            "fixture_uid": [fid, fid],
            "expected_points": [4.0, 2.0],
            "e_assists": [0.1, 0.05],
            "p_play": [0.5, 0.9],
        }
    )
    out = adjust(pred, store, T0 + pd.Timedelta(days=20), model=Constant(None, 1.5), weight=0.75)  # type: ignore[arg-type]
    a = out.iloc[0]
    assert a["assists_chance"] == pytest.approx(0.4 * 0.5 * 1.5)
    assert a["e_assists"] == pytest.approx(0.25 * 0.1 + 0.75 * 0.3)
    assert a["expected_points"] == pytest.approx(4.0 + ASSIST_POINTS * 0.75 * (0.3 - 0.1))
    b = out.iloc[1]  # no team known: v2 unchanged
    assert b["expected_points"] == pytest.approx(2.0) and b["e_assists"] == pytest.approx(0.05)


def test_fit_learns_xa_and_the_fpl_scale(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(chances, "ROUNDS", 5)
    store = SilverStore.from_frames(frames())
    model = AssistModel.fit(store, T0 + pd.Timedelta(days=20))
    assert model.scale == pytest.approx(2 / (2 * (0.2 + 0.4 + 0.6)))  # 2 FPL assists / 2.4 xA
    x = attach(
        pd.DataFrame(
            {
                "player_uid": ["fpl:a1"],
                "team": ["a"],
                "opp": ["b"],
                "home": [0.0],
                "at": [T0 + pd.Timedelta(days=20)],
            }
        ),
        player_states(appearances(store)),
        team_states(appearances(store)),
    )
    assert np.isfinite(model.rate(x)).all()


def test_attach_accepts_mixed_timestamp_resolutions() -> None:
    store = SilverStore.from_frames(frames())
    u = appearances(store)
    at = pd.Series([T0 + pd.Timedelta(days=20)]).astype("datetime64[us, UTC]")
    rows = pd.DataFrame(
        {"player_uid": ["fpl:a1"], "team": ["a"], "opp": ["b"], "home": [0.0], "at": at}
    )
    x = attach(rows, player_states(u), team_states(u))
    assert x["xa90_3"].notna().all()
