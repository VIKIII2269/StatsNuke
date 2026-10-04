from __future__ import annotations

import numpy as np
import pandas as pd

from fplh.features.information_set import InformationSet, SilverStore
from fplh.features.leakage import check_leakage, perturb_future
from fplh.features.minutes import minutes_features
from fplh.features.spine import SPINE_KEYS, build_spine
from fplh.features.transfers import available_at, fpl_round_transfers
from fplh.models.minutes import MinutesModel, MinutesPredictor
from tests.synthetic_silver import deadlines, make, with_understat

DIM = pd.DataFrame(
    {
        "fixture_uid": ["f1", "f2", "f3"],
        "season": ["2030-31"] * 3,
        "round": pd.array([1, 2, 2], dtype="Int64"),
        "kickoff_at": pd.to_datetime(
            ["2030-08-10 14:00", "2030-08-17 11:30", "2030-08-17 14:00"], utc=True
        ),
    }
)


def pm(rows: list[tuple[int, str, int, int, int]]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "season": "2030-31",
                "fixture_uid": f"f{r}",
                "player_uid": p,
                "transfers_in": i,
                "transfers_out": o,
                "selected": s,
                "kickoff_at": DIM["kickoff_at"].iloc[r - 1],
                "observed_at": DIM["kickoff_at"].iloc[r - 1] + pd.Timedelta(days=2),
            }
            for r, p, i, o, s in rows
        ]
    )


def test_round_transfers_are_public_at_their_own_deadline() -> None:
    t = fpl_round_transfers(pm([(1, "a", 0, 0, 1000), (2, "a", 500, 9000, 2000)]), DIM)
    r2 = t[t["round"] == 2].iloc[0]
    assert r2["observed_at"] == pd.Timestamp("2030-08-17 10:00", tz="UTC")  # 11:30 − 90 min
    # owners before = 2000 − 500 + 9000 = 10500; selling share = 9000 / (10500 + 1000)
    assert np.isclose(r2["tr_sell"], 9000 / 11500)
    assert np.isclose(r2["tr_buy"], np.log1p(500 / 11500))
    assert np.isnan(t[t["round"] == 1]["tr_sell"].iloc[0])  # no transfers before GW1


def test_information_set_serves_a_round_only_from_its_deadline() -> None:
    store = SilverStore.from_frames(
        {"fact_player_match": pm([(2, "a", 1, 2, 3000)]), "dim_fixture": DIM}
    )
    d = pd.Timestamp("2030-08-17 10:00", tz="UTC")
    assert InformationSet.at(d - pd.Timedelta(minutes=1), store).table("fpl_round_transfers").empty
    seen = InformationSet.at(d, store)
    assert len(seen.table("fpl_round_transfers")) == 1
    seen.assert_no_leakage()


def test_live_snapshots_stand_in_until_the_history_row_exists() -> None:
    snap = pd.DataFrame(
        {
            "season": ["2030-31"] * 2,
            "code": [7, 7],
            "transfers_in_event": [100, 400],
            "transfers_out_event": [2000, 8000],
            "selected_by_percent": [10.0, 5.0],
            "total_players": [100_000, 100_000],
            "observed_at": pd.to_datetime(["2030-08-15 12:00", "2030-08-17 09:00"], utc=True),
        }
    )
    t = fpl_round_transfers(pd.DataFrame(), DIM, snap)
    assert (t["round"] == 2).all()  # the window open at capture time is round 2's
    late = t.sort_values("observed_at").iloc[-1]
    # 5 % of 100k own now; 5000 − 400 + 8000 owned before the window
    assert np.isclose(late["tr_sell"], 8000 / (12600 + 1000))
    both = fpl_round_transfers(pm([(2, "fpl:7", 1, 2, 3000)]), DIM, snap)
    deadline = pd.Timestamp("2030-08-17 10:00", tz="UTC")
    latest = both[both["observed_at"] <= deadline].sort_values("observed_at").iloc[-1]
    assert latest["observed_at"] == deadline  # the history row supersedes the captures


def test_pre_deadline_columns_are_perturbed_by_when_they_became_public() -> None:
    f = make()
    d = deadlines(f)[2]
    out = perturb_future(f, d)["fact_player_match"]
    when = available_at(f["fact_player_match"], f["dim_fixture"])
    public = when <= d
    pd.testing.assert_series_equal(
        out.loc[public, "transfers_out"], f["fact_player_match"].loc[public, "transfers_out"]
    )
    later = f["fact_player_match"].loc[~public, "transfers_out"]
    assert (out.loc[~public, "transfers_out"] != later).any()


def news(info: InformationSet, spine: pd.DataFrame) -> pd.DataFrame:
    return pd.concat([spine[SPINE_KEYS], minutes_features(info, spine, news=True)], axis=1)


def test_news_features_are_leak_free() -> None:
    f = with_understat(make())
    assert check_leakage(f, deadlines(f), {"news": news}, horizon=2) == []


def test_news_features_and_model() -> None:
    f = with_understat(make())
    store = SilverStore.from_frames(f)
    ds = deadlines(f)
    info = InformationSet.at(ds[3], store)
    spine = build_spine(info, 1)
    x = minutes_features(info, spine, news=True)
    for c in ("tr_sell", "tr_buy", "tr_own", "tr_age", "depth_news"):
        assert c in x
    assert x["tr_sell"].notna().any() and x["tr_sell"].dropna().between(0, 1).all()
    assert x["tr_age"].dropna().ge(0).all()
    assert "tr_sell" not in minutes_features(info, spine)  # v1 features unchanged
    model = MinutesModel.fit(info, rounds=20, news=True)
    p = model.predict(info, spine)
    assert p["p_start"].between(0, 1).all()
    assert MinutesPredictor(news=True).name == "minutes_news"
