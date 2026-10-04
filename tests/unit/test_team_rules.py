"""Phase 4 game state: gameweek scoring with auto-subs, sell prices, chip windows."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fplh.features.information_set import SilverStore
from fplh.features.prices import price_table, sell_price, sell_prices
from fplh.rules.config import Chips, load_rules
from fplh.rules.team import score_gameweek

XI_MIN = {"GK": 1, "DEF": 3, "MID": 2, "FWD": 1}
# 1-11: GK, 4 DEF, 4 MID, 2 FWD; bench: GK2, DEF5, MID5, FWD3
PICKS = ["g1", "d1", "d2", "d3", "d4", "m1", "m2", "m3", "m4", "f1", "f2", "g2", "d5", "m5", "f3"]
POS = {p: {"g": "GK", "d": "DEF", "m": "MID", "f": "FWD"}[p[0]] for p in PICKS}


def score(
    points: dict[str, int],
    minutes: dict[str, int],
    chip: str | None = None,
    captain: str = "m1",
    vice: str = "f1",
) -> tuple[int, tuple[tuple[str, str], ...]]:
    pts = {p: points.get(p, 2) for p in PICKS}
    mins = {p: minutes.get(p, 90) for p in PICKS}
    r = score_gameweek(PICKS, POS, pts, mins, captain, vice, chip, XI_MIN)
    return r.points, r.substitutions


def test_everyone_plays_no_substitutions() -> None:
    pts, subs = score({}, {})
    assert pts == 11 * 2 + 2 and subs == ()  # captain m1 doubled


def test_first_eligible_bench_player_replaces_a_non_playing_starter() -> None:
    pts, subs = score({"m2": 0, "d5": 6}, {"m2": 0})
    assert subs == (("m2", "d5"),) and pts == 10 * 2 + 6 + 2


def test_bench_player_who_did_not_play_is_skipped() -> None:
    _, subs = score({}, {"m2": 0, "d5": 0})
    assert subs == (("m2", "m5"),)


def test_formation_minimums_are_kept() -> None:
    # 3-5-2 starting XI: a defender who misses out can only be replaced by a defender
    picks = [
        "g1",
        "d1",
        "d2",
        "d3",
        "m1",
        "m2",
        "m3",
        "m4",
        "m5",
        "f1",
        "f2",
        "g2",
        "m6",
        "f3",
        "d4",
    ]
    pos = {p: {"g": "GK", "d": "DEF", "m": "MID", "f": "FWD"}[p[0]] for p in picks}
    mins = {p: 90 for p in picks} | {"d1": 0}
    r = score_gameweek(picks, pos, {p: 1 for p in picks}, mins, "m1", "f1", None, XI_MIN)
    assert r.substitutions == (("d1", "d4"),)
    mins["d4"] = 0  # no defender available: d1 stays out, the XI plays with ten
    pts = {p: 1 for p in picks} | {"d1": 0}
    r = score_gameweek(picks, pos, pts, mins, "m1", "f1", None, XI_MIN)
    assert r.substitutions == () and r.points == 10 + 1


def test_goalkeeper_only_for_goalkeeper() -> None:
    _, subs = score({}, {"g1": 0})
    assert subs == (("g1", "g2"),)
    _, subs = score({}, {"g1": 0, "g2": 0})
    assert subs == ()


def test_vice_captain_takes_over_and_triple_captain() -> None:
    pts, _ = score({"m1": 10, "f1": 5}, {"m1": 0}, captain="m1", vice="f1")
    # m1 replaced by m5 (2 points), vice f1 doubled
    assert pts == 9 * 2 + 5 + 2 + 5
    pts, _ = score({"m1": 10}, {}, chip="triple_captain")
    assert pts == 10 * 2 + 10 * 3
    pts, _ = score({"m1": 10}, {"m1": 0, "f1": 0}, captain="m1", vice="f1")
    assert pts == 9 * 2 + 2 + 2  # nobody doubled; m5 and f3 come on


def test_bench_boost_counts_all_fifteen_without_substitutions() -> None:
    pts, subs = score({"g2": 3}, {"m2": 0}, chip="bench_boost")
    assert subs == () and pts == 13 * 2 + 3 + 2 + 2  # m2 (0 min) still scores 2 here


def test_invalid_inputs_are_rejected() -> None:
    with pytest.raises(ValueError, match="15 distinct"):
        score_gameweek(PICKS[:14], POS, {}, {}, "m1", "f1", None, XI_MIN)
    with pytest.raises(ValueError, match="captain"):
        score_gameweek(PICKS, POS, {}, {}, "m1", "m1", None, XI_MIN)
    with pytest.raises(ValueError, match="captain"):
        score_gameweek(PICKS, POS, {}, {}, "m5", "f1", None, XI_MIN)  # captain on the bench
    with pytest.raises(ValueError, match="unknown chip"):
        score_gameweek(PICKS, POS, {}, {}, "m1", "f1", "assistant", XI_MIN)


@pytest.mark.parametrize(
    ("buy", "now", "sell"), [(50, 50, 50), (50, 51, 50), (50, 52, 51), (50, 55, 52), (50, 48, 48)]
)
def test_sell_price(buy: int, now: int, sell: int) -> None:
    assert sell_price(buy, now) == sell
    assert sell_prices(np.array([buy]), np.array([now]))[0] == sell


def test_price_table_carries_prices_over_blanks() -> None:
    pm = pd.DataFrame(
        {
            "season": "2030-31",
            "player_uid": ["a", "a", "b"],
            "round": [1, 3, 2],
            "value": [50, 52, 60],
            "kickoff_at": pd.to_datetime(["2030-08-10", "2030-08-24", "2030-08-17"], utc=True),
        }
    )
    t = price_table(SilverStore.from_frames({"fact_player_match": pm}), "2030-31", last_gw=4)
    assert t.loc["a"].tolist() == [50, 50, 52, 52]
    assert np.isnan(t.loc["b", 1]) and t.loc["b"].tolist()[1:] == [60, 60, 60]


def test_chip_windows_per_season() -> None:
    g = load_rules("2023/24").game
    assert g.chips.allowance()["wildcard"] == [(2, 20), (21, 38)]
    assert g.chips.allowance()["free_hit"] == [(2, 38)] and g.free_transfer_bank_max == 2
    g = load_rules("2025/26").game
    assert g.chips.allowance()["bench_boost"] == [(1, 19), (20, 38)]
    assert g.special_free_transfers == {16: 5} and g.free_transfer_bank_max == 5
    assert load_rules("2022/23").game.special_free_transfers == {17: 15}
    with pytest.raises(ValueError, match="either"):
        Chips(
            sets=2, per_set=["wildcard"], first_set_deadline_gw=19, windows={"wildcard": [(1, 38)]}
        )
    with pytest.raises(ValueError, match="bad gameweek window"):
        Chips(windows={"wildcard": [(0, 38)]})
