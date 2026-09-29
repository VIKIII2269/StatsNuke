"""CI-blocking leakage tests (ARCHITECTURE.md §6.2, §12.7)."""

from __future__ import annotations

import pandas as pd
import pytest

from fplh.features.builders import BUILDERS, build_features
from fplh.features.information_set import InformationSet, LeakageError, SilverStore
from fplh.features.leakage import check_leakage, perturb_future
from fplh.features.spine import build_spine
from tests.synthetic_silver import deadlines, make


def test_every_builder_is_clean_at_every_deadline() -> None:
    frames = make()
    assert check_leakage(frames, deadlines(frames)) == []


def test_a_leaky_builder_is_caught() -> None:
    frames = make()

    def bypasses_the_information_set(info: InformationSet, spine: pd.DataFrame) -> pd.DataFrame:
        # Reaches past the filter into the underlying store: sees the whole season.
        pm = info._store.get("fact_player_match")
        totals = pm.groupby("player_uid")["goals_scored"].sum().rename("leaky_goals")
        keys = spine[["player_uid", "fixture_uid", "deadline_at", "horizon"]]
        return keys.merge(totals, left_on="player_uid", right_index=True, how="left")

    problems = check_leakage(
        frames, deadlines(frames)[1:3], {"leaky": bypasses_the_information_set}
    )
    assert problems and all("depend on future rows" in p for p in problems)


def test_dimension_outcomes_are_hidden() -> None:
    info = InformationSet.at(deadlines(make())[2], SilverStore.from_frames(make()))
    assert "home_goals" not in info.table("dim_fixture").columns


def test_future_rows_are_invisible() -> None:
    frames = make()
    d = deadlines(frames)[3]
    info = InformationSet.at(d, SilverStore.from_frames(frames))
    pm = info.table("fact_player_match")
    assert (pm["observed_at"] <= d).all()
    assert info.max_observed_at is not None and info.max_observed_at <= d
    shuffled = perturb_future(frames, d)
    changed = shuffled["fact_player_match"]["observed_at"] > d
    assert changed.any()


def test_unexposed_tables_raise() -> None:
    info = InformationSet.at(pd.Timestamp("2030-09-01", tz="UTC"), SilverStore.from_frames(make()))
    with pytest.raises(LeakageError):
        info.table("fpl_player_season")  # end-of-season registrations: not point-in-time


def test_spine_and_snapshot_features() -> None:
    frames = make()
    d = deadlines(frames)[2]
    info = InformationSet.at(d, SilverStore.from_frames(frames))
    spine = build_spine(info, horizon=2)
    assert set(spine["horizon"]) == {1, 2}
    assert (spine["kickoff_at"] > d).all()
    feats = build_features(info, spine)
    assert feats["chance_of_playing_next_round"].notna().all()  # snapshots exist before d
    before_capture = InformationSet.at(
        frames["snap_fpl_player"]["observed_at"].min() - pd.Timedelta(days=1),
        SilverStore.from_frames(frames),
    )
    early = build_features(before_capture, build_spine(before_capture, 1))
    assert early.empty or early["chance_of_playing_next_round"].isna().all()  # NULL, not imputed


def test_sql_sees_only_the_information_set() -> None:
    frames = make()
    d = deadlines(frames)[1]
    info = InformationSet.at(d, SilverStore.from_frames(frames))
    latest = info.sql("SELECT max(observed_at) AS m FROM fact_player_match")["m"].iloc[0]
    assert pd.Timestamp(latest) <= d
    assert set(BUILDERS) == {"naive_minutes", "season_totals", "snapshot_status"}
