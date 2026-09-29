from __future__ import annotations

from pathlib import Path

import pytest

from fplh.lake.parquet import sha256_of
from fplh.lake.quality import QualityGateError
from fplh.lake.silver.build import build
from fplh.lake.storage import Lake
from tests.lake_fixtures import build_mini_lake, football_data_csv, merged_gw_rows


def test_build_links_every_source(lake: Lake) -> None:
    build_mini_lake(lake)
    report, t = build(lake)
    assert all(g.passed for g in report.gates), report.summary()
    assert report.rows["fact_player_match"] == 8
    assert report.rows["fact_shot"] == 9
    dim = t["dim_fixture"]
    assert dim[["in_fpl", "in_football_data", "in_understat"]].all().all()
    assert set(dim["fixture_uid"]) == {
        "2025-26:liverpool:bournemouth",
        "2025-26:bournemouth:liverpool",
    }
    # every Understat roster row resolves to the FPL player
    assert t["fact_player_match_understat"]["player_uid"].notna().all()
    assert t["entity_coverage"]["share"].min() == 1.0
    assert t["dim_player"]["understat_player_id"].notna().all()
    # xP never reaches silver (§5.3.1)
    assert "xP" not in t["fact_player_match"].columns


def test_rebuild_is_byte_identical(tmp_path: Path) -> None:
    lake = Lake(str(tmp_path / "lake"))
    build_mini_lake(lake)
    first, _ = build(lake)
    hashes = {k: sha256_of(lake, k) for k in first.written}
    second, _ = build(lake)
    assert {k: sha256_of(lake, k) for k in second.written} == hashes


def test_observed_at_follows_publication_lags(lake: Lake) -> None:
    build_mini_lake(lake)
    _, t = build(lake, write=False)
    pm = t["fact_player_match"]
    assert ((pm["observed_at"] - pm["event_at"]).dt.total_seconds() == 33 * 3600).all()
    odds = t["snap_odds"]
    pre = odds[~odds["is_closing"]]
    assert (pre["observed_at"] < pre["kickoff_at"]).all()
    assert (
        odds.loc[odds["is_closing"], "observed_at"] == odds.loc[odds["is_closing"], "kickoff_at"]
    ).all()


def test_score_disagreement_blocks_build(lake: Lake) -> None:
    bad = football_data_csv().replace(b"Liverpool,Bournemouth,2,1", b"Liverpool,Bournemouth,3,1")
    build_mini_lake(lake, fd_csv=bad)
    with pytest.raises(QualityGateError, match="cross_source_scores"):
        build(lake)
    assert not lake.list("silver")  # nothing written when a gate fails


def test_goal_conservation_blocks_build(lake: Lake) -> None:
    rows = merged_gw_rows()
    rows[0]["goals_scored"] = 1  # Salah's two goals become one
    build_mini_lake(lake, gw_rows=rows)
    with pytest.raises(QualityGateError, match="goal_conservation"):
        build(lake)
