from __future__ import annotations

import pandas as pd

from fplh.lake.quality import (
    check_entity_coverage,
    check_kickoff_agreement,
    check_odds_coverage,
    check_player_plausibility,
    check_understat_shots,
    run_gates,
)
from fplh.lake.silver.build import build
from fplh.lake.storage import Lake
from tests.lake_fixtures import build_mini_lake


def tables(lake: Lake) -> dict[str, pd.DataFrame]:
    build_mini_lake(lake)
    _, t = build(lake, write=False)
    return t


def test_clean_mini_lake_passes_every_gate(lake: Lake) -> None:
    results = run_gates(tables(lake))
    assert all(r.passed for r in results), [r.line() for r in results]


def test_too_many_starters(lake: Lake) -> None:
    t = tables(lake)
    extra = t["fact_player_match"].head(1)
    t["fact_player_match"] = pd.concat(
        [t["fact_player_match"]] + [extra.assign(element=100 + i) for i in range(11)]
    )
    assert not check_player_plausibility(t).passed


def test_shot_conservation(lake: Lake) -> None:
    t = tables(lake)
    t["fact_shot"] = t["fact_shot"].iloc[1:]
    r = check_understat_shots(t)
    assert not r.passed and "1 of 4" in r.details


def test_kickoff_disagreement_warns_only(lake: Lake) -> None:
    t = tables(lake)
    t["us_match"] = t["us_match"].assign(
        kickoff_at=t["us_match"]["kickoff_at"] + pd.Timedelta(days=3)
    )
    r = check_kickoff_agreement(t)
    assert not r.passed and not r.blocking


def test_entity_coverage_gate(lake: Lake) -> None:
    t = tables(lake)
    t["entity_coverage"] = t["entity_coverage"].assign(share=0.99)
    assert not check_entity_coverage(t).passed


def test_odds_coverage_gate(lake: Lake) -> None:
    t = tables(lake)
    t["snap_odds"] = t["snap_odds"][t["snap_odds"]["home_team"] != "liverpool"]
    r = check_odds_coverage(t)
    assert not r.passed and "1 of 2" in r.details
