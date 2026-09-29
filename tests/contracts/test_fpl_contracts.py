from __future__ import annotations

import copy

import pytest
from pydantic import ValidationError

from fplh.collectors.fpl_schema import Bootstrap, ElementSummary, EventLive, Fixture
from fplh.rules import Rules
from fplh.rules.golden import check, history_from_fpl
from tests.conftest import load_sample


def test_bootstrap_contract() -> None:
    boot = Bootstrap.model_validate(load_sample("bootstrap-static.json"))
    assert {t.singular_name_short for t in boot.element_types} == {"GKP", "DEF", "MID", "FWD"}


def test_fixtures_contract() -> None:
    fixtures = [Fixture.model_validate(f) for f in load_sample("fixtures.json")]
    assert any(f.event is None for f in fixtures)  # unscheduled (postponed) fixture


def test_element_summary_contract() -> None:
    ElementSummary.model_validate(load_sample("element-summary.json"))


def test_event_live_contract() -> None:
    EventLive.model_validate(load_sample("event-live.json"))


def test_contract_catches_renamed_field() -> None:
    boot = copy.deepcopy(load_sample("bootstrap-static.json"))
    boot["elements"][0]["chance_of_playing_next"] = boot["elements"][0].pop(
        "chance_of_playing_next_round"
    )
    with pytest.raises(ValidationError, match="chance_of_playing_next_round"):
        Bootstrap.model_validate(boot)


def test_extra_fields_are_tolerated() -> None:
    boot = copy.deepcopy(load_sample("bootstrap-static.json"))
    boot["elements"][0]["brand_new_field"] = 1
    Bootstrap.model_validate(boot)


def test_history_from_fpl_payloads_scores_exactly(rules_2627: Rules) -> None:
    golden = history_from_fpl(
        "2026/27", load_sample("bootstrap-static.json"), [load_sample("element-summary.json")]
    )
    assert golden.frame["position"].tolist() == ["GK"]  # GKP normalised
    report = check(golden, rules_2627, bonus_from_bps=False)  # one player, not a full fixture
    assert report.ok, report.summary()
