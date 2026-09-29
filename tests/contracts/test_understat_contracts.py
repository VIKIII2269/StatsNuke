from __future__ import annotations

import copy
import json

import pytest
from pydantic import ValidationError

from fplh.collectors.understat_schema import LeagueData, MatchData
from tests.conftest import SAMPLES


def load(name: str) -> dict:  # type: ignore[type-arg]
    return json.loads((SAMPLES / "understat" / name).read_text())


def test_league_contract() -> None:
    data = LeagueData.model_validate(load("league.json"))
    assert data.teamsData["87"].history[0].ppda.def_ == 25


def test_match_contract() -> None:
    data = MatchData.model_validate(load("match.json"))
    assert len(data.shotsData["h"]) == 2


def test_renamed_field_fails() -> None:
    bad = copy.deepcopy(load("match.json"))
    for shot in bad["shotsData"]["h"]:
        shot["expectedGoals"] = shot.pop("xG")
    with pytest.raises(ValidationError, match="xG"):
        MatchData.model_validate(bad)
