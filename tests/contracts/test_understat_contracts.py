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
    assert data.dates[0].isResult
    assert all(len(t.history) for t in data.teams.values())


def test_match_contract() -> None:
    data = MatchData.model_validate(load("match.json"))
    assert set(data.shots) == {"h", "a"}
    assert all(s.h_a == "h" for s in data.shots["h"])


def test_renamed_field_fails() -> None:
    bad = copy.deepcopy(load("match.json"))
    for shot in bad["shots"]["h"]:
        shot["expectedGoals"] = shot.pop("xG")
    with pytest.raises(ValidationError, match="xG"):
        MatchData.model_validate(bad)
