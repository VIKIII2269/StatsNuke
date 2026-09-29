"""Response contracts for the FPL API endpoints we depend on.

Models allow extra fields (FPL adds fields often) but require every field that
downstream code reads, with the expected type. Contract tests validate saved sample
payloads against these models, so a rename or removal upstream fails CI (§14).
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict


class _Open(BaseModel):
    model_config = ConfigDict(extra="allow")


class Event(_Open):
    id: int
    deadline_time: str
    finished: bool
    data_checked: bool
    is_current: bool
    is_next: bool


class Team(_Open):
    id: int
    code: int
    name: str
    short_name: str


class ElementType(_Open):
    id: int
    singular_name_short: str


class Element(_Open):
    id: int
    code: int
    web_name: str
    first_name: str
    second_name: str
    element_type: int
    team: int
    now_cost: int
    status: str
    chance_of_playing_next_round: int | None
    chance_of_playing_this_round: int | None
    news: str
    news_added: str | None
    selected_by_percent: str
    ep_next: str | None
    ep_this: str | None
    transfers_in_event: int
    transfers_out_event: int


class Bootstrap(_Open):
    events: list[Event]
    teams: list[Team]
    element_types: list[ElementType]
    elements: list[Element]


class Fixture(_Open):
    id: int
    code: int
    event: int | None
    kickoff_time: str | None
    team_h: int
    team_a: int
    team_h_score: int | None
    team_a_score: int | None
    started: bool | None
    finished: bool
    team_h_difficulty: int
    team_a_difficulty: int
    stats: list[Any]


class HistoryRow(_Open):
    element: int
    fixture: int
    opponent_team: int
    round: int
    kickoff_time: str
    was_home: bool
    total_points: int
    minutes: int
    goals_scored: int
    assists: int
    clean_sheets: int
    goals_conceded: int
    own_goals: int
    penalties_saved: int
    penalties_missed: int
    yellow_cards: int
    red_cards: int
    saves: int
    bonus: int
    bps: int
    starts: int
    clearances_blocks_interceptions: int
    recoveries: int
    tackles: int
    defensive_contribution: int
    expected_goals: str
    expected_assists: str
    expected_goals_conceded: str
    value: int
    selected: int


class ElementSummary(_Open):
    fixtures: list[Any]
    history: list[HistoryRow]
    history_past: list[Any]


class LiveElement(_Open):
    id: int
    stats: dict[str, Any]
    explain: list[Any]


class EventLive(_Open):
    elements: list[LiveElement]
