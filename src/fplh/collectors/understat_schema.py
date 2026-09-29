"""Contracts for the Understat JSON we depend on (values are mostly strings)."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class _Open(BaseModel):
    model_config = ConfigDict(extra="allow")


class Side(_Open):
    id: str
    title: str
    short_title: str


class HA(_Open):
    h: str | None
    a: str | None


class DateEntry(_Open):
    id: str
    isResult: bool
    h: Side
    a: Side
    goals: HA
    xG: HA
    datetime: str


class Ppda(_Open):
    model_config = ConfigDict(extra="allow", populate_by_name=True)

    att: float
    def_: float = Field(alias="def")  # "def" is a Python keyword


class TeamMatch(_Open):
    h_a: str
    xG: float
    xGA: float
    npxG: float
    npxGA: float
    ppda: Ppda
    ppda_allowed: Ppda
    deep: int
    deep_allowed: int
    scored: int
    missed: int
    date: str


class TeamSeason(_Open):
    id: str
    title: str
    history: list[TeamMatch]


class PlayerSeason(_Open):
    id: str
    player_name: str
    games: str
    time: str
    goals: str
    xG: str
    assists: str
    xA: str
    shots: str
    key_passes: str
    position: str
    team_title: str
    npg: str
    npxG: str


class LeagueData(_Open):
    datesData: list[DateEntry]
    teamsData: dict[str, TeamSeason]
    playersData: list[PlayerSeason]


class Shot(_Open):
    id: str
    minute: str
    result: str
    X: str
    Y: str
    xG: str
    player: str
    h_a: str
    player_id: str
    situation: str
    shotType: str
    match_id: str
    date: str
    player_assisted: str | None
    lastAction: str | None


class RosterEntry(_Open):
    id: str
    goals: str
    own_goals: str
    shots: str
    xG: str
    time: str
    player_id: str
    team_id: str
    position: str
    player: str
    h_a: str
    yellow_card: str
    red_card: str
    roster_in: str
    roster_out: str
    key_passes: str
    assists: str
    xA: str


class MatchData(_Open):
    match_info: dict[str, object]
    rostersData: dict[str, dict[str, RosterEntry]]
    shotsData: dict[str, list[Shot]]
