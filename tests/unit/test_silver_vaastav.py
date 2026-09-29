from __future__ import annotations

import pandas as pd
import pytest

from fplh.entities.teams import TeamResolver, UnknownTeamError
from fplh.lake.silver.vaastav import normalise_season
from tests.lake_fixtures import PLAYERS, TEAM_NAMES, merged_gw_rows

RAW = pd.DataFrame(
    [
        {
            "id": p[0],
            "code": p[1],
            "first_name": p[2],
            "second_name": p[3],
            "web_name": p[4],
            "element_type": p[5],
            "team": p[6],
        }
        for p in PLAYERS
    ]
)
MASTER = pd.DataFrame(
    [{"season": "2025-26", "team": t, "team_name": n} for t, n in TEAM_NAMES.items()]
)


def run(gw: pd.DataFrame, **kw: object):  # type: ignore[no-untyped-def]
    return normalise_season(2025, gw, RAW, MASTER, TeamResolver.from_config(), **kw)  # type: ignore[arg-type]


def test_teams_come_from_the_fixture_not_players_raw() -> None:
    gw = pd.DataFrame(merged_gw_rows()).drop(columns=["team", "position"])  # pre-2020/21 shape
    raw_moved = RAW.assign(team=[2, 2, 1, 1])  # end-of-season teams all wrong
    out = normalise_season(2025, gw, raw_moved, MASTER, TeamResolver.from_config())
    salah = out.player_match[out.player_match["element"] == 1]
    assert set(salah["team"]) == {"liverpool"}
    assert set(out.player_match["position"]) == {"GK", "DEF", "MID"}
    assert "xP" not in out.player_match.columns
    fx = out.fixtures.set_index("fpl_fixture_id")
    assert (fx.loc[1, "home_team"], fx.loc[1, "away_team"]) == ("liverpool", "bournemouth")


def test_postponed_placeholder_and_manager_rows_dropped() -> None:
    rows = merged_gw_rows()
    placeholder = {
        **rows[0],
        "round": 5,
        "GW": 5,
        "minutes": 0,
        "goals_scored": 0,
        "team_h_score": None,
        "team_a_score": None,
    }
    manager = {**rows[0], "element": 99, "position": "AM"}
    gkp = {**rows[3]}
    rows[3] = {**gkp, "position": "GKP"}
    out = run(pd.DataFrame([*rows, placeholder, manager]))
    assert out.notes["postponed_placeholders_dropped"] == 1
    assert out.notes["manager_rows_dropped"] == 1
    assert len(out.player_match) == len(rows)
    assert "GKP" not in set(out.player_match["position"])


def test_conflicting_duplicates_raise() -> None:
    rows = merged_gw_rows()
    rows.append({**rows[0], "total_points": 99})
    with pytest.raises(ValueError, match="conflicting duplicate"):
        run(pd.DataFrame(rows))


def test_unknown_team_names_fail_with_suggestions() -> None:
    teams = pd.DataFrame(
        [{"id": 1, "name": "Liverpool FC Women"}, {"id": 2, "name": "Bournemouth"}]
    )
    with pytest.raises(UnknownTeamError, match="closest"):
        run(pd.DataFrame(merged_gw_rows()), season_teams=teams)
