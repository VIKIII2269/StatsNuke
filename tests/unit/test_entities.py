from __future__ import annotations

import pandas as pd
import pytest

from fplh.entities.fixtures import build_dim_fixture
from fplh.entities.players import build_dim_player, fpl_name_variants, link_players
from fplh.entities.teams import TeamResolver, UnknownTeamError, normalise_name, slug


def test_team_resolution() -> None:
    t = TeamResolver.from_config()
    assert t.uid("Man Utd") == t.uid("Manchester United") == t.uid("Man United") == "man-united"
    assert t.uid("Nott'm Forest") == t.uid("Nottingham Forest") == "nottm-forest"
    assert t.uid("Tottenham Hotspur") == t.uid("Spurs") == "tottenham"
    with pytest.raises(UnknownTeamError, match="closest"):
        t.uid("Real Madrid")
    t.add_canonical(["Real Madrid"])
    assert t.uid("Real Madrid") == "real-madrid"
    assert normalise_name("Martin Ødegaard") == "martin odegaard"
    assert normalise_name("Pierre-Emile Højbjerg") == "pierre emile hojbjerg"
    assert normalise_name("Łukasz Fabiański") == "lukasz fabianski"
    assert slug("Nott'm Forest") == "nottm-forest"


def test_dim_fixture_prefers_fpl_and_flags_sources() -> None:
    k = pd.Timestamp("2025-08-15 19:00", tz="UTC")
    fpl = pd.DataFrame(
        {
            "season": ["2025-26"],
            "home_team": ["liverpool"],
            "away_team": ["bournemouth"],
            "kickoff_at": [k],
            "round": [1],
            "fpl_fixture_id": [1],
            "home_goals": [4],
            "away_goals": [2],
        }
    )
    fd = pd.DataFrame(
        {
            "season": ["2025-26", "2025-26"],
            "division": ["E0", "E0"],
            "home_team": ["liverpool", "arsenal"],
            "away_team": ["bournemouth", "chelsea"],
            "kickoff_at": [k + pd.Timedelta(minutes=5), k],
            "home_goals": [4, 1],
            "away_goals": [2, 1],
        }
    )
    us = pd.DataFrame(
        {
            "season": ["2025-26"],
            "home_team": ["liverpool"],
            "away_team": ["bournemouth"],
            "kickoff_at": [k],
            "understat_match_id": [26602],
            "home_goals": [4],
            "away_goals": [2],
        }
    )
    dim = build_dim_fixture(fpl, fd, us).set_index("fixture_uid")
    row = dim.loc["2025-26:liverpool:bournemouth"]
    assert (
        row["kickoff_at"] == k and row["fpl_fixture_id"] == 1 and row["understat_match_id"] == 26602
    )
    assert row[["in_fpl", "in_football_data", "in_understat"]].all()
    assert not dim.loc["2025-26:arsenal:chelsea", "in_fpl"]


def _frames(
    us_names: list[str], us_minutes: list[int], fpl_minutes: list[int]
) -> tuple[pd.DataFrame, ...]:
    n = len(fpl_minutes)
    fm = pd.DataFrame(
        {"season": "2025-26", "team": "t", "code": range(1, n + 1), "minutes": fpl_minutes}
    )
    um = pd.DataFrame(
        {
            "season": "2025-26",
            "team": "t",
            "understat_player_id": range(100, 100 + len(us_names)),
            "player_name": us_names,
            "minutes": us_minutes,
        }
    )
    fp = pd.DataFrame(
        {
            "season": "2025-26",
            "code": range(1, n + 1),
            "first_name": ["Bernardo", "Joseph", "Gabriel"][:n],
            "second_name": ["Mota Veiga de Carvalho e Silva", "Gomez", "Fernando de Jesus"][:n],
            "web_name": ["Bernardo", "Gomez", "G.Jesus"][:n],
            "position": "MID",
        }
    )
    return fm, um, fp


def test_links_by_name_and_minutes() -> None:
    fm, um, fp = _frames(
        ["Bernardo Silva", "Joe Gomez", "Gabriel Jesus"], [2500, 1800, 900], [2500, 1800, 900]
    )
    r = link_players(fm, um, fp)
    assert dict(zip(r.links["code"], r.links["understat_player_id"], strict=True)) == {
        1: 100,
        2: 101,
        3: 102,
    }
    assert r.coverage["share"].iloc[0] == 1.0


def test_minutes_disagreement_blocks_a_name_match() -> None:
    fm, um, fp = _frames(["Bernardo Silva"], [300], [2500])
    r = link_players(fm.head(1), um, fp.head(1))
    assert r.links.empty
    assert r.coverage["share"].iloc[0] == 0.0


def test_one_to_one_assignment() -> None:
    fm, um, fp = _frames(["Bernardo Silva"], [2500], [2500, 2500])
    fp.loc[1, ["first_name", "second_name", "web_name"]] = ["Bernardo", "Silva", "B.Silva"]
    r = link_players(fm.head(2), um, fp.head(2))
    assert len(r.links) == 1


def test_cross_season_evidence_and_overrides() -> None:
    fm, um, fp = _frames(["Bernardo Silva"], [2500], [2500])
    fm2 = pd.concat([fm.head(1), fm.head(1).assign(season="2026-27")])
    um2 = pd.concat([um, um.assign(season="2026-27", player_name="B. Silvestre")])  # renamed later
    fp2 = pd.concat([fp.head(1), fp.head(1).assign(season="2026-27")])
    r = link_players(fm2, um2, fp2)
    assert set(r.links["season"]) == {"2025-26"}  # renamed row is below the review bar
    r2 = link_players(
        fm2, um2, fp2, overrides=[{"code": 1, "understat_player_id": 100, "reason": "t"}]
    )
    assert set(r2.links["method"]) == {"override"} and len(r2.links) == 2
    blocked = link_players(
        fm2, um2, fp2, overrides=[{"code": 1, "understat_player_id": None, "reason": "t"}]
    )
    assert blocked.links.empty
    dim = build_dim_player(fp2, r.links)
    assert dim.iloc[0]["player_uid"] == "fpl:1" and dim.iloc[0]["understat_player_id"] == 100


def test_name_variants() -> None:
    assert "g jesus" in fpl_name_variants("Gabriel", "Fernando de Jesus", "G.Jesus")
