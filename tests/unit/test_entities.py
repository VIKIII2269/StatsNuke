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


FIRST = ["Bernardo", "Joseph", "Gabriel"]
SECOND = ["Mota Veiga de Carvalho e Silva", "Gomez", "Fernando de Jesus"]
WEB = ["Bernardo", "Gomez", "G.Jesus"]


def _apps(
    season: str, team: str, key: str, ids: list[int], fixtures: list[list[int]], minutes: int = 90
) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"season": season, "team": team, key: i, "fixture_uid": f"f{f}", "minutes": minutes}
            for i, fx in zip(ids, fixtures, strict=True)
            for f in fx
        ]
    )


def _frames(
    us_names: list[str], us_fixtures: list[list[int]], fpl_fixtures: list[list[int]]
) -> tuple[pd.DataFrame, ...]:
    n = len(fpl_fixtures)
    fa = _apps("2025-26", "t", "code", list(range(1, n + 1)), fpl_fixtures)
    ua = _apps(
        "2025-26", "t", "understat_player_id", list(range(100, 100 + len(us_names))), us_fixtures
    )
    ua["player_name"] = ua["understat_player_id"].map(
        dict(zip(range(100, 100 + len(us_names)), us_names, strict=True))
    )
    fp = pd.DataFrame(
        {
            "season": "2025-26",
            "code": range(1, n + 1),
            "first_name": FIRST[:n],
            "second_name": SECOND[:n],
            "web_name": WEB[:n],
            "position": "MID",
        }
    )
    return fa, ua, fp


def test_links_by_name_and_appearances() -> None:
    fx = [list(range(30)), list(range(20)), list(range(10))]
    fa, ua, fp = _frames(["Bernardo Silva", "Joe Gomez", "Gabriel Jesus"], fx, fx)
    r = link_players(fa, ua, fp)
    assert dict(zip(r.links["code"], r.links["understat_player_id"], strict=True)) == {
        1: 100,
        2: 101,
        3: 102,
    }
    assert r.coverage["share"].iloc[0] == 1.0


def test_substitute_minute_conventions_do_not_block_links() -> None:
    """Real 2016/17 case: sources count sub minutes differently (272 vs 254 over a
    season); the fixtures played still agree, so the link must hold."""
    fx = [list(range(15))]
    fa, ua, fp = _frames(["Bernardo Silva"], fx, fx)
    fa["minutes"], ua["minutes"] = 18, 17
    assert len(link_players(fa, ua, fp).links) == 1


def test_disjoint_appearances_block_a_name_match() -> None:
    fa, ua, fp = _frames(["Bernardo Silva"], [list(range(20, 30))], [list(range(0, 10))])
    r = link_players(fa, ua, fp)
    assert r.links.empty
    assert r.coverage["share"].iloc[0] == 0.0
    assert len(r.review) == 1


def test_weaker_name_needs_near_perfect_overlap() -> None:
    fx = [list(range(12))]
    fa, ua, fp = _frames(["Bernardo Sylva Junior"], fx, fx)  # name score in 80–92
    assert set(link_players(fa, ua, fp).links["method"]) <= {"name+appearances", "name"}
    fa2, ua2, fp2 = _frames(["Bernardo Sylva Junior"], [list(range(12))], [list(range(9))])
    assert link_players(fa2, ua2, fp2).links.empty  # overlap 0.75: not enough


def test_one_to_one_assignment() -> None:
    fx = [list(range(20)), list(range(20))]
    fa, ua, fp = _frames(["Bernardo Silva"], fx[:1], fx)
    fp.loc[1, ["first_name", "second_name", "web_name"]] = ["Bernardo", "Silva", "B.Silva"]
    assert len(link_players(fa, ua, fp).links) == 1


def test_cross_season_evidence_and_overrides() -> None:
    fx = [list(range(20))]
    fa, ua, fp = _frames(["Bernardo Silva"], fx, fx)
    fa2 = pd.concat([fa, fa.assign(season="2026-27")])
    ua2 = pd.concat([ua, ua.assign(season="2026-27", player_name="B. Silvestre")])  # renamed later
    fp2 = pd.concat([fp, fp.assign(season="2026-27")])
    r = link_players(fa2, ua2, fp2)
    assert set(r.links["season"]) == {"2025-26"}  # renamed row is below the review bar
    r2 = link_players(
        fa2, ua2, fp2, overrides=[{"code": 1, "understat_player_id": 100, "reason": "t"}]
    )
    assert set(r2.links["method"]) == {"override"} and len(r2.links) == 2
    blocked = link_players(
        fa2, ua2, fp2, overrides=[{"code": 1, "understat_player_id": None, "reason": "t"}]
    )
    assert blocked.links.empty
    dim = build_dim_player(fp2, r.links)
    assert dim.iloc[0]["player_uid"] == "fpl:1" and dim.iloc[0]["understat_player_id"] == 100


def test_name_variants() -> None:
    assert "g jesus" in fpl_name_variants("Gabriel", "Fernando de Jesus", "G.Jesus")
