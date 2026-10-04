"""Season replay (ticket 4.2) on a synthetic league: valid squads, budget, scoring and
determinism."""

from __future__ import annotations

import numpy as np
import pandas as pd

from fplh.evaluate.replay import (
    SeasonData,
    compare_strategies,
    expected_points,
    replay_season,
    season_totals,
)
from fplh.rules.config import load_rules
from fplh.rules.team import score_gameweek

TEAMS = [f"c{i}" for i in range(20)]
GWS = range(1, 39)


def league(seed: int = 0, last: int = 7) -> tuple[SeasonData, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    players = []
    for t in TEAMS:
        for pos, n in (("GK", 1), ("DEF", 2), ("MID", 2), ("FWD", 1)):
            for j in range(n):
                players.append((f"{t}-{pos}{j}", pos, t))
    uid = [p[0] for p in players]
    position = pd.Series({p[0]: p[1] for p in players})
    team_of = {p[0]: p[2] for p in players}
    base = {"GK": 45, "DEF": 45, "MID": 60, "FWD": 70}
    price = pd.DataFrame(
        {g: [base[p[1]] + int(rng.integers(0, 40)) for p in players] for g in GWS}, index=uid
    ).astype(float)
    fixtures = {}
    rows = []
    for g in GWS:
        for i in range(0, 20, 2):
            h, a = TEAMS[i], TEAMS[(i + g) % 20 if (i + g) % 20 != i else (i + 1) % 20]
            fx = f"2022-23:{h}:{a}:{g}"
            fixtures[fx] = g
    counts = pd.DataFrame(0, index=TEAMS, columns=list(GWS))
    for fx, g in fixtures.items():
        _, h, a, _ = fx.split(":")
        counts.loc[h, g] += 1
        counts.loc[a, g] += 1
    skill = rng.uniform(1, 6, len(uid))
    pts = pd.DataFrame({g: rng.poisson(skill).astype(int) for g in GWS}, index=uid)
    mins = pd.DataFrame({g: np.where(rng.random(len(uid)) < 0.85, 90, 0) for g in GWS}, index=uid)
    pts = pts.where(mins > 0, 0)
    deadlines = {
        g: pd.Timestamp("2022-08-01", tz="UTC") + pd.Timedelta(days=7 * g)
        for g in range(2, last + 1)
    }
    team_at = pd.DataFrame(
        [(u, g, team_of[u]) for u in uid for g in GWS], columns=["player_uid", "gw", "team"]
    )
    data = SeasonData("2022-23", deadlines, fixtures, counts, price, team_at, position, pts, mins)
    for fx, g in fixtures.items():
        if g not in deadlines:
            continue
        _, h, a, _ = fx.split(":")
        for u in uid:
            if team_of[u] in (h, a):
                rows.append(
                    {
                        "player_uid": u,
                        "fixture_uid": fx,
                        "deadline_at": deadlines[g],
                        "expected_points": skill[uid.index(u)] + rng.normal(0, 0.3),
                    }
                )
    return data, pd.DataFrame(rows)


def test_replay_produces_valid_teams_and_scores() -> None:
    data, pred = league()
    res = replay_season("toy", pred, "repeat", data, horizon=3, time_limit=20)
    log = res.log
    assert list(log["gw"]) == list(range(2, 8))
    assert (log["bank"] >= 0).all()
    assert (log["squad_value"] + log["bank"] <= 1000 + 40 * 15).all()
    rules = load_rules("2022/23")
    for _, r in log.iterrows():
        squad = r["squad"].split(",")
        assert len(squad) == 15
        pos = data.position[squad].value_counts().to_dict()
        assert pos == {"GK": 2, "DEF": 5, "MID": 5, "FWD": 3}
        teams = pd.Series([u.split("-")[0] for u in squad]).value_counts()
        assert teams.max() <= rules.game.max_per_club
        assert r["points"] == r["gross"] - 4 * r["hits"]
    assert res.total == int(log["points"].sum())
    # the first squad is free: no hits at GW2
    assert log.iloc[0]["hits"] == 0


def test_replay_is_deterministic_and_comparable() -> None:
    data, pred = league(1)
    a = replay_season("a", pred, "repeat", data, horizon=2, time_limit=20)
    b = replay_season("a", pred, "repeat", data, horizon=2, time_limit=20)
    pd.testing.assert_frame_equal(a.log, b.log)
    noisy = pred.assign(
        expected_points=pred["expected_points"].sample(frac=1, random_state=0).to_numpy()
    )
    c = replay_season("noise", noisy, "repeat", data, horizon=2, time_limit=20)
    logs = pd.concat([a.log, c.log])
    cmp = compare_strategies(logs, "a", "noise", n_boot=200)
    assert cmp["gameweeks"] == 6 and np.isfinite(cmp["per_gw"])
    assert set(season_totals(logs).index) == {"a", "noise"}


def test_repeat_and_native_forecasts() -> None:
    data, pred = league(2)
    e = expected_points(pred, data, 3, [3, 4, 5], "repeat")
    u = e.index[0]
    team = u.split("-")[0]
    per_fixture = pred[(pred["player_uid"] == u) & (pred["deadline_at"] == data.deadlines[3])]
    per_fixture = per_fixture[per_fixture["fixture_uid"].map(data.gw_of_fixture) == 3][
        "expected_points"
    ].mean()
    for g in (3, 4, 5):
        assert np.isclose(e.loc[u, f"E{g}"], per_fixture * data.fixtures_per_team.loc[team, g])
    native = expected_points(pred, data, 3, [3, 4, 5], "native")
    assert (native[["E4", "E5"]] == 0).all().all()  # these forecasts only cover next week
    assert np.isclose(native.loc[u, "E3"], per_fixture * data.fixtures_per_team.loc[team, 3])


def test_scoring_matches_the_rules_engine() -> None:
    data, pred = league(3)
    res = replay_season("toy", pred, "repeat", data, horizon=2, time_limit=20)
    xi_min = {"GK": 1, "DEF": 3, "MID": 2, "FWD": 1}
    for _, r in res.log.iterrows():
        picks = r["xi"].split(",") + r["bench"].split(",")
        g = int(r["gw"])
        again = score_gameweek(
            picks,
            data.position[picks].to_dict(),
            data.points[g][picks].to_dict(),
            data.minutes[g][picks].to_dict(),
            r["captain"],
            r["vice"],
            r["chip"] if isinstance(r["chip"], str) else None,
            xi_min,
        )
        assert again.points == r["gross"]
        assert data.position[r["bench"].split(",")[0]] == "GK"  # substitute keeper first
