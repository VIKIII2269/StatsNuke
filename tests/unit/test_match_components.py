"""M7–M10 recover known parameters from synthetic leagues (models/defence, gk, cards, bonus)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from fplh.features.information_set import InformationSet, SilverStore
from fplh.models.bonus import DESIGN, bps_design, fit_bonus
from fplh.models.cards import fit_cards
from fplh.models.defence import ACTIONS, fit_defence
from fplh.models.gk import fit_saves, negbin_pmf

TEAMS = [f"t{i}" for i in range(10)]
SQUAD = ["GK", "DEF", "DEF", "DEF", "DEF", "MID", "MID", "MID", "MID", "FWD", "FWD"]
YELLOW = {"GK": 0.05, "DEF": 0.18, "MID": 0.2, "FWD": 0.14}
BPS = {"minutes_gte_60": 6.0, "goal_FWD": 24.0, "goal_MID": 18.0, "assist": 9.0, "save": 2.0}


def league(seasons: tuple[str, ...] = ("2016-17",), seed: int = 0) -> dict[str, pd.DataFrame]:
    """Double round robins; known yellow, save, defensive and BPS processes."""
    rng = np.random.default_rng(seed)
    players = [(f"fpl:{t}{k}", t, pos) for t in TEAMS for k, pos in enumerate(SQUAD)]
    effect = {uid: rng.normal(0, 2.0) for uid, _, _ in players}  # true BPS δ_p
    fixtures, rows, prematch = [], [], []
    t0 = pd.Timestamp("2016-08-13 14:00", tz="UTC")
    day = 0
    for season in seasons:
        for i, (h, a) in enumerate((h, a) for h in TEAMS for a in TEAMS if h != a):
            kickoff = t0 + pd.Timedelta(days=day + 3 * (i // 5))
            uid = f"{season}:{h}:{a}"
            mu = {h: rng.uniform(0.8, 2.2), a: rng.uniform(0.6, 1.8)}
            fixtures.append(
                {
                    "fixture_uid": uid,
                    "season": season,
                    "home_team": h,
                    "away_team": a,
                    "kickoff_at": kickoff,
                    "round": i // 5 + 1,
                }
            )
            prematch.append({"fixture_uid": uid, "pred_home": mu[h], "pred_away": mu[a]})
            for p_uid, team, pos in players:
                if team not in (h, a):
                    continue
                opp = a if team == h else h
                mins = 90
                saves = rng.poisson(rng.gamma(20, (1.2 + 1.2 * mu[opp]) / 20)) if pos == "GK" else 0
                goals = rng.poisson({"FWD": 0.4, "MID": 0.15}.get(pos, 0.0))
                assists = rng.poisson(0.1)
                scale = 2.0 if season >= "2025-26" else 1.0  # tackles drift by ×2
                acts = {
                    "clearances_blocks_interceptions": rng.poisson(5 if pos == "DEF" else 1.5),
                    "tackles": rng.poisson(scale * (1.0 if pos != "FWD" else 0.4)),
                    "recoveries": rng.poisson(4.0),
                }
                ev = {
                    "minutes": mins,
                    "goals_scored": goals,
                    "assists": assists,
                    "goals_conceded": 0,
                    "saves": saves,
                    "penalties_saved": 0,
                    "penalties_missed": 0,
                    "yellow_cards": 0,
                    "red_cards": 0,
                    "own_goals": 0,
                }
                x = bps_design({k: np.array([v]) for k, v in ev.items()}, np.array([pos]))
                bps = sum(BPS.get(c, 0.0) * float(x[c][0]) for c in DESIGN)
                rows.append(
                    {
                        **ev,
                        **acts,
                        "season": season,
                        "player_uid": p_uid,
                        "fixture_uid": uid,
                        "position": pos,
                        "team": team,
                        "opponent": opp,
                        "was_home": team == h,
                        "kickoff_at": kickoff,
                        "yellow_cards": int(rng.random() < 1 - np.exp(-YELLOW[pos])),
                        "penalties_missed": int(rng.random() < 0.02),
                        "bps": round(bps + 3.0 + effect[p_uid] + rng.normal(0, 3.0)),
                        "bonus": 0,
                        "observed_at": kickoff + pd.Timedelta(hours=12),
                        "event_at": kickoff,
                    }
                )
        day += 400
    pm = pd.DataFrame(rows)
    pm["penalties_saved"] = pm["penalties_missed"] * (pm.index % 4 != 0)
    dim = pd.DataFrame(fixtures)
    frames = {"dim_fixture": dim, "fact_player_match": pm}
    return frames | {"_prematch": pd.DataFrame(prematch), "_effect": pd.Series(effect)}


def info_after(frames: dict[str, pd.DataFrame]) -> InformationSet:
    store = SilverStore.from_frames({k: v for k, v in frames.items() if not k.startswith("_")})
    end = frames["fact_player_match"]["observed_at"].max() + pd.Timedelta(days=1)
    return InformationSet.at(end, store)


def test_cards_recover_position_rates() -> None:
    rates = fit_cards(info_after(league()))
    for pos, rate in YELLOW.items():
        assert abs(rates.position_rate[pos] - rate) < 0.04
    unseen = rates.rates_for(pd.Series(["nobody"]), pd.Series(["DEF"]))
    assert unseen[0] == rates.position_rate["DEF"]


def test_saves_recover_the_opponent_slope() -> None:
    f = league(("2016-17", "2017-18", "2018-19"))
    model = fit_saves(info_after(f), f["_prematch"])
    for mu in (0.8, 1.4, 2.0):  # mean saves per 90 against an opponent expecting μ
        assert abs(model.mean(mu, np.array([90]), 1.0)[0] - (1.2 + 1.2 * mu)) < 0.2
    assert abs(model.penalty_save_share - 0.75) < 0.15
    pmf = negbin_pmf(np.array([3.0]), model.size, 30)
    assert abs(pmf.sum() - 1) < 1e-9


def test_defence_detects_and_rescales_definition_drift() -> None:
    model = fit_defence(info_after(league(("2016-17", "2025-26"))))
    assert set(model.drift) >= {"DEF:tackles", "MID:tackles"}
    assert abs(model.drift["DEF:tackles"] - 2.0) < 0.3
    assert "DEF:clearances_blocks_interceptions" not in model.drift
    # after rescaling, the old era's tackles count at the new definition
    assert abs(model.position_rate["DEF"]["tackles"] - 2.0) < 0.2
    assert set(model.size) >= {"DEF", "MID", "FWD"}
    p = model.threshold_prob(
        pd.Series(["fpl:t01"]),
        pd.Series(["DEF"]),
        pd.Series(["t1"]),
        np.array([90]),
        np.array([10]),
    )
    assert 0 < p[0] < 1
    assert len(ACTIONS) == 3


def test_bonus_recovers_weights_and_player_effects() -> None:
    f = league(("2016-17", "2017-18"))
    model = fit_bonus(info_after(f))
    assert abs(model.weights["goal_FWD"] - 24) < 2 and abs(model.weights["save"] - 2) < 0.5
    truth = f["_effect"]
    est = model.players.reindex(truth.index)
    assert np.corrcoef(est.to_numpy(), truth.to_numpy())[0, 1] > 0.8
    assert all(2.0 < s < 4.5 for s in model.sigma.values())
