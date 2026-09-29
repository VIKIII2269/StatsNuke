"""A0 baseline evaluation: walk-forward over each gameweek deadline, scored against
outcomes, logged to the leaderboard (ARCHITECTURE.md §11.7, row A0)."""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from fplh.evaluate import metrics as m
from fplh.evaluate.tracking import log_metrics
from fplh.evaluate.walk_forward import run_walk_forward
from fplh.features.information_set import SilverStore
from fplh.features.spine import historical_deadlines
from fplh.lake.storage import Lake
from fplh.models.baselines import A0Match, A0Player
from fplh.models.market import devig_multiplicative
from fplh.rules.config import load_rules


@dataclass
class SeasonResult:
    season: str
    player_metrics: dict[str, float] = field(default_factory=dict)
    match_metrics: dict[str, float] = field(default_factory=dict)
    run_ids: dict[str, str] = field(default_factory=dict)


def _closing_probs(store: SilverStore) -> pd.DataFrame:
    """De-vigged closing 1X2 per fixture (evaluation reference only, never a feature)."""
    odds = store.get("snap_odds")
    if odds.empty:
        return pd.DataFrame(columns=["fixture_uid", "c_home", "c_draw", "c_away", "c_book"])
    close = odds[odds["is_closing"] & (odds["market"] == "1x2")]
    rows = []
    for uid, g in close.groupby("fixture_uid"):
        for book in ("pinnacle", "market_avg", "bet365"):
            x = g[g["bookmaker"] == book].set_index("outcome")["price"]
            if {"home", "draw", "away"} <= set(x.index):
                p = devig_multiplicative(x[["home", "draw", "away"]].to_numpy())
                rows.append(
                    {
                        "fixture_uid": uid,
                        "c_home": p[0],
                        "c_draw": p[1],
                        "c_away": p[2],
                        "c_book": book,
                    }
                )
                break
    return pd.DataFrame(rows, columns=["fixture_uid", "c_home", "c_draw", "c_away", "c_book"])


def evaluate_season(lake: Lake, season: str, *, log: bool = True) -> SeasonResult:
    store = SilverStore(lake)
    rules = load_rules(season)
    deadlines = list(historical_deadlines(store.get("dim_fixture"), season)["deadline_at"])
    result = SeasonResult(season)

    # Player level
    wf = run_walk_forward(
        store,
        A0Player(rules),
        deadlines,
        rules_config=f"fpl_{season.replace('-', '_')}.yaml",
        lake=lake,
    )
    result.run_ids["player"] = wf.run_id
    actual = store.get("fact_player_match")[
        ["player_uid", "fixture_uid", "total_points", "minutes"]
    ]
    j = wf.predictions.merge(actual, on=["player_uid", "fixture_uid"], how="inner")
    played = j[j["minutes"] > 0]
    result.player_metrics = {
        "mae": m.mae(j["expected_points"], j["total_points"]),
        "rmse": m.rmse(j["expected_points"], j["total_points"]),
        "spearman_within_position": m.spearman_within(
            j["expected_points"], j["total_points"], j["position"]
        ),
        # most rows are unused squad players (0 points, easy to predict); report both
        "mae_played": m.mae(played["expected_points"], played["total_points"]),
        "rmse_played": m.rmse(played["expected_points"], played["total_points"]),
        "spearman_within_position_played": m.spearman_within(
            played["expected_points"], played["total_points"], played["position"]
        ),
        "n_played": float(len(played)),
        "mean_predicted": float(j["expected_points"].mean()),
        "mean_actual": float(j["total_points"].mean()),
        "n": float(len(j)),
        "share_market_available": float(j["market_available"].astype(bool).mean()),
        "n_spine_rows_without_outcome": float(len(wf.predictions) - len(j)),
    }

    # Match level (needs pre-match odds)
    mf = run_walk_forward(store, A0Match(), deadlines, unit="fixture", lake=lake)
    result.run_ids["match"] = mf.run_id
    dim = store.get("dim_fixture")[["fixture_uid", "home_goals", "away_goals"]].dropna()
    mj = mf.predictions.merge(dim, on="fixture_uid").dropna(subset=["p_home"])
    if len(mj):
        y = m.outcome_index(mj["home_goals"].astype(int), mj["away_goals"].astype(int))
        p = mj[["p_home", "p_draw", "p_away"]].to_numpy()
        result.match_metrics = {
            "log_loss": m.log_loss(p, y),
            "rps": m.rps(p, y),
            "n": float(len(mj)),
        }
        cj = mj.merge(_closing_probs(store), on="fixture_uid")
        if len(cj):
            yc = m.outcome_index(cj["home_goals"].astype(int), cj["away_goals"].astype(int))
            cp = cj[["c_home", "c_draw", "c_away"]].to_numpy()
            result.match_metrics |= {
                "closing_log_loss": m.log_loss(cp, yc),
                "closing_rps": m.rps(cp, yc),
                "closing_n": float(len(cj)),
            }
    else:
        result.match_metrics = {"n": 0.0}

    if log:
        log_metrics(
            lake,
            result.run_ids["player"],
            result.player_metrics,
            scope=season,
            tags={"model": "A0", "level": "player"},
        )
        log_metrics(
            lake,
            result.run_ids["match"],
            result.match_metrics,
            scope=season,
            tags={"model": "A0", "level": "match"},
        )
    return result
