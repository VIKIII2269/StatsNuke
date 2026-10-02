"""Phase 2 team-level models in the walk-forward harness (ARCHITECTURE.md §7.2–7.5, §11).

* ``M1Predictor``: refits the M1 filter on everything observable at the deadline (the
  information set's ``us_match`` rows: goals + xG), forecasts the next rounds' expected
  goals (propagated to each kickoff) and turns them into scoreline grids with a goal
  model (G0–G3).
* ``MarketPredictor``: market-only rates, the latest pre-match prices at D de-vigged and
  inverted under the same goal model that draws the grid.
* ``FusedPredictor``: M3 fusion of M1 with those market rates.
* ``fixture_losses``: per-fixture 1X2 log loss, RPS and scoreline-grid log loss.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from fplh.evaluate import metrics as m
from fplh.features.information_set import InformationSet
from fplh.models.baselines import NOT_A_BOOK, market_probabilities
from fplh.models.fusion import Fusion
from fplh.models.goal_benchmarks import GoalModel, benchmarks, markets
from fplh.models.team_strength import TeamStrengthParams, championship_priors, run_filter

CAP = 6
GRID_COLS = [f"g_{h}_{a}" for h in range(CAP + 1) for a in range(CAP + 1)]
FIXTURE_KEYS = ["fixture_uid", "deadline_at", "horizon"]


def observed_matches(info: InformationSet) -> pd.DataFrame:
    us = info.table("us_match")
    if us.empty:
        return us
    played = us[us["is_result"].astype(bool)].copy()
    played["kickoff_at"] = played["event_at"]
    return played[
        [
            "season",
            "kickoff_at",
            "home_team",
            "away_team",
            "home_goals",
            "away_goals",
            "home_xg",
            "away_xg",
        ]
    ]


def season_teams(info: InformationSet) -> dict[str, set[str]]:
    """Teams per season from the published schedule (known before the season)."""
    fx = info.table("dim_fixture")
    out: dict[str, set[str]] = {}
    for season, g in fx.groupby("season"):
        out[str(season)] = set(g["home_team"]) | set(g["away_team"])
    return out


def next_season_label(season: str) -> str:
    start = int(season[:4]) + 1
    return f"{start}-{(start + 1) % 100:02d}"


def championship(info: InformationSet) -> dict[tuple[str, str], tuple[float, float]]:
    """Championship ratings observable at D, keyed by the EPL season they feed."""
    fd = info.table("fd_match")
    if fd.empty:
        return {}
    e1 = fd[fd["division"] == "E1"]
    seasons = {str(s): next_season_label(str(s)) for s in e1["season"].unique()}
    return championship_priors(e1, seasons)


def grid_frame(lh: np.ndarray, la: np.ndarray, model: GoalModel) -> pd.DataFrame:
    rows = []
    for x, y in zip(lh, la, strict=True):
        g = model.grid(float(x), float(y))
        mk = markets(g)
        rows.append(
            [mk["home"], mk["draw"], mk["away"], mk["over"], *g[: CAP + 1, : CAP + 1].ravel()]
        )
    return pd.DataFrame(rows, columns=["p_home", "p_draw", "p_away", "p_over25", *GRID_COLS])


@dataclass
class M1Predictor:
    params: TeamStrengthParams
    goal_model: GoalModel = field(default_factory=lambda: benchmarks()["G0"])
    name: str = "m1"
    version: str = "1"

    def rates(self, info: InformationSet, fixtures: pd.DataFrame) -> pd.DataFrame:
        matches = observed_matches(info)
        teams_by_season = season_teams(info)
        f = run_filter(
            matches,
            self.params,
            teams_by_season,
            teams=sorted({t for ts in teams_by_season.values() for t in ts}),
            championship=championship(info),
        )
        rows = []
        fx = fixtures.sort_values("kickoff_at")
        for uid, season, home, away, kickoff in zip(
            fx["fixture_uid"].astype(str),
            fx["season"].astype(str),
            fx["home_team"].astype(str),
            fx["away_team"].astype(str),
            fx["kickoff_at"],
            strict=True,
        ):
            if season != f.season:
                f.start_season(season, teams_by_season[season])
            lh, la = f.expected_goals(home, away, pd.Timestamp(kickoff))
            rows.append({"fixture_uid": uid, "lam_mod_home": lh, "lam_mod_away": la})
        return pd.DataFrame(rows)

    def predict_fixtures(self, info: InformationSet, fixtures: pd.DataFrame) -> pd.DataFrame:
        r = fixtures[[*FIXTURE_KEYS]].merge(self.rates(info, fixtures), on="fixture_uid")
        grid = grid_frame(
            r["lam_mod_home"].to_numpy(), r["lam_mod_away"].to_numpy(), self.goal_model
        )
        return pd.concat([r.reset_index(drop=True), grid], axis=1)


def market_rates(
    info: InformationSet,
    fixtures: pd.DataFrame,
    goal_model: GoalModel,
    method: str = "multiplicative",
) -> pd.DataFrame:
    """λ^mkt per fixture: the latest pre-match prices at D, de-vigged with ``method`` and
    inverted under ``goal_model`` so its grid reproduces the market's 1X2 (and O/U 2.5)."""
    mp = market_probabilities(info, fixtures[["fixture_uid"]], method)
    has = mp["market_available"].fillna(False).astype(bool)
    lh = mp["lambda_home"].to_numpy(float, copy=True)
    la = mp["lambda_away"].to_numpy(float, copy=True)
    if goal_model.name != "G0":  # independent Poisson: market_probabilities already inverted
        p = mp[["p_home", "p_draw", "p_away", "p_over25"]].to_numpy(float)
        for i in np.flatnonzero(has.to_numpy()):
            over = None if np.isnan(p[i, 3]) else float(p[i, 3])
            lh[i], la[i] = goal_model.invert(p[i, :3].tolist(), over)
    return pd.DataFrame(
        {
            "fixture_uid": mp["fixture_uid"],
            "lam_mkt_home": np.where(has, lh, np.nan),
            "lam_mkt_away": np.where(has, la, np.nan),
        }
    )


@dataclass
class MarketPredictor:
    goal_model: GoalModel = field(default_factory=lambda: benchmarks()["G0"])
    devig_method: str = "multiplicative"
    name: str = "market"
    version: str = "2"

    def predict_fixtures(self, info: InformationSet, fixtures: pd.DataFrame) -> pd.DataFrame:
        mr = market_rates(info, fixtures, self.goal_model, self.devig_method).dropna()
        r = fixtures[[*FIXTURE_KEYS]].merge(mr, on="fixture_uid")
        r = r.rename(columns={"lam_mkt_home": "lambda_home", "lam_mkt_away": "lambda_away"})
        grid = grid_frame(r["lambda_home"].to_numpy(), r["lambda_away"].to_numpy(), self.goal_model)
        return pd.concat([r.reset_index(drop=True), grid], axis=1)


def market_features(
    info: InformationSet,
    fixtures: pd.DataFrame,
    goal_model: GoalModel,
    method: str = "multiplicative",
) -> pd.DataFrame:
    """λ^mkt (``market_rates``), hours from the latest price to kickoff τ, and liquidity."""
    odds = info.table("snap_odds")
    base = fixtures[["fixture_uid", "kickoff_at"]].copy()
    if odds.empty:
        return base.assign(
            lam_mkt_home=np.nan, lam_mkt_away=np.nan, tau_hours=np.nan, liquidity=np.nan
        )
    pre = odds[~odds["is_closing"] & odds["fixture_uid"].isin(base["fixture_uid"])]
    pre = pre[~pre["bookmaker"].isin(NOT_A_BOOK)]
    stats = (
        pre[pre["market"] == "1x2"]
        .groupby("fixture_uid")
        .agg(snap=("observed_at", "max"), n_books=("bookmaker", "nunique"))
    )
    over = (
        pre[(pre["market"] == "1x2")]
        .groupby(["fixture_uid", "bookmaker", "observed_at"])["price"]
        .apply(lambda p: float((1 / p).sum()) - 1)
    )
    overround = over.groupby("fixture_uid").median().rename("overround")
    out = base.merge(stats, left_on="fixture_uid", right_index=True, how="left").merge(
        overround, left_on="fixture_uid", right_index=True, how="left"
    )
    out = out.merge(market_rates(info, fixtures, goal_model, method), on="fixture_uid", how="left")
    out["tau_hours"] = (out["kickoff_at"] - out["snap"]).dt.total_seconds() / 3600
    out["liquidity"] = np.log1p(out["n_books"].fillna(0)) - 20 * (
        out["overround"].fillna(0.1) - 0.05
    )
    return out[["fixture_uid", "lam_mkt_home", "lam_mkt_away", "tau_hours", "liquidity"]]


@dataclass
class FusedPredictor:
    m1: M1Predictor
    fusion: Fusion
    devig_method: str = "multiplicative"
    name: str = "m3_fused"
    version: str = "2"

    def predict_fixtures(self, info: InformationSet, fixtures: pd.DataFrame) -> pd.DataFrame:
        mod = self.m1.rates(info, fixtures)
        mk = market_features(info, fixtures, self.fusion.goal_model, self.devig_method)
        df = (
            fixtures[[*FIXTURE_KEYS]]
            .merge(mod, on="fixture_uid")
            .merge(mk, on="fixture_uid", how="left")
        )
        lh, la, w = self.fusion.rates(df)
        df["lambda_home"], df["lambda_away"], df["market_weight"] = lh, la, w
        grid = grid_frame(lh, la, self.fusion.goal_model)
        return pd.concat([df.reset_index(drop=True), grid], axis=1)


def fixture_losses(pred: pd.DataFrame, outcomes: pd.DataFrame) -> pd.DataFrame:
    """Per-fixture losses; ``outcomes`` has fixture_uid, home_goals, away_goals, round."""
    j = pred.merge(outcomes, on="fixture_uid").dropna(subset=["p_home", "home_goals"])
    if j.empty:
        return j
    y = m.outcome_index(j["home_goals"].astype(int), j["away_goals"].astype(int))
    p = j[["p_home", "p_draw", "p_away"]].to_numpy(float)
    grids = j[GRID_COLS].to_numpy(float).reshape(-1, CAP + 1, CAP + 1)
    h, a = j["home_goals"].astype(int).to_numpy(), j["away_goals"].astype(int).to_numpy()
    j["loss_log"] = [m.log_loss(p[i : i + 1], y[i : i + 1]) for i in range(len(j))]
    j["loss_rps"] = [m.rps(p[i : i + 1], y[i : i + 1]) for i in range(len(j))]
    j["loss_grid"] = [
        m.scoreline_grid_log_loss(grids[i : i + 1], h[i : i + 1], a[i : i + 1], CAP)
        for i in range(len(j))
    ]
    return j


def summarise(losses: pd.DataFrame) -> Mapping[str, float]:
    return {
        "log_loss": float(losses["loss_log"].mean()),
        "rps": float(losses["loss_rps"].mean()),
        "grid_log_loss": float(losses["loss_grid"].mean()),
        "n": float(len(losses)),
    }
