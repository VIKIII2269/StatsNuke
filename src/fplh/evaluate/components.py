"""Phase 3 component checks on the tuning seasons (ARCHITECTURE.md §11.6–11.7).

* **M4 minutes** (ticket 3.3): walk-forward π^S, π^60, π^B at every deadline; Brier and
  ECE per stage (target ECE ≤ 0.02 for start and 60+), and A4: Brier of P(start),
  P(60+) and P(appearance) against the naive shares of the last three fixtures.
* **M5/M6 attack** (ticket 3.4): at every deadline, non-penalty goals of each player who
  appeared in the round, scored by Poisson log loss with the player's minutes as
  exposure; A5 compares the shrunk rate with the raw decayed per-90 rate (same history,
  no shrinkage), for all players and for those with little history (< 10 decayed 90s).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import poisson

from fplh.evaluate import metrics as m
from fplh.evaluate.bootstrap import compare
from fplh.evaluate.walk_forward import cached_walk_forward
from fplh.features.information_set import InformationSet, SilverStore
from fplh.features.minutes import minutes_features, player_history
from fplh.features.spine import historical_deadlines
from fplh.lake.storage import Lake
from fplh.models.attack import fit_attack
from fplh.models.minutes import MinutesPredictor

LOW_HISTORY = 10.0  # decayed 90-minute equivalents


def _block(df: pd.DataFrame) -> pd.Series:
    out: pd.Series = df["season"].astype(str) + ":" + df["round"].astype("Int64").astype(str)
    return out


def evaluate_minutes(lake: Lake, seasons: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    store = SilverStore(lake)
    dim = store.get("dim_fixture")
    deadlines = [d for s in seasons for d in historical_deadlines(dim, s)["deadline_at"]]
    pred = cached_walk_forward(store, MinutesPredictor(), deadlines, lake=lake).predictions
    info = InformationSet.at(max(deadlines) + pd.Timedelta(days=60), store)
    hist = player_history(info)[
        ["player_uid", "fixture_uid", "started", "minutes", "season", "team", "kickoff_at"]
    ]
    rows = pred.merge(hist, on=["player_uid", "fixture_uid"])  # spine rows with an outcome
    rows = rows.merge(dim[["fixture_uid", "round"]], on="fixture_uid").reset_index(drop=True)
    naive = minutes_features(info, rows)
    y_start = rows["started"].to_numpy() > 0
    y_full = rows["minutes"].to_numpy() >= 60
    y_app = rows["minutes"].to_numpy() > 0
    p60 = rows["p_start"] * rows["p_full"]
    papp = rows["p_start"] + (1 - rows["p_start"]) * rows["p_sub"]
    stages = [
        ("start", rows["p_start"].to_numpy(), y_start, np.ones(len(rows), bool)),
        ("60+ | start", rows["p_full"].to_numpy(), y_full, y_start),
        ("sub | not started", rows["p_sub"].to_numpy(), y_app, ~y_start),
        ("60+", p60.to_numpy(), y_full, np.ones(len(rows), bool)),
        ("appearance", papp.to_numpy(), y_app, np.ones(len(rows), bool)),
    ]
    table = pd.DataFrame(
        [
            {
                "stage": name,
                "n": int(mask.sum()),
                "brier": m.brier(p[mask], y[mask]),
                "ece": m.ece(p[mask], y[mask]),
                "mean_p": float(p[mask].mean()),
                "mean_y": float(y[mask].mean()),
            }
            for name, p, y, mask in stages
        ]
    )
    block = _block(rows)
    comps = []
    for name, p, base, y in (
        ("start", rows["p_start"], naive["h_started_3"], y_start),
        ("60+", p60, naive["h_full_3"], y_full),
        ("appearance", papp, naive["h_appeared_3"], y_app),
    ):
        loss_a = (p.to_numpy() - y) ** 2
        loss_b = (base.fillna(0).to_numpy() - y) ** 2
        c = compare(pd.Series(loss_a), pd.Series(loss_b), block.reset_index(drop=True), n_boot=1000)
        comps.append(
            {
                "target": name,
                "brier_model": float(loss_a.mean()),
                "brier_naive_last3": float(loss_b.mean()),
                "diff": c.mean_diff,
                "ci_low": c.ci_low,
                "ci_high": c.ci_high,
            }
        )
    return table, pd.DataFrame(comps)


def evaluate_attack(lake: Lake, seasons: list[str]) -> pd.DataFrame:
    store = SilverStore(lake)
    dim = store.get("dim_fixture")
    shots = store.get("fact_shot")
    np_goals = (
        shots[
            (shots["result"] == "Goal")
            & (shots["situation"] != "Penalty")
            & shots["player_uid"].notna()
        ]
        .groupby(["player_uid", "fixture_uid"])
        .size()
        .rename("np_goals")
    )
    us = store.get("fact_player_match_understat")
    us = us[us["player_uid"].notna() & (us["minutes"] > 0)][
        ["player_uid", "fixture_uid", "minutes"]
    ]
    us = us.join(np_goals, on=["player_uid", "fixture_uid"]).fillna({"np_goals": 0})
    rows = []
    for season in seasons:
        for rnd, deadline in historical_deadlines(dim, season).itertuples(index=False):
            info = InformationSet.at(deadline, store)
            rates = fit_attack(info)
            fx = dim[(dim["season"] == season) & (dim["round"] == rnd)]["fixture_uid"]
            r = us[us["fixture_uid"].isin(set(fx))]
            pl = rates.players.reindex(r["player_uid"].to_numpy())
            group_mean = rates.players.groupby("position")["goal_rate"].mean().mean()
            exposure = pl["exposure"].fillna(0).to_numpy()
            raw = np.where(exposure > 0, pl["raw_goal_rate"].to_numpy(), np.nan)
            frame = r.assign(
                season=season,
                round=rnd,
                shrunk=np.nan_to_num(pl["goal_rate"].to_numpy(), nan=group_mean),
                raw=np.nan_to_num(raw, nan=group_mean),
                history=exposure,
            )
            rows.append(frame)
    df = pd.concat(rows, ignore_index=True)
    out = []
    low = df["history"].to_numpy() < LOW_HISTORY
    for name, sel in (("all", np.ones(len(df), bool)), ("history < 10 × 90'", low)):
        d = df[sel].reset_index(drop=True)
        mu = d["minutes"].to_numpy() / 90
        la = -poisson.logpmf(d["np_goals"], np.clip(d["shrunk"].to_numpy() * mu, 1e-6, None))
        lb = -poisson.logpmf(d["np_goals"], np.clip(d["raw"].to_numpy() * mu, 1e-6, None))
        c = compare(pd.Series(la), pd.Series(lb), _block(d), n_boot=1000)
        out.append(
            {
                "players": name,
                "n": len(d),
                "log_loss_shrunk": float(la.mean()),
                "log_loss_raw": float(lb.mean()),
                "diff": c.mean_diff,
                "ci_low": c.ci_low,
                "ci_high": c.ci_high,
            }
        )
    return pd.DataFrame(out)
