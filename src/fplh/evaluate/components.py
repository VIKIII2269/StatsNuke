"""Phase 3 component checks on the tuning seasons (ARCHITECTURE.md §11.6–11.7).

* **M4 minutes** (ticket 3.3): walk-forward π^S, π^60, π^B at every deadline; Brier and
  ECE per stage (target ECE ≤ 0.02 for start and 60+), and A4: Brier of P(start),
  P(60+) and P(appearance) against the naive shares of the last three fixtures.
* **M7–M10** (tickets 3.5–3.6), each given the player's actual minutes: defensive
  contribution threshold Brier (within 2018/19 and 2025/26 only, open question 1c), save
  point log loss, yellow-card Brier and bonus Brier / error, each against its role-mean or
  league-rate baseline.
* **M5/M6 attack** (ticket 3.4): at every deadline, non-penalty goals of each player who
  appeared in the round, scored by Poisson log loss with the player's minutes as
  exposure; A5 compares the shrunk rate with the raw decayed per-90 rate (same history,
  no shrinkage), for all players and for those with little history (< 10 decayed 90s).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import nbinom, poisson

from fplh.evaluate import metrics as m
from fplh.evaluate.bootstrap import compare
from fplh.evaluate.bps import official_weights as official_bps_weights
from fplh.evaluate.walk_forward import cached_walk_forward
from fplh.features.information_set import InformationSet, SilverStore
from fplh.features.minutes import minutes_features, player_history
from fplh.features.spine import historical_deadlines
from fplh.lake.storage import Lake
from fplh.models.attack import fit_attack
from fplh.models.bonus import BonusModel, bps_rows, fit_bonus
from fplh.models.cards import fit_cards
from fplh.models.defence import fit_defence
from fplh.models.gk import fit_saves, keeper_rows, negbin_pmf
from fplh.models.minutes import MinutesPredictor
from fplh.rules.bonus import assign_bonus_array
from fplh.rules.config import load_rules

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


# --- PR 5: M7 defence, M8 saves, M9 cards, M10 bonus -------------------------------------
# Each component is scored given the player's actual minutes (and, for bonus, the match's
# actual events), so it is judged on what it adds to the simulator, not on minutes.


def _rounds(store: SilverStore, seasons: list[str]) -> list[tuple[str, int, pd.Timestamp]]:
    dim = store.get("dim_fixture")
    return [
        (s, int(r), pd.Timestamp(d))
        for s in seasons
        for r, d in historical_deadlines(dim, s).itertuples(index=False)
    ]


def _round_rows(store: SilverStore, season: str, rnd: int) -> pd.DataFrame:
    dim = store.get("dim_fixture")
    fx = set(dim[(dim["season"] == season) & (dim["round"] == rnd)]["fixture_uid"])
    pm = store.get("fact_player_match")
    out: pd.DataFrame = pm[pm["fixture_uid"].isin(fx) & (pm["minutes"] > 0)].copy()
    out["round"] = rnd
    return out


def _compare_rows(
    name: str, losses: dict[str, np.ndarray], block: pd.Series, model: str
) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    for other, loss in losses.items():
        if other == model:
            continue
        c = compare(
            pd.Series(losses[model]), pd.Series(loss), block.reset_index(drop=True), n_boot=1000
        )
        out.append(
            {
                "target": name,
                "model": model,
                "baseline": other,
                "loss_model": float(losses[model].mean()),
                "loss_baseline": float(loss.mean()),
                "diff": c.mean_diff,
                "ci_low": c.ci_low,
                "ci_high": c.ci_high,
            }
        )
    return out


def evaluate_cards(lake: Lake, seasons: list[str]) -> pd.DataFrame:
    """M9: Brier of P(yellow) for players who played (red-carded rows excluded: FPL scores
    them as the red only), against the position's rate (no player shrinkage)."""
    store = SilverStore(lake)
    frames = []
    for season, rnd, deadline in _rounds(store, seasons):
        rates = fit_cards(InformationSet.at(deadline, store))
        r = _round_rows(store, season, rnd)
        r = r[r["red_cards"] == 0]
        played = r["minutes"].to_numpy() / 90
        own = rates.rates_for(r["player_uid"], r["position"])
        pos = np.array([rates.position_rate.get(str(p), 0.0) for p in r["position"]])
        frames.append(
            r.assign(
                season=season,
                p_model=1 - np.exp(-own * played),
                p_position=1 - np.exp(-pos * played),
                y=(r["yellow_cards"] > 0).astype(float),
            )
        )
    df = pd.concat(frames, ignore_index=True)
    y = df["y"].to_numpy()
    losses = {k: (df[f"p_{k}"].to_numpy() - y) ** 2 for k in ("model", "position")}
    out = pd.DataFrame(_compare_rows("yellow (Brier)", losses, _block(df), "model"))
    out["n"] = len(df)
    out["ece_model"] = m.ece(df["p_model"].to_numpy(), y)
    return out


def prematch_rates(store: SilverStore, before: pd.Timestamp) -> pd.DataFrame:
    """M1 pre-update rates (each from matches before it) with fixture uids."""
    from fplh.evaluate.phase2 import load_m1_params, training_predictions

    dim = store.get("dim_fixture")
    hist = training_predictions(store, before, load_m1_params())
    keys = ["season", "home_team", "away_team"]
    out: pd.DataFrame = hist[[*keys, "pred_home", "pred_away"]].merge(
        dim[[*keys, "fixture_uid"]], on=keys
    )
    return out


def evaluate_saves(lake: Lake, seasons: list[str]) -> pd.DataFrame:
    """M8: log loss of the save points (⌊saves/3⌋) of goalkeepers who played, against the
    league rate per 90 (no opponent, no keeper effect) and the opponent-only model."""
    store = SilverStore(lake)
    rounds = _rounds(store, seasons)
    end = max(d for *_, d in rounds) + pd.Timedelta(days=60)
    prematch = prematch_rates(store, end)
    outcomes = keeper_rows(InformationSet.at(end, store), prematch)
    dim = store.get("dim_fixture")
    round_of = dim.set_index("fixture_uid")["round"]
    outcomes = outcomes.assign(fixture_round=outcomes["fixture_uid"].map(round_of))
    frames = []
    for season, rnd, deadline in rounds:
        info = InformationSet.at(deadline, store)
        model = fit_saves(info, prematch)
        r = outcomes[(outcomes["season"] == season) & (outcomes["fixture_round"] == rnd)]
        if r.empty:
            continue
        per = load_rules(season.replace("-", "/")).saves.per
        base = keeper_rows(info, prematch)
        league = float(base["saves"].sum() / max(base["m"].sum(), 1e-9))
        means = {
            "model": model.mean(
                r["mu_opp"].to_numpy(), r["minutes"].to_numpy(), model.multiplier(r["player_uid"])
            ),
            "opponent_only": model.mean(r["mu_opp"].to_numpy(), r["minutes"].to_numpy(), 1.0),
            "league_rate": r["m"].to_numpy() * league,
        }
        y = r["saves"].to_numpy().astype(int)
        top = 40
        cols = {}
        for k, mean in means.items():
            pmf = negbin_pmf(np.atleast_1d(mean), model.size, top)
            pts = np.add.reduceat(pmf, np.arange(0, top + 1, per), axis=1)
            cols[f"loss_{k}"] = -np.log(
                np.clip(pts[np.arange(len(y)), np.minimum(y, top) // per], 1e-12, None)
            )
            cols[f"mean_{k}"] = mean
        frames.append(r.assign(round=rnd, **cols))
    df = pd.concat(frames, ignore_index=True)
    losses = {k: df[f"loss_{k}"].to_numpy() for k in ("model", "opponent_only", "league_rate")}
    out = pd.DataFrame(_compare_rows("save points (log loss)", losses, _block(df), "model"))
    out["n"] = len(df)
    out["mean_saves"] = float(df["saves"].mean())
    out["mean_predicted"] = float(df["mean_model"].mean())
    return out


def evaluate_defence(lake: Lake, seasons: list[str]) -> pd.DataFrame:
    """M7: Brier of P(defensive contribution) — DEF CBI+T ≥ 10, MID/FWD CBI+T+R ≥ 12 (the
    2025/26 thresholds) — for outfield players who played, against the position's mean
    rate with the same dispersion."""
    from fplh.models.defence import ACTIONS, GROUP

    store = SilverStore(lake)
    frames = []
    for season, rnd, deadline in _rounds(store, seasons):
        model = fit_defence(InformationSet.at(deadline, store))
        r = _round_rows(store, season, rnd)
        r = r[r["position"].isin(["DEF", "MID", "FWD"]) & r[ACTIONS[0]].notna()]
        if r.empty or model.players.empty:
            continue
        thr = np.where(r["position"] == "DEF", 10, 12)
        mask = np.array([[a in GROUP[str(p)] for a in ACTIONS] for p in r["position"]])
        total = (r[list(ACTIONS)].to_numpy(dtype=float) * mask).sum(axis=1)
        p_model = model.threshold_prob(
            r["player_uid"], r["position"], r["opponent"], r["minutes"].to_numpy(), thr
        )
        pos_mean = (
            np.array(
                [
                    sum(model.position_rate.get(str(p), {}).get(a, 0.0) for a in GROUP[str(p)])
                    for p in r["position"]
                ]
            )
            * r["minutes"].to_numpy()
            / 90
        )
        k = model.group_size(r["position"])
        p_position = nbinom.sf(thr - 1, k, k / (k + np.clip(pos_mean, 1e-9, None)))
        frames.append(
            r.assign(
                season=season,
                p_model=p_model,
                p_position=p_position,
                y=(total >= thr).astype(float),
            )
        )
    df = pd.concat(frames, ignore_index=True)
    out = []
    for name, sel in (
        ("DEF ≥ 10", df["position"] == "DEF"),
        ("MID/FWD ≥ 12", df["position"] != "DEF"),
    ):
        d = df[sel].reset_index(drop=True)
        y = d["y"].to_numpy()
        losses = {k: (d[f"p_{k}"].to_numpy() - y) ** 2 for k in ("model", "position")}
        for row in _compare_rows(f"DC {name} (Brier)", losses, _block(d), "model"):
            row.update(
                n=len(d),
                rate=float(y.mean()),
                mean_p=float(d["p_model"].mean()),
                ece_model=m.ece(d["p_model"].to_numpy(), y),
            )
            out.append(row)
    return pd.DataFrame(out)


def evaluate_bonus(lake: Lake, seasons: list[str], n_sims: int = 500) -> pd.DataFrame:
    """M10 given each match's actual events: Brier of P(bonus > 0) and squared error of
    E[bonus] against (a) the official BPS table on the same events with no residual and
    (b) the position's bonus rate per appearance."""
    store = SilverStore(lake)
    rng = np.random.default_rng(0)
    frames = []
    for season, rnd, deadline in _rounds(store, seasons):
        info = InformationSet.at(deadline, store)
        model = fit_bonus(info)
        rules = load_rules(season.replace("-", "/"))
        official = BonusModel(weights=official_bps_weights(rules))
        hist = bps_rows(info)
        hist = hist[hist["season"] == hist["season"].max()]
        rate = hist.groupby("position")["bonus"].mean()
        share = hist.groupby("position")["bonus"].apply(lambda b: float((b > 0).mean()))
        r = _round_rows(store, season, rnd)
        for _, g in r.groupby("fixture_uid"):
            ev = {c: g[c].to_numpy() for c in g.columns if c != "position"}
            pos, uid = g["position"].to_numpy(), g["player_uid"].to_numpy()
            mean, sd = model.expected_bps(ev, pos, uid)
            sims = np.round(mean[None, :] + sd[None, :] * rng.standard_normal((n_sims, len(g))))
            bonus = assign_bonus_array(sims, rules.bonus.ranks)
            off_mean, _ = official.expected_bps(ev, pos, uid)
            off = assign_bonus_array(np.round(off_mean)[None, :], rules.bonus.ranks)[0]
            frames.append(
                g.assign(
                    season=season,
                    e_model=bonus.mean(axis=0),
                    p_model=(bonus > 0).mean(axis=0),
                    e_official=off.astype(float),
                    p_official=(off > 0).astype(float),
                    e_position=g["position"].map(rate).fillna(0.0).to_numpy(),
                    p_position=g["position"].map(share).fillna(0.0).to_numpy(),
                )
            )
    df = pd.concat(frames, ignore_index=True)
    y = df["bonus"].to_numpy(dtype=float)
    out = []
    block = _block(df)
    kinds = ("model", "official", "position")
    brier = {k: (df[f"p_{k}"].to_numpy() - (y > 0)) ** 2 for k in kinds}
    sq = {k: (df[f"e_{k}"].to_numpy() - y) ** 2 for k in kinds}
    ab = {k: np.abs(df[f"e_{k}"].to_numpy() - y) for k in kinds}
    out += _compare_rows("P(bonus > 0) (Brier)", brier, block, "model")
    out += _compare_rows("E[bonus] (squared error)", sq, block, "model")
    out += _compare_rows("E[bonus] (absolute error)", ab, block, "model")
    res = pd.DataFrame(out)
    res["n"] = len(df)
    return res
