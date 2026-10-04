"""Market comparison ledger, paper only (ARCHITECTURE.md §10.3, ticket 4.3).

Nothing here, or anywhere in the package, can place a bet: the HTTP client issues GET
requests only, and ``tests/unit/test_ledger.py`` fails the build if any write verb or
bookmaker order endpoint appears in ``src/``.

For every fixture and outcome (1X2, over/under 2.5) with the model's probability p̂ at the
gameweek deadline (the Phase 2 fused grid, walk-forward) and the best pre-match price o
*observed at or before that deadline* (football-data's pre-closing ``market_max``):

* EV = p̂·o − 1;
* a paper stake by fractional Kelly, f = κ (p̂o − 1)/(o − 1) of the paper bankroll, when
  EV exceeds ``min_ev`` (one bet per fixture and market: the best EV);
* CLV = o / o*_close − 1, with o*_close = 1/p_fair the de-vigged closing price (Pinnacle,
  else the market average).

Mean CLV of the bets, with a gameweek-block bootstrap CI, is the evidence of forecasting
skill: a model with no edge has mean CLV ≈ 0. Paper ROI is reported but is not a criterion
(it is dominated by variance at any realistic sample size).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from fplh.evaluate.bootstrap import compare
from fplh.lake.storage import Lake
from fplh.models.market import devig

OUTCOMES = {"1x2": ("home", "draw", "away"), "total": ("over", "under")}
PROB = {"home": "p_home", "draw": "p_draw", "away": "p_away", "over": "p_over25"}


@dataclass(frozen=True)
class LedgerConfig:
    kappa: float = 0.25
    min_ev: float = 0.03
    bankroll: float = 100.0
    devig_method: str = "multiplicative"
    max_stake_fraction: float = 0.05


def _pivot(odds: pd.DataFrame, closing: bool, books: tuple[str, ...]) -> pd.DataFrame:
    """fixture_uid × outcome prices from the first listed book that has them."""
    o = odds[(odds["is_closing"] == closing) & odds["bookmaker"].isin(books)]
    o = o[(o["market"] == "1x2") | ((o["market"] == "total") & (o["line"] == 2.5))]
    rank = {b: i for i, b in enumerate(books)}
    o = o.assign(_rank=o["bookmaker"].map(rank)).sort_values("_rank")
    first = o.drop_duplicates(["fixture_uid", "market", "outcome"])
    out: pd.DataFrame = first.pivot_table(
        index="fixture_uid", columns="outcome", values="price", aggfunc="first"
    )
    seen = first.groupby("fixture_uid")["observed_at"].max()
    return out.assign(observed_at=seen)


def fair_closing(odds: pd.DataFrame, method: str) -> pd.DataFrame:
    close = _pivot(odds, True, ("pinnacle", "market_avg"))
    out = pd.DataFrame(index=close.index)
    for outs in OUTCOMES.values():
        cols = [c for c in outs if c in close]
        if len(cols) != len(outs):
            continue
        ok = close[cols].notna().all(axis=1)
        fair = np.full((len(close), len(outs)), np.nan)
        if ok.any():
            fair[ok.to_numpy()] = devig(close.loc[ok, cols].to_numpy(float), method)
        for j, c in enumerate(outs):
            out[f"fair_{c}"] = fair[:, j]
    return out


def ledger(
    probs: pd.DataFrame, odds: pd.DataFrame, rounds: pd.Series, config: LedgerConfig | None = None
) -> pd.DataFrame:
    """One row per fixture and outcome with a price at the deadline.

    ``probs``: fixture_uid, deadline_at, p_home, p_draw, p_away, p_over25 (horizon 1);
    ``odds``: ``snap_odds``; ``rounds``: fixture_uid → season:round block label."""
    cfg = config or LedgerConfig()
    pre = _pivot(odds, False, ("market_max",))
    fair = fair_closing(odds, cfg.devig_method)
    p = probs.drop_duplicates("fixture_uid", keep="last").set_index("fixture_uid")
    p = p.join(pre, how="inner", rsuffix="_odds").join(fair, how="left")
    p = p[p["observed_at"] <= p["deadline_at"]]  # the price must be available at the deadline
    rows = []
    for fx, r in p.iterrows():
        for market, outs in OUTCOMES.items():
            for o in outs:
                price = r.get(o)
                prob = (
                    r["p_over25"]
                    if o == "over"
                    else (1 - r["p_over25"] if o == "under" else r[PROB[o]])
                )
                if pd.isna(price) or pd.isna(prob) or price <= 1:
                    continue
                close = r.get(f"fair_{o}")
                rows.append(
                    {
                        "fixture_uid": fx,
                        "block": rounds.get(fx, str(fx)),
                        "market": market,
                        "outcome": o,
                        "p_model": float(prob),
                        "price": float(price),
                        "ev": float(prob * price - 1),
                        "clv": float(price * close - 1) if pd.notna(close) else np.nan,
                    }
                )
    return pd.DataFrame(rows)


def paper_bets(
    book: pd.DataFrame, results: pd.DataFrame, config: LedgerConfig | None = None
) -> pd.DataFrame:
    """The best-EV outcome per fixture and market above ``min_ev``, staked by fractional
    Kelly on a running paper bankroll, settled on the results (home_goals, away_goals)."""
    cfg = config or LedgerConfig()
    if book.empty:
        return book
    best = book.sort_values("ev", ascending=False).drop_duplicates(["fixture_uid", "market"])
    bets = best[best["ev"] > cfg.min_ev].join(
        results.set_index("fixture_uid")[["home_goals", "away_goals"]], on="fixture_uid"
    )
    bets = bets.dropna(subset=["home_goals"]).sort_values(["block", "fixture_uid"])
    h, a = bets["home_goals"].to_numpy(), bets["away_goals"].to_numpy()
    won = {
        "home": h > a,
        "draw": h == a,
        "away": h < a,
        "over": h + a > 2.5,
        "under": h + a < 2.5,
    }
    bets["won"] = np.select([bets["outcome"] == k for k in won], list(won.values()), False)
    bank = cfg.bankroll
    stakes, profit = [], []
    for ev, price, w in zip(bets["ev"], bets["price"], bets["won"], strict=True):
        f = min(cfg.kappa * ev / (price - 1), cfg.max_stake_fraction)
        stake = f * bank
        pnl = stake * (price - 1) if w else -stake
        bank += pnl
        stakes.append(stake)
        profit.append(pnl)
    return bets.assign(stake=stakes, profit=profit, bankroll=np.cumsum(profit) + cfg.bankroll)


def summarise(bets: pd.DataFrame, n_boot: int = 2000) -> dict[str, float]:
    if bets.empty:
        return {"bets": 0.0}
    clv = bets.dropna(subset=["clv"]).reset_index(drop=True)
    zero = pd.Series(np.zeros(len(clv)))
    c = compare(-clv["clv"], zero, clv["block"], n_boot=n_boot)  # loss = −CLV
    return {
        "bets": float(len(bets)),
        "mean_ev": float(bets["ev"].mean()),
        "mean_clv": -c.mean_diff,
        "clv_ci_low": -c.ci_high,
        "clv_ci_high": -c.ci_low,
        "staked": float(bets["stake"].sum()),
        "profit": float(bets["profit"].sum()),
        "roi": float(bets["profit"].sum() / bets["stake"].sum()) if bets["stake"].sum() else 0.0,
        "final_bankroll": float(bets["bankroll"].iloc[-1]),
        "hit_rate": float(bets["won"].mean()),
    }


def run_ledger(
    lake: Lake, seasons: list[str], config: LedgerConfig | None = None
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float]]:
    """The ledger over the cached walk-forward fused forecasts of ``seasons``."""
    from fplh.evaluate.phase3 import FusedRates
    from fplh.evaluate.walk_forward import cached_walk_forward
    from fplh.features.information_set import SilverStore
    from fplh.features.spine import historical_deadlines
    from fplh.models.market import load_devig_method

    cfg = config or LedgerConfig(devig_method=load_devig_method())
    store = SilverStore(lake)
    dim = store.get("dim_fixture")
    ds = [d for s in seasons for d in historical_deadlines(dim, s)["deadline_at"]]
    probs = cached_walk_forward(
        store, FusedRates(store, min(ds)), ds, lake=lake, unit="fixture"
    ).predictions
    rounds = dim.set_index("fixture_uid")
    label = rounds["season"].astype(str) + ":" + rounds["round"].astype("Int64").astype(str)
    book = ledger(probs, store.get("snap_odds"), label, cfg)
    bets = paper_bets(book, dim[["fixture_uid", "home_goals", "away_goals"]], cfg)
    return book, bets, summarise(bets)
