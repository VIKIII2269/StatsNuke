"""Live consensus-value paper tracker over The Odds API snapshots (paper only, model v2).

The free plan's UK region carries about 20 bookmakers and three exchanges but not
Pinnacle, so the fair price comes from the Betfair exchange: its back prices, de-vigged
with the power method, are close to the sharp line. Historically, with football-data's
Betfair exchange prices as the anchor, soft books' prices more than 3 % above it beat the
close by +2.9 % [+1.3 %, +4.5 %] on 2022/23–2024/25 (22 bets).

For every snapshot, fixture and market, each soft book's price is compared with the
anchor's fair price at the same snapshot. A paper bet is the first snapshot at which an
outcome clears ``min_ev`` (one per fixture, market, outcome and book). CLV is measured
against the anchor's fair price at the last snapshot before kickoff. Nothing here can
place a bet: the tracker only reads stored snapshots.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from fplh.models.market import devig

ANCHOR = "betfair_ex_uk"
EXCHANGES = ("betfair_ex_uk", "betfair_ex_eu", "smarkets", "matchbook")
OUTCOMES = {"1x2": ("home", "draw", "away"), "total": ("over", "under")}
FIXTURE = ["home_team", "away_team", "kickoff_at"]


def anchor_fair(odds: pd.DataFrame, anchor: str = ANCHOR, method: str = "power") -> pd.DataFrame:
    """Fair probability per (fixture, snapshot, market, outcome) from the anchor's prices."""
    a = odds[(odds["bookmaker"] == anchor) & odds["market"].isin(OUTCOMES)]
    a = a[(a["market"] == "1x2") | (a["line"] == 2.5)]
    rows = []
    for key, g in a.groupby([*FIXTURE, "observed_at", "market"], sort=False):
        outs = OUTCOMES[str(key[-1])]
        prices = g.drop_duplicates("outcome").set_index("outcome")["price"].reindex(outs)
        if prices.isna().any() or (prices <= 1).any():
            continue
        fair = devig(prices.to_numpy(float), method)
        for o, p in zip(outs, fair, strict=True):
            rows.append((*key, o, float(p)))
    return pd.DataFrame(rows, columns=[*FIXTURE, "observed_at", "market", "outcome", "fair"])


def live_consensus(
    odds: pd.DataFrame, *, min_ev: float = 0.03, anchor: str = ANCHOR, method: str = "power"
) -> pd.DataFrame:
    """Paper bets: soft books' prices above the anchor's fair price, with CLV where the
    fixture has a pre-kickoff snapshot after the bet."""
    odds = odds[odds["observed_at"] < odds["kickoff_at"]]  # pre-match snapshots only
    fair = anchor_fair(odds, anchor, method)
    if fair.empty:
        return pd.DataFrame()
    soft = odds[~odds["bookmaker"].isin(EXCHANGES) & odds["market"].isin(OUTCOMES)]
    soft = soft[(soft["market"] == "1x2") | (soft["line"] == 2.5)]
    keys = [*FIXTURE, "observed_at", "market", "outcome"]
    cand = soft.merge(fair, on=keys, how="inner")
    cand["ev"] = cand["fair"] * cand["price"] - 1
    bets = cand[cand["ev"] > min_ev].sort_values("observed_at")
    bets = bets.drop_duplicates([*FIXTURE, "market", "outcome", "bookmaker"], keep="first")
    close = fair.sort_values("observed_at").drop_duplicates(
        [*FIXTURE, "market", "outcome"], keep="last"
    )
    close = close.rename(columns={"fair": "fair_close", "observed_at": "close_at"})
    out = bets.merge(close, on=[*FIXTURE, "market", "outcome"], how="left")
    later = out["close_at"] > out["observed_at"]
    out["clv"] = np.where(later, out["price"] * out["fair_close"] - 1, np.nan)
    cols = [*FIXTURE, "market", "outcome", "bookmaker", "observed_at", "price", "fair", "ev"]
    result: pd.DataFrame = out[[*cols, "close_at", "fair_close", "clv"]].reset_index(drop=True)
    return result


BETS_KEY = "state/paper_bets.parquet"
BET_KEY = [*FIXTURE, "market", "outcome", "bookmaker"]


def _won(outcome: str, home: float, away: float) -> bool:
    return {
        "home": home > away,
        "draw": home == away,
        "away": home < away,
        "over": home + away > 2.5,
        "under": home + away < 2.5,
    }[outcome]


def update_bets(
    stored: pd.DataFrame, current: pd.DataFrame, results: pd.DataFrame, now: pd.Timestamp
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(new bets, the whole book). A bet is kept as first seen; later snapshots only update
    its closing price. Once kicked off and with a result it is settled at 1 unit:
    ``results`` is the season's fixtures: home_team, away_team, home_goals, away_goals."""
    cur = current.copy()
    if stored.empty:
        new = cur
        book = cur
    else:
        known = stored.set_index(BET_KEY).index
        is_new = ~cur.set_index(BET_KEY).index.isin(known)
        new = cur[is_new]
        close = cur.set_index(BET_KEY)[["close_at", "fair_close"]]
        book = stored.set_index(BET_KEY)
        book.update(close)
        book = pd.concat([book.reset_index(), new], ignore_index=True)
    book = book.drop(columns=["won", "profit", "settled"], errors="ignore")
    book["clv"] = np.where(
        book["close_at"] > book["observed_at"], book["price"] * book["fair_close"] - 1, np.nan
    )
    # one league season: each ordered pair meets once, so teams identify the match (the
    # feeds' kickoff times can differ by minutes)
    res = results.dropna(subset=["home_goals", "away_goals"]).drop_duplicates(
        ["home_team", "away_team"], keep="last"
    )
    book = book.merge(
        res[["home_team", "away_team", "home_goals", "away_goals"]],
        on=["home_team", "away_team"],
        how="left",
    )
    book["settled"] = book["home_goals"].notna() & (book["kickoff_at"] <= now)
    won = [
        _won(o, h, a) if s else False
        for o, h, a, s in zip(
            book["outcome"], book["home_goals"], book["away_goals"], book["settled"], strict=True
        )
    ]
    book["won"] = np.where(book["settled"], won, np.nan)
    book["profit"] = np.where(book["settled"], np.where(won, book["price"] - 1, -1.0), np.nan)
    book = book.drop(columns=["home_goals", "away_goals"])
    return new.reset_index(drop=True), book.sort_values("observed_at").reset_index(drop=True)


def summary(book: pd.DataFrame, n_boot: int = 2000, seed: int = 0) -> dict[str, float]:
    """Settled bets: mean CLV with a match-day block bootstrap CI, hit rate, ROI."""
    s = book[book["settled"].astype(bool) & book["clv"].notna()] if not book.empty else book
    if s.empty:
        return {"bets": 0.0}
    clv = s["clv"].to_numpy(float)
    day = pd.to_datetime(s["kickoff_at"]).dt.date.to_numpy()
    blocks = [clv[day == d] for d in np.unique(day)]
    rng = np.random.default_rng(seed)
    boots = [
        np.concatenate([blocks[i] for i in rng.integers(0, len(blocks), len(blocks))]).mean()
        for _ in range(n_boot)
    ]
    return {
        "bets": float(len(s)),
        "mean_clv": float(clv.mean()),
        "clv_low": float(np.quantile(boots, 0.025)),
        "clv_high": float(np.quantile(boots, 0.975)),
        "hit_rate": float(s["won"].astype(float).mean()),
        "roi": float(s["profit"].sum() / len(s)),
    }
