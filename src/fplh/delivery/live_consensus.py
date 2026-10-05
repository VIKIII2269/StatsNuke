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
