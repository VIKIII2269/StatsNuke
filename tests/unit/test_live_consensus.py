from __future__ import annotations

import numpy as np
import pandas as pd

from fplh.delivery.live_consensus import live_consensus
from fplh.models.market import devig

KO = pd.Timestamp("2026-10-10 11:30", tz="UTC")


def rows(book: str, at: pd.Timestamp, prices: tuple[float, float, float]) -> list[dict]:
    return [
        {
            "home_team": "ars",
            "away_team": "lee",
            "kickoff_at": KO,
            "bookmaker": book,
            "market": "1x2",
            "line": np.nan,
            "outcome": o,
            "price": p,
            "observed_at": at,
        }
        for o, p in zip(("home", "draw", "away"), prices, strict=True)
    ]


def test_soft_price_above_the_exchange_fair_price_is_a_paper_bet_with_clv() -> None:
    t1, t2 = KO - pd.Timedelta(days=2), KO - pd.Timedelta(hours=1)
    odds = pd.DataFrame(
        [
            *rows("betfair_ex_uk", t1, (1.30, 6.0, 11.0)),
            *rows("skybet", t1, (1.28, 5.5, 13.0)),  # away price stale and high
            *rows("smarkets", t1, (1.29, 6.2, 14.0)),  # an exchange: never bet
            *rows("betfair_ex_uk", t2, (1.28, 6.2, 12.5)),  # the line moved towards skybet
            *rows("skybet", KO + pd.Timedelta(minutes=5), (9.0, 9.0, 9.0)),  # in-play: ignored
        ]
    )
    bets = live_consensus(odds, min_ev=0.03)
    assert list(bets["bookmaker"]) == ["skybet"] and list(bets["outcome"]) == ["away"]
    fair = devig(np.array([1.30, 6.0, 11.0]), "power")
    close = devig(np.array([1.28, 6.2, 12.5]), "power")
    b = bets.iloc[0]
    assert np.isclose(b["ev"], 13.0 * fair[2] - 1)
    assert np.isclose(b["clv"], 13.0 * close[2] - 1)


def test_no_anchor_no_bets_and_no_clv_without_a_later_snapshot() -> None:
    t1 = KO - pd.Timedelta(days=2)
    assert live_consensus(pd.DataFrame(rows("skybet", t1, (1.3, 6.0, 11.0)))).empty
    odds = pd.DataFrame(
        [*rows("betfair_ex_uk", t1, (1.30, 6.0, 11.0)), *rows("skybet", t1, (1.28, 5.5, 13.0))]
    )
    bets = live_consensus(odds)
    assert len(bets) == 1 and np.isnan(bets.iloc[0]["clv"])


def test_bets_are_kept_as_first_seen_closed_and_settled() -> None:
    from fplh.delivery.live_consensus import summary, update_bets

    t1, t2 = KO - pd.Timedelta(days=2), KO - pd.Timedelta(hours=1)
    first = pd.DataFrame(
        [*rows("betfair_ex_uk", t1, (1.30, 6.0, 11.0)), *rows("skybet", t1, (1.28, 5.5, 13.0))]
    )
    new, book = update_bets(
        pd.DataFrame(),
        live_consensus(first),
        pd.DataFrame(columns=["home_team", "away_team", "home_goals", "away_goals"]),
        t1,
    )
    assert len(new) == 1 and not book["settled"].any() and np.isnan(book["clv"].iloc[0])
    later = pd.concat(
        [first, pd.DataFrame(rows("betfair_ex_uk", t2, (1.28, 6.2, 12.5)))], ignore_index=True
    )
    results = pd.DataFrame(
        {"home_team": ["ars"], "away_team": ["lee"], "home_goals": [0], "away_goals": [1]}
    )
    new2, book2 = update_bets(book, live_consensus(later), results, KO + pd.Timedelta(hours=3))
    assert new2.empty and len(book2) == 1  # the same bet, not a second one
    b = book2.iloc[0]
    close = devig(np.array([1.28, 6.2, 12.5]), "power")
    assert b["price"] == 13.0 and np.isclose(b["clv"], 13.0 * close[2] - 1)
    assert b["settled"] and b["won"] == 1 and np.isclose(b["profit"], 12.0)
    s = summary(book2, n_boot=50)
    assert s["bets"] == 1 and np.isclose(s["roi"], 12.0) and np.isclose(s["mean_clv"], b["clv"])
