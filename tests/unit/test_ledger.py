"""Paper ledger (ticket 4.3): EV, fractional Kelly, CLV; and no code path can place a bet."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from fplh.delivery.ledger import (
    LedgerConfig,
    consensus_ledger,
    fair_closing,
    ledger,
    paper_bets,
    summarise,
)
from fplh.models.market import devig

SRC = Path(__file__).resolve().parents[2] / "src"
T0 = pd.Timestamp("2024-08-16 18:00", tz="UTC")


def odds_row(
    fx: str,
    book: str,
    closing: bool,
    market: str,
    outcome: str,
    price: float,
    seen: pd.Timestamp,
    line: float | None = None,
) -> dict[str, object]:
    return {
        "fixture_uid": fx,
        "bookmaker": book,
        "is_closing": closing,
        "market": market,
        "outcome": outcome,
        "price": price,
        "observed_at": seen,
        "line": line,
    }


def frames() -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    rows = []
    for fx, seen in (("s:a:b", T0 - pd.Timedelta(hours=3)), ("s:c:d", T0 + pd.Timedelta(hours=3))):
        for o, p in (("home", 2.5), ("draw", 3.4), ("away", 3.0)):
            rows.append(odds_row(fx, "market_max", False, "1x2", o, p, seen))
        for o, p in (("home", 2.2), ("draw", 3.3), ("away", 3.4)):
            rows.append(odds_row(fx, "pinnacle", True, "1x2", o, p, T0 + pd.Timedelta(days=1)))
        for o, p in (("over", 1.9), ("under", 2.0)):
            rows.append(odds_row(fx, "market_max", False, "total", o, p, seen, 2.5))
            rows.append(
                odds_row(fx, "pinnacle", True, "total", o, p - 0.05, T0 + pd.Timedelta(days=1), 2.5)
            )
    probs = pd.DataFrame(
        {
            "fixture_uid": ["s:a:b", "s:c:d"],
            "deadline_at": [T0, T0],
            "p_home": [0.50, 0.50],
            "p_draw": [0.25, 0.25],
            "p_away": [0.25, 0.25],
            "p_over25": [0.5, 0.5],
        }
    )
    return probs, pd.DataFrame(rows), pd.Series({"s:a:b": "s:1", "s:c:d": "s:1"})


def test_ev_and_clv_use_prices_available_at_the_deadline() -> None:
    probs, odds, rounds = frames()
    book = ledger(probs, odds, rounds)
    assert set(book["fixture_uid"]) == {"s:a:b"}  # s:c:d's price came after the deadline
    home = book[(book["outcome"] == "home")].iloc[0]
    assert np.isclose(home["ev"], 0.5 * 2.5 - 1)
    fair = (1 / 2.2) / (1 / 2.2 + 1 / 3.3 + 1 / 3.4)  # multiplicative de-vig of the close
    assert np.isclose(home["clv"], 2.5 * fair - 1)
    assert set(book["outcome"]) == {"home", "draw", "away", "over", "under"}


@pytest.mark.parametrize("method", ["multiplicative", "power", "shin"])
def test_closing_prices_are_devigged_one_market_at_a_time(method: str) -> None:
    _, odds, _ = frames()
    fair = fair_closing(odds, method)
    assert len(fair) == 2
    sums = fair[["fair_home", "fair_draw", "fair_away"]].sum(axis=1)
    assert np.allclose(sums, 1.0)
    assert np.allclose(fair[["fair_over", "fair_under"]].sum(axis=1), 1.0)
    one = fair_closing(odds[odds["fixture_uid"] == "s:a:b"], method)
    assert np.allclose(one.loc["s:a:b"], fair.loc["s:a:b"])  # independent of other rows


def test_fractional_kelly_paper_stakes_and_settlement() -> None:
    probs, odds, rounds = frames()
    book = ledger(probs, odds, rounds)
    results = pd.DataFrame({"fixture_uid": ["s:a:b"], "home_goals": [2], "away_goals": [1]})
    cfg = LedgerConfig(kappa=0.25, min_ev=0.03, max_stake_fraction=1.0)
    bets = paper_bets(book, results, cfg)
    assert list(bets["outcome"]) == ["home"]  # best EV per market, above the threshold
    f = 0.25 * (0.5 * 2.5 - 1) / (2.5 - 1)
    assert np.isclose(bets["stake"].iloc[0], f * 100)
    assert bets["won"].iloc[0] and np.isclose(bets["profit"].iloc[0], f * 100 * 1.5)
    s = summarise(bets, n_boot=100)
    assert s["bets"] == 1 and np.isclose(s["roi"], 1.5)


def test_no_code_path_can_place_a_bet() -> None:
    """The package may read odds but never write to a bookmaker or exchange."""
    write = re.compile(
        r"\.(post|put|patch|delete)\(|method\s*=\s*['\"](POST|PUT|PATCH|DELETE)['\"]"
        r"|place_?bet|placeOrders?|/orders\b|betslip|submit_?bet",
        re.IGNORECASE,
    )
    hits = [
        f"{p.relative_to(SRC)}:{i}: {line.strip()}"
        for p in SRC.rglob("*.py")
        for i, line in enumerate(p.read_text().splitlines(), 1)
        if write.search(line)
    ]
    assert hits == [], "write call or bet placement found:\n" + "\n".join(hits)


def test_consensus_bets_soft_prices_above_the_sharp_fair_price() -> None:
    t = T0 - pd.Timedelta(days=1)
    rows = [
        *(
            odds_row("s:a:b", "pinnacle", False, "1x2", o, p, t)
            for o, p in (("home", 2.0), ("draw", 3.5), ("away", 4.0))
        ),
        *(
            odds_row("s:a:b", "pinnacle", True, "1x2", o, p, T0)
            for o, p in (("home", 1.9), ("draw", 3.6), ("away", 4.4))
        ),
        *(
            odds_row("s:a:b", "bet365", False, "1x2", o, p, t)
            for o, p in (("home", 2.2), ("draw", 3.2), ("away", 3.8))
        ),
        # the market maximum is not a book: never bet
        *(
            odds_row("s:a:b", "market_max", False, "1x2", o, p, t)
            for o, p in (("home", 9.0), ("draw", 9.0), ("away", 9.0))
        ),
    ]
    book = consensus_ledger(pd.DataFrame(rows), pd.Series({"s:a:b": "s:1"}), method="power")
    assert set(book["bookmaker"]) == {"bet365"}
    fair = devig(np.array([2.0, 3.5, 4.0]), "power")
    close = devig(np.array([1.9, 3.6, 4.4]), "power")
    home = book[book["outcome"] == "home"].iloc[0]
    assert np.isclose(home["ev"], 2.2 * fair[0] - 1)
    assert np.isclose(home["clv"], 2.2 * close[0] - 1)
    assert home["ev"] > 0 > book[book["outcome"] == "draw"]["ev"].iloc[0]


def test_hybrid_blends_the_model_into_the_fair_price() -> None:
    t = T0 - pd.Timedelta(days=1)
    rows = [
        *(
            odds_row("s:a:b", "pinnacle", False, "1x2", o, p, t)
            for o, p in (("home", 2.0), ("draw", 3.5), ("away", 4.0))
        ),
        *(
            odds_row("s:a:b", "bet365", False, "1x2", o, p, t)
            for o, p in (("home", 2.2), ("draw", 3.2), ("away", 3.8))
        ),
    ]
    model = pd.DataFrame(
        {
            "fixture_uid": ["s:a:b"],
            "p_home": [0.6],
            "p_draw": [0.2],
            "p_away": [0.2],
            "p_over25": [0.5],
        }
    )
    labels = pd.Series({"s:a:b": "s:1"})
    a = consensus_ledger(pd.DataFrame(rows), labels)
    b = consensus_ledger(pd.DataFrame(rows), labels, model=model, model_weight=0.5)
    pa = a[a["outcome"] == "home"]["p_model"].iloc[0]
    pb = b[b["outcome"] == "home"]["p_model"].iloc[0]
    assert np.isclose(pb, 0.5 * pa + 0.5 * 0.6)
