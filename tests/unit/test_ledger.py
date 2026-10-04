"""Paper ledger (ticket 4.3): EV, fractional Kelly, CLV; and no code path can place a bet."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from fplh.delivery.ledger import LedgerConfig, ledger, paper_bets, summarise

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
