"""Exercise 17: EV, fractional Kelly, closing-line value — and why ROI is noisy.

Run: uv run python docs/learn/exercises/ex17_ledger.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from _check import close, run, task

from fplh.delivery.ledger import LedgerConfig, paper_bets

# ------------------------------------------------------------------ demo
book = pd.DataFrame(
    {
        "fixture_uid": ["f1", "f2", "f3"],
        "block": ["s:1", "s:1", "s:2"],
        "market": ["1x2"] * 3,
        "outcome": ["home", "away", "draw"],
        "p_model": [0.50, 0.30, 0.30],
        "price": [2.20, 3.80, 3.30],
        "ev": [0.10, 0.14, -0.01],
        "clv": [0.02, -0.03, 0.0],
    }
)
results = pd.DataFrame(
    {"fixture_uid": ["f1", "f2", "f3"], "home_goals": [2, 0, 1], "away_goals": [1, 1, 1]}
)
BETS = paper_bets(book, results, LedgerConfig())
print(BETS[["fixture_uid", "outcome", "price", "ev", "stake", "profit", "bankroll"]])


# ------------------------------------------------------------------ your tasks
def ev(p: float, price: float) -> float:
    """Expected value per unit staked: p·o − 1."""
    raise NotImplementedError


def kelly_fraction(p: float, price: float, kappa: float = 0.25, cap: float = 0.05) -> float:
    """Fractional Kelly κ·(p·o − 1)/(o − 1), capped at `cap` (0 if EV ≤ 0)."""
    raise NotImplementedError


def clv(price_taken: float, p_fair_close: float) -> float:
    """o_taken / (1 / p_fair_close) − 1."""
    raise NotImplementedError


def no_edge_season(seed: int, n: int = 400) -> tuple[float, float]:
    """Simulate a bettor with NO edge: each bet's true probability p ~ U(0.25, 0.6) equals the
    fair closing probability; the price taken is fair 1/p times exp(N(0, 0.03)) noise; stake
    1 unit; outcome ~ Bernoulli(p). Return (mean CLV, ROI = profit / staked).
    Use rng = np.random.default_rng(seed) and draw in the order p, noise, outcome."""
    raise NotImplementedError


@task("ev and kelly: the lesson's example")
def _() -> None:
    close(ev(0.5, 2.2), 0.10)
    close(kelly_fraction(0.5, 2.2), 0.25 * 0.1 / 1.2)
    close(kelly_fraction(0.4, 2.8), 0.25 * 0.12 / 1.8)
    close(kelly_fraction(0.6, 3.0), 0.05)  # capped
    close(kelly_fraction(0.3, 3.0), 0.0)  # negative EV → no bet


@task("kelly_fraction reproduces the ledger's stakes on a running bankroll")
def _() -> None:
    first, second = BETS.iloc[0], BETS.iloc[1]
    close(kelly_fraction(0.5, 2.2) * 100.0, first["stake"])  # paper bankroll starts at 100
    close(kelly_fraction(0.3, 3.8) * first["bankroll"], second["stake"])  # f3 (EV < 3%) skipped


@task("clv: took 2.10 when the fair close was 0.50")
def _() -> None:
    close(clv(2.10, 0.50), 0.05)


@task("no edge → mean CLV ≈ 0, but ROI swings a lot across seasons")
def _() -> None:
    runs = [no_edge_season(s) for s in range(30)]
    clvs, rois = np.array([r[0] for r in runs]), np.array([r[1] for r in runs])
    assert abs(clvs.mean()) < 0.01, clvs.mean()
    assert rois.std() > 5 * clvs.std(), (rois.std(), clvs.std())


run(globals())
