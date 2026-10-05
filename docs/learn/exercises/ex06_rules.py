"""Exercise 06: scoring rules, bonus ranking with ties, and a property check.

Run: uv run python docs/learn/exercises/ex06_rules.py
"""

from __future__ import annotations

import numpy as np
from _check import close, run, task

from fplh.rules import load_rules
from fplh.rules.bonus import assign_bonus_array
from fplh.rules.engine import EVENT_COLUMNS, score_arrays

RULES = load_rules("2025/26")

# ------------------------------------------------------------------ demo
ev = {c: np.zeros(4, dtype=int) for c in EVENT_COLUMNS}
ev.update(
    minutes=np.array([90, 75, 30, 0]),
    goals_scored=np.array([0, 1, 0, 0]),
    assists=np.array([0, 1, 0, 0]),
    goals_conceded=np.array([3, 0, 1, 0]),
)
ev.update(
    {
        c: np.zeros(4, dtype=int)
        for c in ("clearances_blocks_interceptions", "tackles", "recoveries")
    }
)
pos = np.array(["DEF", "MID", "FWD", "GK"])
parts = score_arrays(ev, pos, RULES)
print({k: v.tolist() for k, v in parts.items() if v.any()})
print("totals:", sum(parts.values()).tolist())


def random_events(rng: np.random.Generator, n: int) -> tuple[dict[str, np.ndarray], np.ndarray]:
    e = {
        "minutes": rng.choice([0, 20, 59, 60, 90], n),
        "goals_scored": rng.poisson(0.3, n),
        "assists": rng.poisson(0.2, n),
        "goals_conceded": rng.poisson(1.2, n),
        "saves": rng.poisson(1.0, n),
    }
    e = {k: np.where(e["minutes"] > 0, v, 0) if k != "minutes" else v for k, v in e.items()}
    full = {
        c: np.zeros(n, dtype=int)
        for c in (*EVENT_COLUMNS, "clearances_blocks_interceptions", "tackles", "recoveries")
    }
    full.update(e)
    return full, rng.choice(["GK", "DEF", "MID", "FWD"], n)


# ------------------------------------------------------------------ your tasks
def my_points(e: dict[str, np.ndarray], position: np.ndarray) -> np.ndarray:
    """2025/26 points from ONLY: appearance (1 if 1–59, 2 if 60+), goals by position
    (GK 10, DEF 6, MID 5, FWD 4), assists 3, clean sheet (60+ minutes and 0 conceded:
    GK 4, DEF 4, MID 1, FWD 0), −1 per 2 goals conceded for GK/DEF, +1 per 3 saves.
    Vectorised: no Python loops over players."""
    raise NotImplementedError


def competition_ranks(bps: np.ndarray) -> np.ndarray:
    """1 + number of players with strictly greater BPS (1-D array)."""
    raise NotImplementedError


def bonus(bps: np.ndarray) -> np.ndarray:
    """3/2/1 bonus for ranks 1/2/3 (ties share the rank), 0 otherwise."""
    raise NotImplementedError


def order_independent(e: dict[str, np.ndarray], position: np.ndarray, seed: int) -> bool:
    """Property: permuting the players permutes the points (score_arrays). Return True if
    score_arrays(permuted)[total] == score_arrays(original)[total][perm]."""
    raise NotImplementedError


@task("my_points matches score_arrays on 2,000 random players")
def _() -> None:
    e, p = random_events(np.random.default_rng(6), 2000)
    want = sum(score_arrays(e, p, RULES).values())
    close(my_points(e, p), want)


@task("competition_ranks handles ties")
def _() -> None:
    close(competition_ranks(np.array([40, 35, 35, 20])), [1, 2, 2, 4])


@task("bonus matches the official tie cases")
def _() -> None:
    for b, want in (
        ([40, 40, 30], [3, 3, 1]),
        ([40, 35, 35], [3, 2, 2]),
        ([40, 35, 30, 30], [3, 2, 1, 1]),
        ([31, 31, 31, 20], [3, 3, 3, 0]),
    ):
        close(bonus(np.array(b)), want)


@task("bonus agrees with assign_bonus_array on random matches")
def _() -> None:
    rng = np.random.default_rng(1)
    for _ in range(200):
        b = rng.integers(0, 15, 8)
        close(bonus(b), assign_bonus_array(b))


@task("order_independent holds")
def _() -> None:
    e, p = random_events(np.random.default_rng(2), 300)
    assert all(order_independent(e, p, s) for s in range(5))


run(globals())
