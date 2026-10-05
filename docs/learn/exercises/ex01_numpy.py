"""Exercise 01: NumPy and pandas idioms used across fplh.

Run: uv run python docs/learn/exercises/ex01_numpy.py
Implement each function below (replace ``raise NotImplementedError``) until every task passes.
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
from _check import close, run, task
from scipy.stats import poisson

# ------------------------------------------------------------------ demo (read, then run)
K = np.arange(11)
lh, la = 1.6, 1.1
grid = np.outer(poisson.pmf(K, lh), poisson.pmf(K, la))
print(f"P(1-1) under independent Poisson(1.6, 1.1): {grid[1, 1]:.4f}")

x = np.random.default_rng(0).random(200_000)
t0 = time.perf_counter()
slow = sum(v * v for v in x)
t1 = time.perf_counter()
fast = float(x @ x)
t2 = time.perf_counter()
print(f"sum of squares: loop {1e3 * (t1 - t0):.1f} ms vs vectorised {1e3 * (t2 - t1):.3f} ms")


# ------------------------------------------------------------------ your tasks
def outer_grid(p_home: np.ndarray, p_away: np.ndarray) -> np.ndarray:
    """grid[i, j] = p_home[i] * p_away[j], using broadcasting (no np.outer, no loops)."""
    raise NotImplementedError


def row_ranks(x: np.ndarray) -> np.ndarray:
    """0-based rank of each element within its row (smallest = 0), for a 2-D array.
    Hint: argsort twice along axis=1."""
    raise NotImplementedError


def count_per_slot(
    match: np.ndarray, minute: np.ndarray, n_matches: int, n_slots: int
) -> np.ndarray:
    """(n_matches, n_slots) counts of events; repeated (match, minute) pairs must all count.
    Hint: np.add.at."""
    raise NotImplementedError


def latest_position(df: pd.DataFrame) -> pd.Series:
    """For a frame with player_uid, kickoff_at, position: each player's position in their most
    recent match, as a Series indexed by player_uid (the idiom of shrinkage.player_groups)."""
    raise NotImplementedError


def state_before(goals: np.ndarray) -> np.ndarray:
    """Given goals per minute (1-D), the goals scored strictly before each minute."""
    raise NotImplementedError


@task("outer_grid matches np.outer")
def _() -> None:
    a, b = poisson.pmf(K, 1.4), poisson.pmf(K, 0.9)
    close(outer_grid(a, b), np.outer(a, b))


@task("row_ranks ranks each row")
def _() -> None:
    x = np.array([[30, 10, 20], [5, 9, 1]])
    close(row_ranks(x), [[2, 0, 1], [1, 2, 0]])


@task("count_per_slot counts duplicates")
def _() -> None:
    got = count_per_slot(np.array([0, 0, 0, 1]), np.array([3, 3, 7, 0]), 2, 10)
    want = np.zeros((2, 10))
    want[0, 3], want[0, 7], want[1, 0] = 2, 1, 1
    close(got, want)


@task("latest_position takes the most recent row per player")
def _() -> None:
    df = pd.DataFrame(
        {
            "player_uid": ["a", "a", "b"],
            "kickoff_at": pd.to_datetime(["2024-08-01", "2024-09-01", "2024-08-15"]),
            "position": ["MID", "FWD", "DEF"],
        }
    ).sample(frac=1, random_state=1)
    got = latest_position(df).sort_index()
    assert got.to_dict() == {"a": "FWD", "b": "DEF"}, got.to_dict()


@task("state_before excludes the current minute")
def _() -> None:
    close(state_before(np.array([0, 1, 0, 2, 0])), [0, 0, 1, 1, 3])


run(globals())
