"""Exercise 16: integer programming from scratch, linearisation, and the repo's optimiser.

Run: uv run python docs/learn/exercises/ex16_milp.py
"""

from __future__ import annotations

import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from _check import close, run, task
from scipy.optimize import Bounds, LinearConstraint, milp  # noqa: F401  (you will need them)

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # repo root, for tests.*
from fplh.optimize.milp import (  # noqa: F401  (State and optimise are for your solution)
    SquadRules,
    State,
    brute_force_single_week,
    next_free_transfers,
    optimise,
)
from tests.unit.test_milp import TOY, players

# ------------------------------------------------------------------ demo
P = players(0)
print(P.head(4))
BUDGET = (
    int(
        sum(
            P[P["position"] == q].nsmallest(n, "price")["price"].sum() for q, n in TOY.squad.items()
        )
    )
    + 60
)

# a tiny selection problem: pick exactly 3 of 6 items, total cost ≤ 20, maximise value
VALUE = np.array([5.0, 4.0, 3.5, 6.0, 2.0, 4.5])
COST = np.array([8, 6, 5, 9, 2, 7])


# ------------------------------------------------------------------ your tasks
def pick_three(value: np.ndarray, cost: np.ndarray, budget: float) -> np.ndarray:
    """Binary x maximising value·x with Σx = 3 and cost·x ≤ budget, via scipy.optimize.milp
    (it minimises, so pass −value; integrality=1; bounds 0..1). Return x as 0/1 ints."""
    raise NotImplementedError


def and_feasible_z(a: int, b: int) -> list[int]:
    """Binary z values satisfying z ≤ a, z ≤ b, z ≥ a + b − 1 (enumerate z ∈ {0, 1})."""
    raise NotImplementedError


def sell_price(bought: int, now: int) -> int:
    """FPL sell price in £0.1m units: half of any rise, rounded down."""
    raise NotImplementedError


def free_transfers_next(free: int, transfers: int, cap: int) -> int:
    """No chip: min(cap, max(min(free, cap) − transfers, 0) + 1)."""
    raise NotImplementedError


def toy_objective(players: pd.DataFrame, rules: SquadRules, budget: int) -> float:
    """Run fplh's optimise for one week [2] from an empty squad with `budget` and unlimited
    free transfers (State(gameweek=2, squad={}, bank=budget, free_transfers=15)),
    beta=0.1, chip_cost={}, pool_size=50, transfer_penalty=0. Return the plan's objective."""
    raise NotImplementedError


@task("pick_three equals brute force")
def _() -> None:
    best = max(
        (sum(VALUE[list(c)]), c)
        for c in itertools.combinations(range(6), 3)
        if sum(COST[list(c)]) <= 20
    )
    x = pick_three(VALUE, COST, 20)
    assert x.sum() == 3
    assert COST @ x <= 20
    close(VALUE @ x, best[0])


@task("AND linearisation gives z = a·b")
def _() -> None:
    for a, b in itertools.product((0, 1), repeat=2):
        assert and_feasible_z(a, b) == [a * b], (a, b)


@task("sell_price examples")
def _() -> None:
    assert sell_price(75, 80) == 77
    assert sell_price(60, 63) == 61
    assert sell_price(80, 75) == 75


@task("free_transfers_next matches the repo (no chip)")
def _() -> None:
    rules = SquadRules(squad={}, xi_min={}, ft_cap=5)
    for free, n in itertools.product(range(1, 6), range(0, 5)):
        assert free_transfers_next(free, n, 5) == next_free_transfers(rules, free, n, None, 99)


@task("the repo's optimiser matches brute force on a toy game")
def _() -> None:
    close(toy_objective(P, TOY, BUDGET), brute_force_single_week(P, TOY, BUDGET, 2), tol=1e-6)


run(globals())
