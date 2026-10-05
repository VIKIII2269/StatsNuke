"""Reference solutions for exercise 16."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, milp

from fplh.optimize.milp import SquadRules, State, optimise


def pick_three(value: np.ndarray, cost: np.ndarray, budget: float) -> np.ndarray:
    n = len(value)
    a = np.vstack([np.ones(n), cost])
    res = milp(
        -value,
        constraints=LinearConstraint(a, [3, -np.inf], [3, budget]),
        integrality=np.ones(n),
        bounds=Bounds(0, 1),
    )
    return np.round(res.x).astype(int)


def and_feasible_z(a: int, b: int) -> list[int]:
    return [z for z in (0, 1) if z <= a and z <= b and z >= a + b - 1]


def sell_price(bought: int, now: int) -> int:
    return bought + (now - bought) // 2 if now > bought else now


def free_transfers_next(free: int, transfers: int, cap: int) -> int:
    return min(cap, max(min(free, cap) - transfers, 0) + 1)


def toy_objective(players: pd.DataFrame, rules: SquadRules, budget: int) -> float:
    state = State(gameweek=2, squad={}, bank=budget, free_transfers=15)
    plan = optimise(
        players, state, rules, [2], beta=0.1, chip_cost={}, pool_size=50, transfer_penalty=0
    )[0]
    return plan.objective
