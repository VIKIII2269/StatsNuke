"""Reference solutions for exercise 06."""

from __future__ import annotations

import numpy as np

from fplh.rules import load_rules
from fplh.rules.engine import score_arrays

RULES = load_rules("2025/26")


def my_points(e: dict[str, np.ndarray], position: np.ndarray) -> np.ndarray:
    m = e["minutes"]
    played = m > 0
    pts = np.where(m >= 60, 2, np.where(played, 1, 0))
    goal = np.select([position == "GK", position == "DEF", position == "MID"], [10, 6, 5], 4)
    pts = pts + e["goals_scored"] * goal + 3 * e["assists"]
    cs = np.select([position == "GK", position == "DEF", position == "MID"], [4, 4, 1], 0)
    pts = pts + np.where((m >= 60) & (e["goals_conceded"] == 0), cs, 0)
    gkdef = (position == "GK") | (position == "DEF")
    pts = pts - np.where(gkdef, e["goals_conceded"] // 2, 0)
    return pts + e["saves"] // 3


def competition_ranks(bps: np.ndarray) -> np.ndarray:
    b = np.asarray(bps)
    return 1 + (b[None, :] > b[:, None]).sum(axis=1)


def bonus(bps: np.ndarray) -> np.ndarray:
    r = competition_ranks(bps)
    return np.select([r == 1, r == 2, r == 3], [3, 2, 1], 0)


def order_independent(e: dict[str, np.ndarray], position: np.ndarray, seed: int) -> bool:
    perm = np.random.default_rng(seed).permutation(len(position))
    a = sum(score_arrays(e, position, RULES).values())[perm]
    b = sum(score_arrays({k: v[perm] for k, v in e.items()}, position[perm], RULES).values())
    return bool(np.array_equal(a, b))
