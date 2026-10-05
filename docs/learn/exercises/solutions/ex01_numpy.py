"""Reference solutions for exercise 01."""

from __future__ import annotations

import numpy as np
import pandas as pd


def outer_grid(p_home: np.ndarray, p_away: np.ndarray) -> np.ndarray:
    return p_home[:, None] * p_away[None, :]


def row_ranks(x: np.ndarray) -> np.ndarray:
    return np.argsort(np.argsort(x, axis=1), axis=1)


def count_per_slot(
    match: np.ndarray, minute: np.ndarray, n_matches: int, n_slots: int
) -> np.ndarray:
    out = np.zeros((n_matches, n_slots), dtype=np.int64)
    np.add.at(out, (match, minute), 1)
    return out


def latest_position(df: pd.DataFrame) -> pd.Series:
    return (
        df.sort_values("kickoff_at")
        .drop_duplicates("player_uid", keep="last")
        .set_index("player_uid")["position"]
    )


def state_before(goals: np.ndarray) -> np.ndarray:
    return np.cumsum(goals) - goals
