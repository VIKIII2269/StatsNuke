"""Bonus points from BPS with the official tie rule.

Players are ranked within each fixture by *competition ranking* (1 + number of players
with strictly more BPS), and rank r earns ``ranks[r-1]`` for r ≤ len(ranks). This
reproduces the official tie cases: a tie for 1st gives 3, 3, 1; a tie for 2nd gives
3, 2, 2; a tie for 3rd gives 3, 2, 1, 1.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd


def assign_bonus(
    bps: pd.Series,
    fixture: pd.Series,
    ranks: Sequence[int] = (3, 2, 1),
    eligible: pd.Series | None = None,
) -> pd.Series:
    """Bonus per row. ``eligible`` (e.g. ``minutes > 0``) excludes rows from ranking."""
    if eligible is None:
        eligible = pd.Series(True, index=bps.index)
    work = pd.DataFrame({"bps": bps.where(eligible), "fixture": fixture})
    rank = work.groupby("fixture")["bps"].rank(method="min", ascending=False)
    award = np.zeros(len(work), dtype=np.int64)
    r = rank.to_numpy()
    for i, pts in enumerate(ranks, start=1):
        award[r == i] = pts
    return pd.Series(award, index=bps.index, name="bonus")


def assign_bonus_array(
    bps: np.ndarray, ranks: Sequence[int] = (3, 2, 1), eligible: np.ndarray | None = None
) -> np.ndarray:
    """Vectorised over leading axes: ``bps`` is (..., players in one fixture).
    ``eligible`` (broadcastable, e.g. ``minutes > 0``) excludes players from ranking:
    they get no bonus and do not push anyone down."""
    b = np.asarray(bps, dtype=float)
    if eligible is not None:
        b = np.where(eligible, b, -np.inf)
    # rank = 1 + number of players with strictly greater BPS
    rank = 1 + (b[..., None, :] > b[..., :, None]).sum(axis=-1)
    award = np.zeros(b.shape, dtype=np.int64)
    for i, pts in enumerate(ranks, start=1):
        award[rank == i] = pts
    if eligible is not None:
        award = np.where(eligible, award, 0)
    return award
