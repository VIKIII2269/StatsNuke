"""FPL prices as game state for season replays (ARCHITECTURE.md §10.1).

A player's price in gameweek g is the ``value`` (£0.1m) on his ``fact_player_match`` row for
that gameweek: the price at its deadline, which is public then. It is the environment a
replay trades in, never a model feature. Gameweeks without a row (blanks) carry the last
known price forward; before a player's first row he cannot be bought (NaN).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from fplh.features.information_set import SilverStore


def price_table(store: SilverStore, season: str, last_gw: int = 38) -> pd.DataFrame:
    """index player_uid, columns 1..last_gw: price in £0.1m (NaN before the first row)."""
    pm = store.get("fact_player_match")
    pm = pm[(pm["season"] == season) & pm["player_uid"].notna()]
    wide = (
        pm.sort_values("kickoff_at")
        .drop_duplicates(["player_uid", "round"], keep="first")
        .pivot(index="player_uid", columns="round", values="value")
        .reindex(columns=range(1, last_gw + 1))
    )
    out: pd.DataFrame = wide.astype(float).ffill(axis=1)
    return out


def sell_price(buy: int, now: int) -> int:
    """Half of any rise, rounded down to £0.1m; the full fall otherwise."""
    if now > buy:
        return int(buy + (now - buy) // 2)
    return int(now)


def sell_prices(buy: np.ndarray, now: np.ndarray) -> np.ndarray:
    out: np.ndarray = np.where(now > buy, buy + (now - buy) // 2, now).astype(np.int64)
    return out
