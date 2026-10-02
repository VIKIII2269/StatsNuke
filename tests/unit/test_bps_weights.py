from __future__ import annotations

import numpy as np
import pandas as pd

from fplh.evaluate.bps import design, estimate_bps_weights, official_weights
from fplh.rules.config import load_rules


def synthetic(rules_season: str, n: int = 4000, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    pos = rng.choice(["GK", "DEF", "MID", "FWD"], n)
    df = pd.DataFrame(
        {
            "position": pos,
            "minutes": rng.choice([0, 25, 59, 60, 75, 90], n),
            "goals_scored": rng.poisson(0.15, n) * (pos != "GK"),
            "assists": rng.poisson(0.1, n),
            "clean_sheets": rng.binomial(1, 0.3, n),
            "goals_conceded": rng.poisson(1.2, n),
            "saves": rng.poisson(2.0, n) * (pos == "GK"),
            "penalties_saved": rng.binomial(1, 0.05, n) * (pos == "GK"),
            "penalties_missed": rng.binomial(1, 0.02, n),
            "yellow_cards": rng.binomial(1, 0.1, n),
            "red_cards": rng.binomial(1, 0.01, n),
            "own_goals": rng.binomial(1, 0.01, n),
            "key_passes": rng.poisson(1.0, n),
            "shots": rng.poisson(1.0, n),
        }
    )
    weights = pd.Series({**official_weights(load_rules(rules_season)), "goal_GK": 0.0})
    x = design(df)
    df.loc[x.index, "bps"] = (
        x[weights.index].to_numpy() @ weights.to_numpy()
        + 2 * x["key_passes"]
        + rng.normal(0, 0.5, len(x))
    )
    return df


def test_recovers_generating_weights() -> None:
    rules = load_rules("2023/24")
    table = estimate_bps_weights(synthetic("2023/24"), rules).set_index("component")
    identified = table.drop(index=["goal_GK"])
    np.testing.assert_allclose(
        identified["estimated"].iloc[:-2], identified["official"].iloc[:-2], atol=0.3
    )
    assert abs(table.loc["key_passes", "estimated"] - 2.0) < 0.1
    assert table.attrs["r2"] > 0.95


def test_sixty_minutes_counts_as_sixty_plus() -> None:
    df = pd.DataFrame({"position": ["MID"] * 3, "minutes": [59, 60, 0]}).assign(
        goals_scored=0,
        assists=0,
        clean_sheets=0,
        goals_conceded=0,
        saves=0,
        penalties_saved=0,
        penalties_missed=0,
        yellow_cards=0,
        red_cards=0,
        own_goals=0,
    )
    x = design(df)
    assert len(x) == 2  # no row for a player who did not play
    assert x["minutes_lt_60"].tolist() == [1, 0] and x["minutes_gte_60"].tolist() == [0, 1]
