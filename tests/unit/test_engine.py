from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest

from fplh.rules import Rules, score
from fplh.rules.engine import COMPONENTS, DEFENSIVE_COLUMNS, EVENT_COLUMNS, score_arrays

ZERO = dict.fromkeys((*EVENT_COLUMNS, *DEFENSIVE_COLUMNS), 0)


def row(position: str, **events: Any) -> dict[str, Any]:
    return {"position": position, **ZERO, **events}


def total(rules: Rules, **kw: Any) -> int:
    return int(score(pd.DataFrame([row(**kw)]), rules)["total"].iloc[0])


CASES = [
    # (description, row, expected points)
    (
        "GK clean sheet, 7 saves, pen save",
        row("GK", minutes=90, saves=7, penalties_saved=1),
        2 + 4 + 2 + 5,
    ),
    ("GK goal", row("GK", minutes=90, goals_scored=1, goals_conceded=1), 2 + 10),
    ("DEF conceded 5 on pitch", row("DEF", minutes=90, goals_conceded=5), 2 - 2),
    ("DEF subbed at 59 without conceding: no CS", row("DEF", minutes=59), 1),
    ("DEF 60 min clean sheet", row("DEF", minutes=60), 2 + 4),
    (
        "MID goal+assist+CS+DC",
        row("MID", minutes=90, goals_scored=1, assists=1, recoveries=12),
        2 + 5 + 3 + 1 + 2,
    ),
    (
        "FWD sub goal, yellow, DC with recoveries",
        row(
            "FWD",
            minutes=30,
            goals_scored=1,
            yellow_cards=1,
            goals_conceded=1,
            recoveries=6,
            tackles=6,
        ),
        1 + 4 - 1 + 2,
    ),
    (
        "DEF: recoveries do not count",
        row(
            "DEF",
            minutes=90,
            goals_conceded=1,
            clearances_blocks_interceptions=5,
            tackles=4,
            recoveries=10,
        ),
        2,
    ),
    (
        "DEF DC at threshold",
        row("DEF", minutes=90, goals_conceded=1, clearances_blocks_interceptions=6, tackles=4),
        2 + 2,
    ),
    (
        "GK gets no DC points",
        row("GK", minutes=90, goals_conceded=1, clearances_blocks_interceptions=20),
        2,
    ),
    (
        "own goal, pen miss, red",
        row("MID", minutes=90, goals_conceded=2, own_goals=1, penalties_missed=1, red_cards=1),
        2 - 2 - 2 - 3,
    ),
    ("bonus passes through", row("FWD", minutes=90, goals_conceded=1, bonus=3), 2 + 3),
    ("did not play", row("DEF", minutes=0), 0),
    ("FWD CS is worth 0", row("FWD", minutes=90), 2),
]


@pytest.mark.parametrize(("desc", "r", "expected"), CASES, ids=[c[0] for c in CASES])
def test_hand_scored_cases(rules_2627: Rules, desc: str, r: dict[str, Any], expected: int) -> None:
    out = score(pd.DataFrame([r]), rules_2627)
    assert int(out["total"].iloc[0]) == expected, out.T


def test_breakdown_sums_to_total(rules_2627: Rules) -> None:
    df = pd.DataFrame([c[1] for c in CASES])
    out = score(df, rules_2627)
    assert list(out.columns) == [*COMPONENTS, "total"]
    assert (out[list(COMPONENTS)].sum(axis=1) == out["total"]).all()


def test_missing_column_raises(rules_2627: Rules) -> None:
    df = pd.DataFrame([row("MID", minutes=90)]).drop(columns=["saves"])
    with pytest.raises(KeyError, match="saves"):
        score(df, rules_2627)


def test_missing_defensive_columns_raise(rules_2627: Rules) -> None:
    df = pd.DataFrame([row("MID", minutes=90)]).drop(columns=["recoveries"])
    with pytest.raises(KeyError, match="recoveries"):
        score(df, rules_2627)


def test_unknown_position_raises(rules_2627: Rules) -> None:
    with pytest.raises(ValueError, match="AM"):
        score(pd.DataFrame([row("AM", minutes=90)]), rules_2627)


def test_arrays_broadcast_over_simulations(rules_2627: Rules) -> None:
    sims, players = 1000, 3
    rng = np.random.default_rng(0)
    events = {c: np.zeros((sims, players), dtype=int) for c in (*EVENT_COLUMNS, *DEFENSIVE_COLUMNS)}
    events["minutes"] = np.full((sims, players), 90)
    events["goals_scored"] = rng.poisson(0.3, (sims, players))
    events["goals_conceded"] = rng.poisson(1.2, (sims, 1)).repeat(players, axis=1)
    pos = np.broadcast_to(np.array(["GK", "DEF", "FWD"]), (sims, players))
    parts = score_arrays(events, pos, rules_2627)
    assert parts["goals"].shape == (sims, players)
    np.testing.assert_array_equal(parts["goals"][:, 2], events["goals_scored"][:, 2] * 4)
    # clean sheets agree with the shared team scoreline (coherence, P3)
    cs = events["goals_conceded"][:, 0] == 0
    np.testing.assert_array_equal(parts["clean_sheet"][:, 1] == 4, cs)
