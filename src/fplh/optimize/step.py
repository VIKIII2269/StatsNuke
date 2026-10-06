"""One gameweek's decision, shared by the season replay and the live model team.

``players_frame`` assembles the optimiser's input from expected points and game state;
``decide`` solves the week (``plan_week``) and applies the transfers at the deadline's
prices (sell prices by the official formula); ``commit`` moves the state on: chips used,
the free hit's squad reverting, and the next week's free transfers.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from fplh.features.prices import sell_price
from fplh.optimize.milp import Plan, SquadRules, State, plan_week


@dataclass
class Decision:
    plan: Plan
    squad: dict[str, int]  # the squad this gameweek, with purchase prices (£0.1m)
    bank: int


def players_frame(
    expected: pd.DataFrame,
    position: pd.Series,
    team: pd.Series,
    price: pd.Series,
    state: State,
) -> pd.DataFrame:
    """index player_uid: position, team, price and the E{g} columns, for every player with
    a forecast or in the squad (blank weeks are 0); unpriced non-squad players drop out."""
    ids = sorted(set(expected.index) | set(state.squad))
    players = pd.DataFrame(index=pd.Index(ids, name="player_uid"))
    players["position"] = position.reindex(ids)
    players["team"] = team.reindex(ids)
    players["price"] = price.reindex(ids)
    players = players.join(expected)
    for c in expected.columns:
        players[c] = players[c].fillna(0.0)
    players = players.dropna(subset=["position", "team", "price"])
    out: pd.DataFrame = players[players.index.isin(state.squad) | (players["price"] > 0)]
    return out


def decide(
    players: pd.DataFrame,
    state: State,
    rules: SquadRules,
    horizon: list[int],
    price: pd.Series,
    *,
    delta: float = 0.9,
    beta: float = 0.1,
    chip_cost: dict[str, float] | None = None,
    time_limit: float = 30.0,
) -> Decision:
    """Solve the week and apply its transfers at this deadline's prices."""
    state.gameweek = horizon[0]
    plan = plan_week(
        players,
        state,
        rules,
        horizon,
        delta=delta,
        beta=beta,
        chip_cost=chip_cost,
        time_limit=time_limit,
    )
    bank = state.bank
    squad = dict(state.squad)
    for p in plan.sells:
        bank += sell_price(squad.pop(p), int(price[p]))
    for p in plan.buys:
        squad[p] = int(price[p])
        bank -= int(price[p])
    if bank < 0:
        raise RuntimeError(f"GW{horizon[0]}: negative bank {bank}")
    return Decision(plan, squad, bank)


def commit(state: State, decision: Decision, gw: int) -> None:
    """The state after the gameweek: a free-hit squad reverts, its transfers not kept."""
    plan = decision.plan
    if plan.chip:
        state.chips_used.setdefault(plan.chip, []).append(gw)
    if plan.chip != "free_hit":
        state.squad, state.bank = decision.squad, decision.bank
    state.free_transfers = plan.free_transfers_next
