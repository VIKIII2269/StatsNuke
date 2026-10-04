"""The squad MILP (ticket 4.1): brute-force optima on small games, and each constraint
holding where the unconstrained optimum would break it."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fplh.optimize.milp import (
    SquadRules,
    State,
    brute_force_single_week,
    next_free_transfers,
    optimise,
    plan_week,
    remaining_windows,
)
from fplh.rules.config import load_rules

TOY = SquadRules(
    squad={"GK": 1, "DEF": 2, "MID": 2, "FWD": 1},
    xi_min={"GK": 1, "DEF": 1, "MID": 1, "FWD": 1},
    xi_size=4,
    max_per_club=2,
    hit_cost=4,
    ft_cap=2,
    chips_preserve_transfers=False,
)
NO_CHIPS: dict[str, float] = {}


def players(seed: int, n_per: int = 3, gws: tuple[int, ...] = (2,)) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for pos, count in (("GK", n_per), ("DEF", n_per + 1), ("MID", n_per + 1), ("FWD", n_per)):
        for j in range(count):
            row = {
                "player_uid": f"{pos}{j}",
                "position": pos,
                "team": f"t{rng.integers(0, 6)}",
                "price": int(rng.integers(40, 90)),
            }
            row.update({f"E{g}": float(rng.uniform(0, 8)) for g in gws})
            rows.append(row)
    return pd.DataFrame(rows).set_index("player_uid")


def fresh(budget: int, gw: int = 2) -> State:
    return State(gameweek=gw, squad={}, bank=budget, free_transfers=15)


@pytest.mark.parametrize("seed", range(6))
def test_single_week_matches_brute_force(seed: int) -> None:
    p = players(seed)
    cheapest = sum(
        p[p["position"] == q].nsmallest(n, "price")["price"].sum() for q, n in TOY.squad.items()
    )
    budget = int(cheapest) + 60  # binding for some seeds, never infeasible
    plan = optimise(
        p, fresh(budget), TOY, [2], beta=0.1, chip_cost=NO_CHIPS, pool_size=50, transfer_penalty=0
    )[0]
    assert plan.objective == pytest.approx(brute_force_single_week(p, TOY, budget, 2), abs=1e-6)
    # composition, clubs, budget and XI validity of the returned plan
    pos = p["position"]
    assert {q: int((pos[plan.squad] == q).sum()) for q in TOY.squad} == dict(TOY.squad)
    assert p.loc[plan.squad, "team"].value_counts().max() <= TOY.max_per_club
    assert p.loc[plan.squad, "price"].sum() <= budget
    assert len(plan.xi) == 4 and (pos[plan.xi] == "GK").sum() == 1
    assert plan.captain in plan.xi and plan.captain == max(plan.xi, key=lambda u: p.loc[u, "E2"])


def test_club_limit_binds() -> None:
    p = players(1)
    p.loc[["DEF0", "DEF1", "MID0"], "team"] = "big"
    p.loc[["DEF0", "DEF1", "MID0"], "E2"] = 50.0
    plan = optimise(p, fresh(1000), TOY, [2], chip_cost=NO_CHIPS)[0]
    assert sum(u in plan.squad for u in ("DEF0", "DEF1", "MID0")) == 2


def test_budget_binds() -> None:
    p = players(2)
    p.loc["FWD0", ["price", "E2"]] = [200, 30.0]
    others = p.drop(index="FWD0")
    cheapest = sum(
        others[others["position"] == q].nsmallest(n, "price")["price"].sum()
        for q, n in TOY.squad.items()
    )
    cheap = optimise(p, fresh(int(cheapest) + 20), TOY, [2], chip_cost=NO_CHIPS)[0]
    rich = optimise(p, fresh(2000), TOY, [2], chip_cost=NO_CHIPS)[0]
    assert "FWD0" not in cheap.squad and "FWD0" in rich.squad


def owned_state(p: pd.DataFrame, squad: list[str], ft: int, gw: int = 2, bank: int = 0) -> State:
    return State(gw, {u: int(p.loc[u, "price"]) for u in squad}, bank, ft)


BASE = ["GK0", "DEF0", "DEF1", "MID0", "MID1", "FWD0"]


def test_transfers_use_free_transfers_then_hits() -> None:
    p = players(3)
    p["team"] = [f"t{i}" for i in range(len(p))]
    p["price"] = 50
    p[["E2"]] = 1.0
    p.loc["MID2", "E2"] = 9.0  # worth a free transfer
    plan = optimise(p, owned_state(p, BASE, 1), TOY, [2], chip_cost=NO_CHIPS)[0]
    assert plan.buys == ["MID2"] and plan.hits == 0
    p.loc["DEF2", "E2"] = 9.0  # a second upgrade is worth −4
    plan = optimise(p, owned_state(p, BASE, 1), TOY, [2], chip_cost=NO_CHIPS)[0]
    assert set(plan.buys) == {"MID2", "DEF2"} and plan.hits == 1
    p.loc["DEF2", "E2"] = 3.5  # +2.5 in the XI or +0.25 on the bench: not worth −4
    plan = optimise(p, owned_state(p, BASE, 1), TOY, [2], chip_cost=NO_CHIPS)[0]
    assert plan.buys == ["MID2"] and plan.hits == 0


def test_free_transfers_bank_up_to_the_cap() -> None:
    p = players(4)
    p["team"] = [f"t{i}" for i in range(len(p))]
    p["price"] = 50
    p["E2"] = 1.0
    plan = optimise(p, owned_state(p, BASE, 1), TOY, [2], chip_cost=NO_CHIPS)[0]
    assert plan.buys == [] and plan.free_transfers_next == 2
    plan = optimise(p, owned_state(p, BASE, 2), TOY, [2], chip_cost=NO_CHIPS)[0]
    assert plan.free_transfers_next == 2  # capped
    p.loc["MID2", "E2"] = 9.0
    plan = optimise(p, owned_state(p, BASE, 2), TOY, [2], chip_cost=NO_CHIPS)[0]
    assert plan.free_transfers_next == 2  # 2 − 1 + 1


def test_sell_price_limits_what_can_be_bought() -> None:
    p = players(5)
    p["team"] = [f"t{i}" for i in range(len(p))]
    p["E2"] = 1.0
    p["price"] = 50
    p.loc["MID0", "price"] = 54  # bought at 50: sells for 52, not 54
    p.loc["MID2", ["price", "E2"]] = [53, 9.0]
    s = owned_state(p, BASE, 1)
    s.squad["MID0"] = 50
    plan = optimise(p, s, TOY, [2], chip_cost=NO_CHIPS)[0]
    assert "MID2" not in plan.buys  # 52 < 53
    s.bank = 1
    plan = optimise(p, s, TOY, [2], chip_cost=NO_CHIPS)[0]
    assert plan.buys == ["MID2"] and plan.sells == ["MID0"]


def with_chips(**windows: list[tuple[int, int]]) -> SquadRules:
    return SquadRules(**{**TOY.__dict__, "chip_windows": windows})


def test_free_hit_keeps_the_regular_squad() -> None:
    p = players(6, gws=(2, 3))
    p["team"] = [f"t{i}" for i in range(len(p))]
    p["price"] = 50
    p[["E2", "E3"]] = 1.0
    p.loc[BASE, "E3"] = 6.0  # the owned squad is good next week …
    others = [u for u in p.index if u not in BASE]
    p.loc[others, "E2"] = 8.0  # … everyone else is much better this week
    rules = with_chips(free_hit=[(2, 2)])
    plan = optimise(p, owned_state(p, BASE, 1), rules, [2, 3], chip_cost=NO_CHIPS)[0]
    assert plan.chip == "free_hit"
    assert set(plan.squads[0]) == set(BASE) and set(plan.squads[1]) == set(BASE)
    assert not set(plan.squad) & set(BASE) or set(plan.squad) != set(BASE)
    assert plan.buys == [] and plan.hits == 0


def test_wildcard_waives_hits_and_windows_are_respected() -> None:
    p = players(7)
    p["team"] = [f"t{i}" for i in range(len(p))]
    p["price"] = 50
    p["E2"] = 1.0
    others = [u for u in p.index if u not in BASE]
    p.loc[others, "E2"] = 9.0
    plan = optimise(
        p, owned_state(p, BASE, 1), with_chips(wildcard=[(2, 2)]), [2], chip_cost=NO_CHIPS
    )[0]
    assert plan.chip == "wildcard" and plan.hits == 0 and len(plan.buys) == 6
    # outside its window the wildcard is not available: hits instead
    plan = optimise(
        p, owned_state(p, BASE, 1), with_chips(wildcard=[(5, 9)]), [2], chip_cost=NO_CHIPS
    )[0]
    assert plan.chip is None and plan.hits > 0
    # a window that outlasts the horizon carries the opportunity cost
    plan = optimise(
        p,
        owned_state(p, BASE, 1),
        with_chips(wildcard=[(2, 9)]),
        [2],
        chip_cost={"wildcard": 500.0},
    )[0]
    assert plan.chip is None


def test_bench_boost_and_triple_captain() -> None:
    p = players(8)
    p["team"] = [f"t{i}" for i in range(len(p))]
    p["price"] = 50
    p["E2"] = 1.0
    p.loc[BASE, "E2"] = 5.0
    plan = optimise(
        p, owned_state(p, BASE, 1), with_chips(bench_boost=[(2, 2)]), [2], chip_cost=NO_CHIPS
    )[0]
    assert plan.chip == "bench_boost"
    assert plan.expected_points == pytest.approx(6 * 5.0 + 5.0)
    plan = optimise(
        p, owned_state(p, BASE, 1), with_chips(triple_captain=[(2, 2)]), [2], chip_cost=NO_CHIPS
    )[0]
    assert plan.chip == "triple_captain"
    assert plan.expected_points == pytest.approx(4 * 5.0 + 2 * 5.0)


def test_special_free_transfers_and_preserved_banking() -> None:
    p = players(9)
    p["team"] = [f"t{i}" for i in range(len(p))]
    p["price"] = 50
    p["E2"] = 1.0
    for u in ("DEF2", "MID2", "FWD1"):
        p.loc[u, "E2"] = 9.0
    special = SquadRules(**{**TOY.__dict__, "special_free_transfers": {2: 5}})
    plan = optimise(p, owned_state(p, BASE, 1), special, [2], chip_cost=NO_CHIPS)[0]
    assert len(plan.buys) == 3 and plan.hits == 0
    others = [u for u in p.index if u not in BASE]
    p.loc[others, "E2"] = 9.0
    keep = SquadRules(
        **{**TOY.__dict__, "chip_windows": {"wildcard": [(2, 2)]}, "chips_preserve_transfers": True}
    )
    plan = optimise(p, owned_state(p, BASE, 2), keep, [2], chip_cost=NO_CHIPS)[0]
    assert plan.chip == "wildcard" and plan.free_transfers_next == 2
    lose = SquadRules(**{**keep.__dict__, "chips_preserve_transfers": False})
    plan = optimise(p, owned_state(p, BASE, 2), lose, [2], chip_cost=NO_CHIPS)[0]
    assert plan.chip == "wildcard" and plan.free_transfers_next == 1


def test_top_k_plans_are_distinct_and_ordered() -> None:
    p = players(10)
    plans = optimise(p, owned_state(p, BASE, 1, bank=100), TOY, [2], chip_cost=NO_CHIPS, top_k=3)
    assert len(plans) == 3
    assert plans[0].objective >= plans[1].objective >= plans[2].objective
    firsts = {(tuple(sorted(x.buys)), tuple(sorted(x.sells)), x.chip) for x in plans}
    assert len(firsts) == 3


def test_next_free_transfers_rules() -> None:
    cap2 = SquadRules(**{**TOY.__dict__, "ft_cap": 2})
    assert next_free_transfers(cap2, 1, 0, None, 3) == 2
    assert next_free_transfers(cap2, 2, 0, None, 3) == 2
    assert next_free_transfers(cap2, 2, 1, None, 3) == 2
    assert next_free_transfers(cap2, 1, 3, None, 3) == 1  # hits taken
    assert next_free_transfers(cap2, 2, 6, "wildcard", 3) == 1
    keep = SquadRules(**{**TOY.__dict__, "ft_cap": 5, "chips_preserve_transfers": True})
    assert next_free_transfers(keep, 4, 9, "free_hit", 3) == 4
    assert next_free_transfers(keep, 15, 15, None, 3) == 1  # after an unlimited week
    special = SquadRules(**{**keep.__dict__, "special_free_transfers": {16: 5}})
    assert next_free_transfers(special, 1, 0, None, 16) == 5


def test_remaining_windows_and_real_rules() -> None:
    rules = SquadRules.from_rules(load_rules("2023/24"))
    assert rules.ft_cap == 2 and not rules.chips_preserve_transfers
    left = remaining_windows(rules, State(25, {}, 0, 1, {"wildcard": [8]}))
    assert left["wildcard"] == [(21, 38)] and left["free_hit"] == [(2, 38)]
    assert SquadRules.from_rules(load_rules("2025/26")).special_free_transfers == {16: 5}


def test_plan_week_holds_or_plays_chips() -> None:
    p = players(11, gws=(2, 3, 4))
    p["team"] = [f"t{i}" for i in range(len(p))]
    p["price"] = 50
    p[["E2", "E3", "E4"]] = 1.0
    p.loc[BASE, ["E2", "E3", "E4"]] = 4.0
    s = owned_state(p, BASE, 1)
    # bench boost window ends inside the horizon (no opportunity cost): the best week wins
    rules = with_chips(bench_boost=[(2, 4)])
    p.loc[BASE, "E4"] = 9.0  # a double gameweek for the whole squad: bench worth more then
    plan = plan_week(p, s, rules, [2, 3, 4], chip_cost={})
    assert plan.chip is None  # held for GW4
    p.loc[BASE, ["E2", "E4"]] = [9.0, 4.0]  # the double gameweek is now
    plan = plan_week(p, s, rules, [2, 3, 4], chip_cost={})
    assert plan.chip == "bench_boost"
    # a window that outlasts the horizon needs a gain above its opportunity cost
    keep = with_chips(bench_boost=[(2, 30)])
    plan = plan_week(p, s, keep, [2, 3, 4], chip_cost={"bench_boost": 1000.0})
    assert plan.chip is None


def test_plan_week_free_hit_and_wildcard() -> None:
    p = players(12, gws=(2, 3))
    p["team"] = [f"t{i}" for i in range(len(p))]
    p["price"] = 50
    p[["E2", "E3"]] = 1.0
    p.loc[BASE, "E3"] = 6.0
    others = [u for u in p.index if u not in BASE]
    p.loc[others, "E2"] = 8.0
    s = owned_state(p, BASE, 1)
    plan = plan_week(p, s, with_chips(free_hit=[(2, 3)]), [2, 3], chip_cost={})
    assert plan.chip == "free_hit" and plan.buys == []
    p.loc[others, ["E2", "E3"]] = 8.0  # everyone else is better for good: wildcard
    plan = plan_week(p, s, with_chips(free_hit=[(2, 3)], wildcard=[(2, 3)]), [2, 3], chip_cost={})
    assert plan.chip == "wildcard" and plan.hits == 0 and len(plan.buys) >= 5


def test_owned_players_must_be_priced() -> None:
    p = players(13)
    s = owned_state(p, BASE, 1)
    with pytest.raises(ValueError, match="owned players"):
        optimise(p.drop(index="MID0"), s, TOY, [2], chip_cost=NO_CHIPS)
