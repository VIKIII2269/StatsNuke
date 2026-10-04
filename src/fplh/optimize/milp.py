"""Multi-gameweek FPL squad optimiser (ARCHITECTURE.md §10.1), a MILP solved with HiGHS
through ``scipy.optimize.milp``.

Variables per pool player p and horizon week t: squad x, starting XI y, captain k, buy b,
sell s, free-hit squad xf, bench u (continuous), bench-boost z, triple-captain bonus kt;
per week: hits h, free transfers F, an "overdrawn" binary o, and the chips w (wildcard),
f (free hit), bb (bench boost), tc (triple captain).

    max Σ_t δ^(t−1) [ Σ_p E_pt (y + k + kt) + β Σ_p E_pt u + (1 − β) Σ_p E_pt z − hit·h_t ]
        − Σ chip opportunity costs

Constraints (all t):

1. composition per position and at most ``max_per_club`` per club (regular and free-hit
   squads);
2. a valid XI from the squad of the week (the free-hit squad when f_t = 1), ``xi_size``
   players with one goalkeeper and the per-position minimums;
3. one captain in the XI;
4. squad flow x_t = x_{t−1} + b_t − s_t; no transfers in a free-hit week (the regular squad
   carries over unchanged);
5. budget: bank_t = bank_{t−1} + Σ sell·s − Σ price·b ≥ 0 (sell prices of owned players by
   the FPL formula; later weeks at current prices); the free-hit squad costs at most the
   bank plus the squad's sell value;
6. hits h_t ≥ Σ_p b_pt − F_t, waived in wildcard and free-hit weeks;
7. free-transfer banking F_{t+1} = min(cap, max(F_t − Σ b_t, 0) + 1), linearised with the
   binary o_t (the optimiser maximises F, so upper bounds suffice); in seasons where chips
   preserve banked transfers, F_{t+1} = F_t after a wildcard or free hit; special weeks fix F;
8. chips: at most one per week, each remaining copy at most once within its gameweek window.

A tiny penalty per transfer (``transfer_penalty``) breaks ties against like-for-like
churn, which in the game costs the banked transfer it uses.

Chips are valued only within the horizon, so each chip whose window outlasts the horizon
carries an opportunity cost (points it is expected to be worth later); a chip that expires
within the horizon carries none (use it or lose it).
"""

from __future__ import annotations

import itertools
from collections.abc import Mapping
from dataclasses import dataclass, field, replace

import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix

from fplh.rules.config import Rules

CHIP_NAMES = ("wildcard", "free_hit", "bench_boost", "triple_captain")
# Points a chip copy is expected to be worth if kept (a priori, the same for every strategy):
# roughly its value in a double or blank gameweek rather than an ordinary one.
DEFAULT_CHIP_COST = {
    "wildcard": 20.0,
    "free_hit": 15.0,
    "bench_boost": 15.0,
    "triple_captain": 10.0,
}


@dataclass(frozen=True)
class SquadRules:
    squad: Mapping[str, int]
    xi_min: Mapping[str, int]
    xi_size: int = 11
    max_per_club: int = 3
    hit_cost: int = 4
    ft_cap: int = 5
    chips_preserve_transfers: bool = True
    chip_windows: Mapping[str, list[tuple[int, int]]] = field(default_factory=dict)
    special_free_transfers: Mapping[int, int] = field(default_factory=dict)

    @classmethod
    def from_rules(cls, rules: Rules) -> SquadRules:
        g = rules.game
        return cls(
            {str(q): int(n) for q, n in g.squad.items()},
            {str(q): int(n) for q, n in g.xi_min.items()},
            11,
            g.max_per_club,
            g.hit_cost,
            g.free_transfer_bank_max,
            bool({"wildcard", "free_hit"} & set(g.free_transfers_preserved_on)),
            g.chips.allowance(),
            dict(g.special_free_transfers),
        )


@dataclass
class State:
    gameweek: int  # the gameweek being decided
    squad: dict[str, int]  # owned player → purchase price (£0.1m)
    bank: int  # £0.1m
    free_transfers: int
    chips_used: dict[str, list[int]] = field(default_factory=dict)  # chip → gameweeks played


@dataclass
class Plan:
    objective: float
    gameweek: int
    chip: str | None
    buys: list[str]
    sells: list[str]
    squad: list[str]  # the squad that plays this week (the free-hit squad if f = 1)
    xi: list[str]
    bench: list[str]  # substitute goalkeeper first, then outfield by expected points
    captain: str
    vice: str
    hits: int
    free_transfers_next: int
    expected_points: float  # this week's XI + captain (+ bench boost / triple captain)
    squads: list[list[str]] = field(default_factory=list)  # regular squad per horizon week
    xis: list[list[str]] = field(default_factory=list)  # XI per horizon week
    captains: list[str] = field(default_factory=list)  # captain per horizon week


class _Model:
    """Variables by name, linear rows appended; maximised with HiGHS."""

    def __init__(self) -> None:
        self.lb: list[float] = []
        self.ub: list[float] = []
        self.integer: list[int] = []
        self.c: list[float] = []
        self.rows: list[tuple[dict[int, float], float, float]] = []

    def var(self, lb: float = 0.0, ub: float = 1.0, integer: bool = True, obj: float = 0.0) -> int:
        self.lb.append(lb)
        self.ub.append(ub)
        self.integer.append(1 if integer else 0)
        self.c.append(obj)
        return len(self.c) - 1

    def add(self, coefs: Mapping[int, float], lo: float = -np.inf, hi: float = np.inf) -> None:
        self.rows.append((dict(coefs), lo, hi))

    def solve(self, time_limit: float, gap: float) -> tuple[np.ndarray, float] | None:
        r, cidx, vals = [], [], []
        for i, (coefs, _, _) in enumerate(self.rows):
            for j, v in coefs.items():
                r.append(i)
                cidx.append(j)
                vals.append(v)
        a = coo_matrix((vals, (r, cidx)), shape=(len(self.rows), len(self.c))).tocsr()
        lo = np.array([row[1] for row in self.rows])
        hi = np.array([row[2] for row in self.rows])
        res = milp(
            -np.asarray(self.c),
            constraints=LinearConstraint(a, lo, hi),
            integrality=np.asarray(self.integer),
            bounds=Bounds(np.asarray(self.lb), np.asarray(self.ub)),
            options={"time_limit": time_limit, "mip_rel_gap": gap, "presolve": True},
        )
        if res.x is None:
            return None
        return np.round(res.x, 6), float(-res.fun)


def next_free_transfers(
    rules: SquadRules, free: int, transfers: int, chip: str | None, next_gw: int
) -> int:
    """Free transfers for the next gameweek after making ``transfers`` with ``free``."""
    if next_gw in rules.special_free_transfers:
        return rules.special_free_transfers[next_gw]
    if chip in ("wildcard", "free_hit"):
        return min(rules.ft_cap, free) if rules.chips_preserve_transfers else 1
    return min(rules.ft_cap, max(min(free, rules.ft_cap) - transfers, 0) + 1)


def remaining_windows(rules: SquadRules, state: State) -> dict[str, list[tuple[int, int]]]:
    """Per chip, the windows of copies not yet played (a play consumes its own window)."""
    out: dict[str, list[tuple[int, int]]] = {}
    for chip, windows in rules.chip_windows.items():
        left = list(windows)
        for gw in state.chips_used.get(chip, []):
            for w in left:
                if w[0] <= gw <= w[1]:
                    left.remove(w)
                    break
        out[chip] = left
    return out


def pool(
    players: pd.DataFrame,
    horizon: list[int],
    state: State,
    rules: SquadRules,
    size: int,
    delta: float,
) -> pd.DataFrame:
    """Owned players, the ``size`` best per position by discounted horizon points, and the
    cheapest few per position (so a budget-feasible squad always exists)."""
    weights = np.array([delta**i for i in range(len(horizon))])
    value = players[[f"E{t}" for t in horizon]].to_numpy(float) @ weights
    p = players.assign(_value=value)
    keep = set(state.squad) & set(p.index)
    for pos in rules.squad:
        grp = p[p["position"] == pos]
        keep |= set(grp.nlargest(size, "_value").index)
        keep |= set(grp.nsmallest(rules.squad[pos] + 2, "price").index)
    out: pd.DataFrame = p.loc[sorted(keep)].drop(columns="_value")
    return out


def optimise(
    players: pd.DataFrame,
    state: State,
    rules: SquadRules,
    horizon: list[int],
    *,
    delta: float = 0.9,
    beta: float = 0.1,
    pool_size: int = 25,
    chip_cost: Mapping[str, float] | None = None,
    top_k: int = 1,
    transfer_penalty: float = 0.01,
    force_chip: tuple[str, int] | None = None,
    time_limit: float = 60.0,
    gap: float = 5e-4,
) -> list[Plan]:
    """``players``: index player_uid with position, team, price (£0.1m, current) and one
    column ``E{gw}`` of expected points per horizon gameweek (0 in a blank). Returns up to
    ``top_k`` plans, best first, differing in this week's transfers or chip.
    ``force_chip=(chip, gameweek)`` plays that chip then and no other (no opportunity cost)."""
    costs = dict(DEFAULT_CHIP_COST if chip_cost is None else chip_cost)
    missing = [u for u in state.squad if u not in players.index or pd.isna(players.loc[u, "price"])]
    if missing:  # an owned player must stay representable (sold or kept), with a price
        raise ValueError(f"owned players without a row or price: {missing}")
    if force_chip is not None:
        rules = replace(rules, chip_windows={force_chip[0]: [(force_chip[1], force_chip[1])]})
        costs = {}
    players = pool(players, horizon, state, rules, pool_size, delta)
    uids = list(players.index)
    n, hz = len(uids), len(horizon)
    pos = players["position"].to_numpy(str)
    team = players["team"].to_numpy(str)
    price = players["price"].to_numpy(float)
    owned = np.array([u in state.squad for u in uids])
    sell = np.array(
        [
            (state.squad[u] + max(price[i] - state.squad[u], 0) // 2)
            if owned[i] and price[i] > state.squad[u]
            else price[i]
            for i, u in enumerate(uids)
        ]
    )
    exp = players[[f"E{t}" for t in horizon]].to_numpy(float)
    windows = remaining_windows(rules, state)

    avail = {
        c: [any(lo <= gw <= hi for lo, hi in windows.get(c, [])) for gw in horizon]
        for c in CHIP_NAMES
    }
    big_t = 15.0  # bounds every transfer and free-transfer count
    big_fh = float(state.bank + sell[owned].sum() + price.max() * sum(rules.squad.values()))

    m = _Model()
    none = -1
    x, y, k, b, s, u = (np.full((n, hz), none) for _ in range(6))
    xf, z, kt = (np.full((n, hz), none) for _ in range(3))  # only where the chip is playable
    h, over = np.full(hz, none), np.full(hz, none)
    bank = np.full(hz, none)
    chip = {c: np.full(hz, none) for c in CHIP_NAMES}
    for t in range(hz):
        disc = delta**t
        for i in range(n):
            e = exp[i, t]
            x[i, t] = m.var()
            y[i, t] = m.var(obj=disc * e)
            k[i, t] = m.var(obj=disc * e)
            b[i, t] = m.var(obj=-transfer_penalty)  # breaks ties against pointless churn
            s[i, t] = m.var()
            u[i, t] = m.var(integer=False, obj=disc * beta * e)
            if avail["free_hit"][t]:
                xf[i, t] = m.var()
            if avail["bench_boost"][t]:
                z[i, t] = m.var(integer=False, obj=disc * (1 - beta) * e)
            if avail["triple_captain"][t]:
                kt[i, t] = m.var(integer=False, obj=disc * e)
        h[t] = m.var(ub=big_t, obj=-disc * rules.hit_cost)
        over[t] = m.var()
        bank[t] = m.var(lb=0, ub=big_fh, integer=False)
        for c in CHIP_NAMES:
            if avail[c][t]:
                forced = force_chip is not None and force_chip == (c, horizon[t])
                chip[c][t] = m.var(lb=1.0 if forced else 0.0)
    # free transfers now (15 = unlimited, e.g. the initial squad); special weeks fix the count
    f0 = rules.special_free_transfers.get(horizon[0], state.free_transfers)
    ft = np.full(hz + 1, none)
    for t in range(hz + 1):
        gw = horizon[t] if t < hz else horizon[-1] + 1
        special = rules.special_free_transfers.get(gw)
        cap = max(rules.ft_cap, special or 0, f0 if t == 0 else 0)
        ft[t] = m.var(lb=1, ub=cap)
    m.add({int(ft[0]): 1}, f0, f0)
    for t, gw in enumerate(horizon):
        if gw in rules.special_free_transfers and t > 0:
            fixed = rules.special_free_transfers[gw]
            m.add({int(ft[t]): 1}, fixed, fixed)

    def terms(*pairs: tuple[int, float]) -> dict[int, float]:
        """Coefficients, skipping variables that do not exist this week."""
        return {int(v): c for v, c in pairs if v != none}

    for t in range(hz):
        wc, fh, bb, tc = (int(chip[c][t]) for c in CHIP_NAMES)
        chips_t = terms((wc, 1), (fh, 1), (bb, 1), (tc, 1))
        if len(chips_t) > 1:
            m.add(chips_t, hi=1)
        # 1. composition and clubs, regular and free-hit squads
        for q, nq in rules.squad.items():
            idx = np.flatnonzero(pos == q)
            m.add({int(x[i, t]): 1 for i in idx}, nq, nq)
            if fh != none:
                m.add({**{int(xf[i, t]): 1 for i in idx}, fh: -nq}, 0, 0)
        for c in np.unique(team):
            idx = np.flatnonzero(team == c)
            m.add({int(x[i, t]): 1 for i in idx}, hi=rules.max_per_club)
            if fh != none:
                m.add({**{int(xf[i, t]): 1 for i in idx}, fh: -rules.max_per_club}, hi=0)
        # 2. XI from the week's squad (the free-hit squad in a free-hit week)
        m.add({int(y[i, t]): 1 for i in range(n)}, rules.xi_size, rules.xi_size)
        for q, mn in rules.xi_min.items():
            idx = np.flatnonzero(pos == q)
            if q == "GK":
                m.add({int(y[i, t]): 1 for i in idx}, 1, 1)
            else:
                m.add({int(y[i, t]): 1 for i in idx}, lo=mn)
        for i in range(n):
            xi_, yi, ui = int(x[i, t]), int(y[i, t]), int(u[i, t])
            if fh == none:
                m.add({yi: 1, xi_: -1}, hi=0)
                m.add({ui: 1, xi_: -1, yi: 1}, hi=0)  # bench: squad but not XI
            else:
                xfi = int(xf[i, t])
                m.add({yi: 1, xi_: -1, fh: -1}, hi=0)
                m.add({yi: 1, xfi: -1, fh: 1}, hi=1)
                m.add({ui: 1, xi_: -1, yi: 1, fh: -1}, hi=0)
                m.add({ui: 1, xfi: -1, yi: 1, fh: 1}, hi=1)
            if bb != none:
                m.add({int(z[i, t]): 1, ui: -1}, hi=0)
                m.add({int(z[i, t]): 1, bb: -1}, hi=0)
            # 3. captain (and the triple-captain bonus)
            m.add({int(k[i, t]): 1, yi: -1}, hi=0)
            if tc != none:
                m.add({int(kt[i, t]): 1, int(k[i, t]): -1}, hi=0)
                m.add({int(kt[i, t]): 1, tc: -1}, hi=0)
        m.add({int(k[i, t]): 1 for i in range(n)}, 1, 1)
        # 4. squad flow; no transfers in a free-hit week
        for i in range(n):
            prev = {} if t == 0 else {int(x[i, t - 1]): -1}
            rhs = (1.0 if owned[i] else 0.0) if t == 0 else 0.0
            m.add({int(x[i, t]): 1, **prev, int(b[i, t]): -1, int(s[i, t]): 1}, rhs, rhs)
            m.add({int(b[i, t]): 1, int(s[i, t]): 1}, hi=1)
        if fh != none:
            m.add({**{int(b[i, t]): 1 for i in range(n)}, fh: big_t}, hi=big_t)
        # 5. budget
        prev_bank = {} if t == 0 else {int(bank[t - 1]): -1}
        rhs = float(state.bank) if t == 0 else 0.0
        m.add(
            {
                int(bank[t]): 1,
                **prev_bank,
                **{int(s[i, t]): -float(sell[i]) for i in range(n)},
                **{int(b[i, t]): float(price[i]) for i in range(n)},
            },
            rhs,
            rhs,
        )
        if fh != none:  # the free-hit squad costs at most the bank plus the squad's value
            squad_value = {int(x[i, t - 1]): -float(sell[i]) for i in range(n)} if t > 0 else {}
            const = float(state.bank + sell[owned].sum()) if t == 0 else 0.0
            m.add(
                {
                    **{int(xf[i, t]): float(price[i]) for i in range(n)},
                    **squad_value,
                    **prev_bank,
                    fh: big_fh,
                },
                hi=const + big_fh,
            )
        # 6. hits, waived with a wildcard or free hit
        transfers = {int(b[i, t]): 1.0 for i in range(n)}
        m.add(
            {
                int(h[t]): 1,
                **{j: -1.0 for j in transfers},
                int(ft[t]): 1,
                **terms((wc, big_t), (fh, big_t)),
            },
            lo=0,
        )
        # 7. free-transfer banking
        nxt = int(ft[t + 1])
        nxt_gw = horizon[t + 1] if t + 1 < hz else horizon[-1] + 1
        if nxt_gw in rules.special_free_transfers:
            continue  # fixed above (or beyond the horizon)
        m.add({nxt: 1}, hi=rules.ft_cap)
        o = int(over[t])
        if rules.chips_preserve_transfers:
            m.add(
                {
                    nxt: 1,
                    int(ft[t]): -1,
                    **transfers,
                    o: -big_t,
                    **terms((wc, -big_t), (fh, -big_t)),
                },
                hi=1,
            )
            if wc != none or fh != none:  # kept, no accrual
                m.add({nxt: 1, int(ft[t]): -1, **terms((wc, big_t), (fh, big_t))}, hi=big_t)
        else:
            m.add({nxt: 1, int(ft[t]): -1, **transfers, o: -big_t}, hi=1)
        m.add({nxt: 1, o: big_t}, hi=1 + big_t)

    # 8. each remaining chip copy at most once in its window; opportunity costs
    end = horizon[-1]
    for c in CHIP_NAMES:
        for lo, hi in windows.get(c, []):
            ts = [t for t, gw in enumerate(horizon) if lo <= gw <= hi and chip[c][t] != none]
            if not ts:
                continue
            m.add({int(chip[c][t]): 1 for t in ts}, hi=1)
            if hi > end:  # the copy would still be playable after the horizon
                for t in ts:
                    m.c[int(chip[c][t])] -= costs.get(c, 0.0)

    plans: list[Plan] = []
    first = [int(b[i, 0]) for i in range(n)] + [int(s[i, 0]) for i in range(n)]
    first += [int(chip[c][0]) for c in CHIP_NAMES if chip[c][0] != none]
    for _ in range(top_k):
        sol = m.solve(time_limit, gap)
        if sol is None:
            break
        v, obj = sol
        plan = _plan(v, obj, uids, pos, exp, horizon, x, y, k, b, s, xf, chip, h)
        plan.free_transfers_next = next_free_transfers(
            rules, f0, len(plan.buys), plan.chip, horizon[0] + 1
        )
        plans.append(plan)
        ones = [j for j in first if v[j] > 0.5]
        zeros = [j for j in first if v[j] <= 0.5]
        m.add({**{j: 1.0 for j in ones}, **{j: -1.0 for j in zeros}}, hi=len(ones) - 1)
    return plans


def plan_week(
    players: pd.DataFrame,
    state: State,
    rules: SquadRules,
    horizon: list[int],
    *,
    delta: float = 0.9,
    beta: float = 0.1,
    chip_cost: Mapping[str, float] | None = None,
    pool_size: int = 25,
    transfer_penalty: float = 0.01,
    time_limit: float = 60.0,
    gap: float = 5e-4,
) -> Plan:
    """This week's decision with chips decomposed (the joint chip model's LP relaxation is
    weak: an 85 % gap after 30 s on a real week, against optimality in about 1 s without).

    1. the plan without chips;
    2. bench boost and triple captain valued in each horizon week from that plan (bench and
       captain points), the free hit by a solve with it forced in each week, the wildcard
       by a solve with it forced now;
    3. a chip is played now only if its best week is this week and its gain beats the
       opportunity cost of a copy that would outlast the horizon.
    """
    costs = dict(DEFAULT_CHIP_COST if chip_cost is None else chip_cost)
    windows = remaining_windows(rules, state)

    def solve(r: SquadRules, force: tuple[str, int] | None = None) -> list[Plan]:
        return optimise(
            players,
            state,
            r,
            horizon,
            delta=delta,
            beta=beta,
            pool_size=pool_size,
            chip_cost={},
            transfer_penalty=transfer_penalty,
            force_chip=force,
            time_limit=time_limit,
            gap=gap,
        )

    base = solve(replace(rules, chip_windows={}))[0]
    e = {gw: players[f"E{gw}"].to_dict() for gw in horizon}

    def window(c: str, gw: int) -> tuple[int, int] | None:
        return next(((lo, hi) for lo, hi in windows.get(c, []) if lo <= gw <= hi), None)

    def cost(c: str, gw: int) -> float:
        w = window(c, gw)
        return costs.get(c, 0.0) if w is not None and w[1] > horizon[-1] else 0.0

    best: tuple[float, str | None, int, Plan] = (0.0, None, 0, base)
    for t, gw in enumerate(horizon):
        disc = delta**t
        if window("bench_boost", gw):
            bench = [p for p in base.squads[t] if p not in base.xis[t]]
            gain = disc * (1 - beta) * sum(e[gw].get(p, 0.0) for p in bench)
            net = gain - cost("bench_boost", gw)
            if net > best[0]:
                best = (net, "bench_boost", t, base)
        if window("triple_captain", gw):
            net = disc * e[gw].get(base.captains[t], 0.0) - cost("triple_captain", gw)
            if net > best[0]:
                best = (net, "triple_captain", t, base)
        if window("free_hit", gw):
            fh = solve(rules, ("free_hit", gw))
            if fh:
                net = fh[0].objective - base.objective - cost("free_hit", gw)
                if net > best[0]:
                    best = (net, "free_hit", t, fh[0])
    if window("wildcard", horizon[0]):
        wc = solve(rules, ("wildcard", horizon[0]))
        if wc:
            net = wc[0].objective - base.objective - cost("wildcard", horizon[0])
            if net > best[0]:
                best = (net, "wildcard", 0, wc[0])
    _, played, t, plan = best
    if played is None or t != 0:
        return base  # no chip, or hold it for a better week in the horizon
    if played in ("bench_boost", "triple_captain"):
        gw = horizon[0]
        extra = (
            sum(e[gw].get(p, 0.0) for p in base.bench)
            if played == "bench_boost"
            else e[gw].get(base.captain, 0.0)
        )
        return replace(base, chip=played, expected_points=base.expected_points + extra)
    return plan


def _plan(
    v: np.ndarray,
    obj: float,
    uids: list[str],
    pos: np.ndarray,
    exp: np.ndarray,
    horizon: list[int],
    x: np.ndarray,
    y: np.ndarray,
    k: np.ndarray,
    b: np.ndarray,
    s: np.ndarray,
    xf: np.ndarray,
    chip: dict[str, np.ndarray],
    h: np.ndarray,
) -> Plan:
    def on(arr: np.ndarray) -> list[str]:
        return [uids[i] for i in range(len(uids)) if v[int(arr[i])] > 0.5]

    played = next((c for c in CHIP_NAMES if chip[c][0] >= 0 and v[int(chip[c][0])] > 0.5), None)
    squad = on(xf[:, 0]) if played == "free_hit" else on(x[:, 0])
    xi = on(y[:, 0])
    e0 = dict(zip(uids, exp[:, 0], strict=True))
    by_e = sorted(xi, key=lambda p: -e0[p])
    captain = on(k[:, 0])[0]
    vice = next(p for p in by_e if p != captain)
    bench_all = [p for p in squad if p not in xi]
    gk_of = dict(zip(uids, pos, strict=True))
    bench = sorted(bench_all, key=lambda p: (gk_of[p] != "GK", -e0[p]))
    mult = 3 if played == "triple_captain" else 2
    ev = sum(e0[p] for p in xi) + (mult - 1) * e0[captain]
    if played == "bench_boost":
        ev += sum(e0[p] for p in bench)
    return Plan(
        objective=obj,
        gameweek=horizon[0],
        chip=played,
        buys=on(b[:, 0]),
        sells=on(s[:, 0]),
        squad=squad,
        xi=xi,
        bench=bench,
        captain=captain,
        vice=vice,
        hits=round(float(v[int(h[0])])),
        free_transfers_next=0,  # set by the caller from the rules
        expected_points=float(ev),
        squads=[on(x[:, t]) for t in range(len(horizon))],
        xis=[on(y[:, t]) for t in range(len(horizon))],
        captains=[on(k[:, t])[0] for t in range(len(horizon))],
    )


def brute_force_single_week(
    players: pd.DataFrame, rules: SquadRules, budget: int, gw: int
) -> float:
    """Exhaustive optimum of a one-week squad selection with no current squad (tests)."""
    uids = list(players.index)
    pos = players["position"].to_dict()
    team = players["team"].to_dict()
    best = -np.inf
    size = sum(rules.squad.values())
    for squad in itertools.combinations(uids, size):
        if sum(players.loc[list(squad), "price"]) > budget:
            continue
        if any(sum(pos[p] == q for p in squad) != n for q, n in rules.squad.items()):
            continue
        if any(sum(team[p] == c for p in squad) > rules.max_per_club for c in set(team.values())):
            continue
        for xi in itertools.combinations(squad, rules.xi_size):
            cnt = {q: sum(pos[p] == q for p in xi) for q in rules.squad}
            if cnt.get("GK", 0) != 1 or any(cnt.get(q, 0) < mn for q, mn in rules.xi_min.items()):
                continue
            e = {p: float(players.loc[p, f"E{gw}"]) for p in squad}
            val = sum(e[p] for p in xi) + max(e[p] for p in xi)
            bench = [p for p in squad if p not in xi]
            val += 0.1 * sum(e[p] for p in bench)
            best = max(best, val)
    return best
