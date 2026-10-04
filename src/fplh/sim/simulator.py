"""Player-level match simulator (ARCHITECTURE.md §8), vectorised over simulations.

Per fixture, ``n_sims`` matches are simulated:

1. **Team process.** The goal process (G-level from ``configs/models/goal_process.yaml``)
   at the fixture's nominal rates gives goal and red-card minutes per side
   (``sim/team.py``). Player allocation does not feed back into team intensities
   (lineup-aware rates are G7, Phase 5), so team goal totals are exactly the process's.
2. **Starting XI.** Systematic sampling in a random order: 1 goalkeeper and 10 outfield
   players with inclusion probabilities exactly π^S (after scaling each side's π^S to sum
   to 1 and 10, capped at 1).
3. **Exits.** A starter plays 60+ with probability π^60; their exit minute is drawn from
   the empirical distribution of starters' exits in that band (90 = to the end), from
   Understat lineups observable at the deadline. Exits beyond the substitution limit stay on.
4. **Entrants.** As many bench players as there are exits, sampled with probabilities
   proportional to π^B, enter at the exit minutes in random order.
5. **Red cards** from the team process send off an on-pitch player who was not due to be
   substituted (no replacement), drawn ∝ M9 yellow rate when given.
6. **Goals.** Each team goal is an own goal (credited to an opponent on the pitch,
   defenders most likely), a penalty (the on-pitch taker with the most decayed attempts)
   or open play: scorer ∝ M5/M6 goal rate among players on the pitch, then an FPL assist
   with the league's share, assister ∝ assist rate among team-mates.
7. **Penalty misses** at the league rate per side, by the on-pitch taker; the opposing
   goalkeeper on the pitch saves a share of them (M8).
8. **Other events** (``Components`` and the optional ``SideInputs`` fields):
   * yellow cards (M9): P = 1 − exp(−rate·minutes/90), at most one, none with a red;
   * goalkeeper saves (M8): NegBin with mean m·g_k·(a + b·μ_opp), μ_opp the opponent's
     mean goals;
   * defensive actions (M7), only when the season's rules score them: Poisson per action
     with a Gamma frailty per player-match shared by the actions;
   * bonus (M10): BPS from the simulated events plus the player's residual, then the
     official allocation over both sides' players who played.
9. **Points** from ``rules.engine.score_arrays``.

Random numbers come from one generator per fixture seeded by (seed, fixture uid), so
replays are identical and comparisons can share common random numbers.
"""

from __future__ import annotations

import zlib
from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt
import pandas as pd

from fplh.models.bonus import BonusModel
from fplh.models.defence import ACTIONS
from fplh.models.gk import SaveModel
from fplh.models.goal_process import REGULAR, GoalProcessParams
from fplh.rules.bonus import assign_bonus_array
from fplh.rules.config import Rules
from fplh.rules.engine import score_arrays
from fplh.sim.team import simulate_team

Array = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]
FULL = 90
NEVER = 10_000
OG_WEIGHT = {"GK": 0.5, "DEF": 3.0, "MID": 1.0, "FWD": 0.5}
THRESHOLDS = (2, 6, 10, 15)


@dataclass
class SideInputs:
    player_uid: npt.NDArray[np.str_]
    position: npt.NDArray[np.str_]
    p_start: Array
    p_full: Array
    p_sub: Array
    goal_rate: Array
    assist_rate: Array
    pen_weight: Array
    # PR 5 components; None leaves the event at zero
    yellow_rate: Array | None = None  # M9, per 90
    save_mult: Array | None = None  # M8 goalkeeper multiplier g_k
    dc_rate: Array | None = None  # M7 (P, 3) per-90 rates of ACTIONS, opponent included
    dc_size: Array | None = None  # M7 NegBin size of the player's group total

    def __len__(self) -> int:
        return len(self.player_uid)


@dataclass
class FixtureInputs:
    fixture_uid: str
    nominal: tuple[float, float]  # per-90 base rates (home, away) for the goal process
    sides: tuple[SideInputs, SideInputs]
    sub_limit: int = 5
    mean_goals: tuple[float, float] | None = None  # μ (home, away) for saves; else simulated


@dataclass
class Components:
    """League-level parts of M8 and M10 (``models/gk.py``, ``models/bonus.py``)."""

    saves: SaveModel | None = None
    bonus: BonusModel | None = None


@dataclass
class TimingModel:
    """Empirical starter exit minutes: P(exit at 60..90 | 60+) and P(exit at 0..59 | <60)."""

    full: Array = field(default_factory=lambda: np.r_[np.zeros(30), 1.0])  # 60..89, 90
    early: Array = field(default_factory=lambda: np.full(60, 1 / 60))  # 0..59

    @classmethod
    def from_lineups(cls, lineups: pd.DataFrame) -> TimingModel:
        st = lineups[lineups["started"].astype(bool) & (lineups["off_reason"] != "red")]
        off = st["off_minute"].clip(upper=FULL).to_numpy(dtype=int)
        full = np.bincount(off[off >= 60] - 60, minlength=31)[:31].astype(float)
        early = np.bincount(off[off < 60], minlength=60)[:60].astype(float)
        if full.sum() == 0 or early.sum() == 0:
            return cls()
        return cls(full / full.sum(), early / early.sum())


@dataclass
class League:
    penalty_share: float = 0.075
    own_goal_share: float = 0.033
    assist_share: float = 0.88
    penalty_misses_per_side: float = 0.021


@dataclass
class SideResult:
    player_uid: npt.NDArray[np.str_]
    position: npt.NDArray[np.str_]
    events: dict[str, IntArray]  # (S, P) per FPL event column
    started: npt.NDArray[np.bool_]
    points: IntArray  # (S, P)


@dataclass
class FixtureResult:
    fixture_uid: str
    home_goals: IntArray
    away_goals: IntArray
    sides: tuple[SideResult, SideResult]


def systematic_sample(weights: Array, rng: np.random.Generator) -> npt.NDArray[np.bool_]:
    """(S, P) inclusion indicators; row s has exactly round(Σ weights[s]) picks and
    P(pick p) = weights[s, p] (each ≤ 1). The order is random per row."""
    s, p = weights.shape
    order = np.argsort(rng.random((s, p)), axis=1)
    w = np.take_along_axis(weights, order, axis=1)
    c = np.cumsum(w, axis=1)
    u = rng.random((s, 1))
    picked_sorted = (np.floor(c - u) - np.floor(c - w - u)) >= 1
    picked = np.zeros((s, p), dtype=bool)
    np.put_along_axis(picked, order, picked_sorted, axis=1)
    return picked


def _capped(p: Array, total: float) -> Array:
    """Scale ``p`` to sum to ``total`` with every entry ≤ 1 (iterative capping)."""
    q = np.clip(np.asarray(p, dtype=float), 1e-9, None)
    fixed = np.zeros(len(q), dtype=bool)
    for _ in range(len(q) + 1):
        free = ~fixed
        remaining = total - fixed.sum()
        if remaining <= 0 or not free.any():
            break
        q[free] = q[free] * remaining / q[free].sum()
        over = free & (q > 1)
        if not over.any():
            break
        q[over] = 1.0
        fixed |= over
    out: Array = np.minimum(q, 1.0)
    return out


def _choose(weights: Array, rng: np.random.Generator) -> IntArray:
    """One index per row ∝ weights (rows with zero weight return −1)."""
    total = weights.sum(axis=1)
    c = np.cumsum(weights, axis=1)
    u = rng.random(len(weights)) * total
    idx = (c < u[:, None]).sum(axis=1)
    out: IntArray = np.where(total > 0, np.minimum(idx, weights.shape[1] - 1), -1)
    return out


def _lineups(
    side: SideInputs, n: int, limit: int, timing: TimingModel, rng: np.random.Generator
) -> tuple[npt.NDArray[np.bool_], IntArray, IntArray]:
    """Starting XI (S, P), on and off minutes (S, P)."""
    gk = side.position == "GK"
    xi = np.zeros((n, len(side)), dtype=bool)
    # two strata: exactly one goalkeeper and ten outfield players
    for members, size in ((gk, 1.0), (~gk, min(10.0, float((~gk).sum())))):
        if members.any():
            w = _capped(side.p_start[members], size)
            xi[:, members] = systematic_sample(np.tile(w, (n, 1)), rng)
    full = rng.random((n, len(side))) < side.p_full[None, :]
    late = 60 + rng.choice(31, size=(n, len(side)), p=timing.full)
    early = rng.choice(60, size=(n, len(side)), p=timing.early)
    off = np.where(xi, np.where(full, late, early), 0).astype(np.int64)
    on = np.where(xi, 0, NEVER).astype(np.int64)
    exiting = xi & (off < FULL)
    # exits beyond the substitution limit stay on (keep the earliest ``limit``)
    rank = np.argsort(np.argsort(np.where(exiting, off, NEVER), axis=1, kind="stable"), axis=1)
    off = np.where(exiting & (rank >= limit), FULL, off)
    exiting = xi & (off < FULL)
    k = exiting.sum(axis=1)
    bench = ~xi
    q = np.where(bench, np.clip(side.p_sub, 1e-6, None)[None, :], 0.0)
    # inclusion weights summing to k per row (each ≤ 1): scale, cap, then hand the
    # capped excess to the uncapped players once
    tot = q.sum(axis=1, keepdims=True)
    weights = np.minimum(q * (k[:, None] / np.where(tot > 0, tot, 1.0)), 1.0)
    deficit = k - weights.sum(axis=1)
    room = np.where(weights < 1, q, 0.0)
    room_total = np.where(room.sum(axis=1) > 0, room.sum(axis=1), 1.0)
    weights = np.minimum(weights + room * (deficit / room_total)[:, None], 1.0)
    entrants = systematic_sample(weights, rng) & bench
    # pair entrants with exit minutes in time order (entrants in random order)
    exit_minutes = np.sort(np.where(exiting, off, NEVER), axis=1)
    order = np.where(entrants, rng.random(entrants.shape), np.inf)
    entrant_rank = np.argsort(np.argsort(order, axis=1), axis=1)
    slot = np.minimum(entrant_rank, exit_minutes.shape[1] - 1)
    minute = np.take_along_axis(exit_minutes, slot, axis=1)
    use = entrants & (minute < NEVER)
    on = np.where(use, minute, on)
    off = np.where(use, FULL, off)
    return xi, on, off


def simulate_fixture(
    fx: FixtureInputs,
    params: GoalProcessParams,
    rules: Rules,
    league: League,
    timing: TimingModel,
    n_sims: int,
    seed: int = 0,
    components: Components | None = None,
) -> FixtureResult:
    comp = components or Components()
    rng = np.random.default_rng([seed, zlib.crc32(fx.fixture_uid.encode())])
    team = simulate_team(
        params, np.array([fx.nominal], dtype=float), n_sims, int(rng.integers(2**31)), record=True
    )
    if team.goal_slots is None or team.red_slots is None:  # pragma: no cover
        raise RuntimeError("team simulation did not record events")
    goal_slots = team.goal_slots[0].astype(np.int64)  # (S, T, 2)
    red_slots = team.red_slots[0].astype(np.int64)
    s_idx = np.arange(n_sims)
    lineups = [_lineups(side, n_sims, fx.sub_limit, timing, rng) for side in fx.sides]
    ons = [lu[1] for lu in lineups]
    offs = [lu[2] for lu in lineups]
    counts = {
        name: [np.zeros((n_sims, len(side)), dtype=np.int64) for side in fx.sides]
        for name in (
            "goals_scored",
            "assists",
            "own_goals",
            "penalties_missed",
            "penalties_saved",
            "red_cards",
        )
    }

    sent_off = [np.full((n_sims, len(side)), NEVER, dtype=np.int64) for side in fx.sides]

    def on_pitch(k: int, rows: IntArray, t: IntArray) -> npt.NDArray[np.bool_]:
        """Players on the pitch in slot t; off ≥ 90 means on until the final whistle."""
        off = offs[k][rows]
        out: npt.NDArray[np.bool_] = (
            (ons[k][rows] <= t[:, None])
            & ((off > t[:, None]) | (off >= FULL))
            & (sent_off[k][rows] > t[:, None])
        )
        return out

    # red cards, in time order: an outfield player on the pitch who is not due to be
    # substituted leaves, unreplaced (two reds in one slot are handled one at a time);
    # the player is drawn ∝ yellow rate (M9) when given
    outfield = [
        (side.position != "GK")[None, :]
        * (side.yellow_rate + 1e-6 if side.yellow_rate is not None else np.ones(len(side)))[None, :]
        for side in fx.sides
    ]
    for k in (0, 1):
        for t in np.unique(np.nonzero(red_slots[:, :, k])[1]):
            for nth in range(int(red_slots[:, t, k].max())):
                rows = np.flatnonzero(red_slots[:, t, k] > nth)
                ts = np.full(len(rows), t)
                eligible = (on_pitch(k, rows, ts) & (offs[k][rows] >= FULL)) * outfield[k]
                pick = _choose(eligible.astype(float), rng)
                ok = pick >= 0
                offs[k][rows[ok], pick[ok]] = min(int(t), FULL)
                sent_off[k][rows[ok], pick[ok]] = t
                counts["red_cards"][k][rows[ok], pick[ok]] += 1

    pen_share, og_share = league.penalty_share, league.own_goal_share
    assist_p = min(league.assist_share / max(1 - pen_share - og_share, 1e-9), 0.95)
    for k in (0, 1):
        side, opp = fx.sides[k], fx.sides[1 - k]
        rows, ts = np.nonzero(goal_slots[:, :, k])
        reps = goal_slots[rows, ts, k]
        rows, ts = np.repeat(rows, reps), np.repeat(ts, reps)
        if not len(rows):
            continue
        kind = rng.random(len(rows))
        is_og = kind < og_share
        is_pen = (kind >= og_share) & (kind < og_share + pen_share)
        own = on_pitch(k, rows, ts)
        # own goals: an opponent on the pitch
        og_w = np.array([OG_WEIGHT.get(str(p), 1.0) for p in opp.position])
        opp_on = on_pitch(1 - k, rows, ts)
        who = _choose(opp_on * og_w[None, :], rng)
        sel = is_og & (who >= 0)
        np.add.at(counts["own_goals"][1 - k], (rows[sel], who[sel]), 1)
        # penalties: the on-pitch taker
        taker_w = np.where(side.pen_weight > 0, side.pen_weight, 1e-6 * (side.goal_rate + 1e-9))
        taker = np.argmax(own * taker_w[None, :], axis=1)
        has = own.any(axis=1)
        sel = is_pen & has
        np.add.at(counts["goals_scored"][k], (rows[sel], taker[sel]), 1)
        # open play: scorer ∝ goal rate, then an assister ∝ assist rate
        scorer = _choose(own * side.goal_rate[None, :], rng)
        sel = ~is_og & ~is_pen & (scorer >= 0)
        np.add.at(counts["goals_scored"][k], (rows[sel], scorer[sel]), 1)
        mates = own * side.assist_rate[None, :]
        mates[np.arange(len(rows)), np.maximum(scorer, 0)] = 0.0
        assister = _choose(mates, rng)
        assisted = sel & (rng.random(len(rows)) < assist_p) & (assister >= 0)
        np.add.at(counts["assists"][k], (rows[assisted], assister[assisted]), 1)
    # penalty misses
    for k in (0, 1):
        side = fx.sides[k]
        n_miss = rng.poisson(league.penalty_misses_per_side, n_sims)
        rows = np.repeat(s_idx, n_miss)
        if not len(rows):
            continue
        ts = rng.integers(0, REGULAR, len(rows))
        own = on_pitch(k, rows, ts)
        taker_w = np.where(side.pen_weight > 0, side.pen_weight, 1e-6 * (side.goal_rate + 1e-9))
        taker = np.argmax(own * taker_w[None, :], axis=1)
        ok = own.any(axis=1)
        np.add.at(counts["penalties_missed"][k], (rows[ok], taker[ok]), 1)
        # FPL counts a saved penalty as missed too: the keeper on the pitch saves a share
        share = comp.saves.penalty_save_share if comp.saves is not None else 0.0
        keeper = on_pitch(1 - k, rows, ts) & (fx.sides[1 - k].position == "GK")[None, :]
        gk_idx = _choose(keeper.astype(float), rng)
        saved = ok & (gk_idx >= 0) & (rng.random(len(rows)) < share)
        np.add.at(counts["penalties_saved"][1 - k], (rows[saved], gk_idx[saved]), 1)

    cum = np.cumsum(goal_slots, axis=1)  # (S, T, 2) goals up to and including slot t
    t_max = goal_slots.shape[1]
    dc_active = bool(
        rules.defensive_contribution.DEF.points or rules.defensive_contribution.MID_FWD.points
    )
    all_events = []
    for k in (0, 1):
        side = fx.sides[k]
        on, off = ons[k], offs[k]
        played = np.clip(np.minimum(off, FULL) - np.minimum(on, FULL), 0, FULL)
        start_t = np.minimum(on, t_max) - 1
        end_t = np.minimum(np.where(off >= FULL, t_max, off), t_max) - 1
        opp_cum = cum[:, :, 1 - k]
        conceded = np.where(
            played > 0,
            np.take_along_axis(opp_cum, np.clip(end_t, 0, None), axis=1) * (end_t >= 0)
            - np.take_along_axis(opp_cum, np.clip(start_t, 0, None), axis=1) * (start_t >= 0),
            0,
        )
        m = played / 90
        zeros = np.zeros_like(played)
        yellow = zeros
        if side.yellow_rate is not None:  # M9: at most one, and none with a red
            p_yellow = 1 - np.exp(-side.yellow_rate[None, :] * m)
            yellow = ((rng.random(played.shape) < p_yellow) & (counts["red_cards"][k] == 0)).astype(
                np.int64
            )
        saves = zeros
        if comp.saves is not None and side.save_mult is not None:  # M8
            mu_opp = (
                fx.mean_goals[1 - k]
                if fx.mean_goals is not None
                else float((team.away if k == 0 else team.home)[0].mean())
            )
            gk = (side.position == "GK")[None, :]
            mean = comp.saves.mean(mu_opp, played, side.save_mult[None, :]) * gk
            size = comp.saves.size
            saves = rng.poisson(rng.gamma(size, 1.0, played.shape) * mean / size)
        actions = {a: zeros for a in ACTIONS}
        if dc_active and side.dc_rate is not None:  # M7: shared Gamma frailty per match
            size_p = side.dc_size if side.dc_size is not None else np.full(len(side), 20.0)
            frailty = rng.gamma(size_p[None, :], 1.0 / size_p[None, :], played.shape)
            for j, a in enumerate(ACTIONS):
                actions[a] = rng.poisson(side.dc_rate[None, :, j] * m * frailty)
        all_events.append(
            {
                "minutes": played,
                "goals_scored": counts["goals_scored"][k],
                "assists": counts["assists"][k],
                "goals_conceded": conceded,
                "own_goals": counts["own_goals"][k],
                "penalties_saved": counts["penalties_saved"][k],
                "penalties_missed": counts["penalties_missed"][k],
                "yellow_cards": yellow,
                "red_cards": counts["red_cards"][k],
                "saves": saves,
                "bonus": zeros,
                **actions,
            }
        )
    if comp.bonus is not None:  # M10: BPS over both sides, official allocation
        bps = np.concatenate(
            [
                comp.bonus.sample_bps(ev, side.position, side.player_uid, rng)
                for ev, side in zip(all_events, fx.sides, strict=True)
            ],
            axis=1,
        )
        played_any = np.concatenate([ev["minutes"] > 0 for ev in all_events], axis=1)
        bonus = assign_bonus_array(bps, rules.bonus.ranks, eligible=played_any)
        split = len(fx.sides[0])
        all_events[0]["bonus"] = bonus[:, :split].astype(np.int64)
        all_events[1]["bonus"] = bonus[:, split:].astype(np.int64)
    results = []
    for k in (0, 1):
        side, events = fx.sides[k], all_events[k]
        position = np.broadcast_to(side.position, events["minutes"].shape)
        pts = score_arrays(events, position, rules)
        total = sum(pts.values())
        results.append(
            SideResult(side.player_uid, side.position, events, lineups[k][0], np.asarray(total))
        )
    return FixtureResult(fx.fixture_uid, team.home[0], team.away[0], (results[0], results[1]))


def summarise(result: FixtureResult) -> pd.DataFrame:
    """Per player: expected points and the §8.4 summary quantities."""
    rows = []
    for side in result.sides:
        pts = side.points
        ev = side.events
        frame = {
            "player_uid": side.player_uid,
            "fixture_uid": result.fixture_uid,
            "expected_points": pts.mean(axis=0),
            "p_start": side.started.mean(axis=0),
            "p_60": (ev["minutes"] >= 60).mean(axis=0),
            "expected_minutes": ev["minutes"].mean(axis=0),
            "e_goals": ev["goals_scored"].mean(axis=0),
            "e_assists": ev["assists"].mean(axis=0),
            "p_clean_sheet": ((ev["minutes"] >= 60) & (ev["goals_conceded"] == 0)).mean(axis=0),
            "e_saves": ev["saves"].mean(axis=0),
            "p_yellow": (ev["yellow_cards"] > 0).mean(axis=0),
            "e_bonus": ev["bonus"].mean(axis=0),
        }
        for t in THRESHOLDS:
            frame[f"p_pts_ge_{t}"] = (pts >= t).mean(axis=0)
        rows.append(pd.DataFrame(frame))
    return pd.concat(rows, ignore_index=True)
