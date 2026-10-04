"""Team-only match simulator for the goal process (ARCHITECTURE.md §7.5, gap 5).

Vectorised over (fixtures × simulations), one-minute slots, using the same intensity as
``models.goal_process``: no players, lineups or bookings, only team intensities, game
state, red cards, frailty and second-half stoppage. Random numbers are drawn per slot
for the simulation axis only and shared across fixtures (common random numbers), so
neighbouring nominal rates give smoothly varying results, which the emulator needs.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from fplh.models.goal_process import (
    BUCKETS,
    DELTAS,
    REGULAR,
    SLOTS,
    GoalProcessParams,
    bucket,
    g_bin,
)

Array = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]


@dataclass
class StartState:
    """In-play start: the slot to start from and the state before it, per fixture."""

    minute: IntArray  # (F,)
    home_goals: IntArray
    away_goals: IntArray
    home_reds: IntArray
    away_reds: IntArray


@dataclass
class TeamSim:
    home: IntArray  # (F, S) final goals
    away: IntArray
    next_goal: IntArray  # (F, S): 0 home, 1 away, −1 none (after the start slot)
    goal_slots: npt.NDArray[np.int8] | None = None  # (F, S, SLOTS, 2) goals per slot
    red_slots: npt.NDArray[np.int8] | None = None  # (F, S, SLOTS, 2) reds per slot
    length: IntArray | None = None  # (F, S) slots played (90 + stoppage)


def _poisson(lam: Array, u: Array) -> IntArray:
    """Inverse-CDF Poisson draw (λ per slot is small; counts above 4 are negligible)."""
    p = np.exp(-lam)
    cdf = p.copy()
    k = np.zeros(lam.shape, dtype=np.int64)
    for j in range(1, 5):
        more = u > cdf
        k += more
        p = p * lam / j
        cdf = cdf + p
    return k


class Intensity:
    """Slot log-intensity lookups for one parameter set (precomputed once per run)."""

    def __init__(self, params: GoalProcessParams) -> None:
        self.params = params
        t = np.arange(SLOTS)
        self.log_g = np.asarray(params.log_g)[g_bin(t)] - np.log(REGULAR)  # per slot
        self.bucket = bucket(t)
        # β by (Δ + 2, bucket); Δ = 0 is the reference
        table = np.zeros((5, len(BUCKETS)))
        for i, d in enumerate(DELTAS):
            table[d + 2] = params.beta_table()[i]
        self.beta = table

    def log_lambda(
        self, log_nominal: Array, t: int, delta: IntArray, red_own: IntArray, red_opp: IntArray
    ) -> Array:
        """log λ for slot ``t``: log(nominal/90) + log g + β(Δ, bucket) + red cards."""
        p = self.params
        out: Array = (
            log_nominal
            + self.log_g[t]
            + self.beta[np.clip(delta, -2, 2) + 2, self.bucket[t]]
            + p.red_own * np.minimum(red_own, 2)
            + p.red_opp * np.minimum(red_opp, 2)
        )
        return out


def simulate_team(
    params: GoalProcessParams,
    nominal: Array,
    n_sims: int,
    seed: int = 0,
    *,
    start: StartState | None = None,
    record: bool = False,
) -> TeamSim:
    """``nominal``: (F, 2) per-90 base rates (home, away) in a level 11-v-11 state."""
    nominal = np.asarray(nominal, dtype=float)
    f = nominal.shape[0]
    rng = np.random.default_rng(seed)
    shape = (f, n_sims)
    log_nom = np.log(nominal)[:, :, None]  # (F, 2, 1)
    if np.isfinite(params.log_frailty_shape):
        a = float(np.exp(params.log_frailty_shape))
        eps = rng.gamma(a, 1.0 / a, size=n_sims)
    else:
        eps = np.ones(n_sims)
    surv = np.r_[np.ones(REGULAR), params.stoppage.survival]
    v = rng.random(n_sims)
    goals = np.zeros((2, *shape), dtype=np.int64)
    reds = np.zeros((2, *shape), dtype=np.int64)
    t0 = np.zeros(f, dtype=np.int64)
    if start is not None:
        t0 = start.minute.astype(np.int64)
        goals[0] += start.home_goals[:, None]
        goals[1] += start.away_goals[:, None]
        reds[0] += start.home_reds[:, None]
        reds[1] += start.away_reds[:, None]
    next_goal = np.full(shape, -1, dtype=np.int64)
    goal_slots = np.zeros((*shape, SLOTS, 2), dtype=np.int8) if record else None
    red_slots = np.zeros((*shape, SLOTS, 2), dtype=np.int8) if record else None
    length = np.full(shape, REGULAR, dtype=np.int64)
    third_effect = np.asarray(params.red_third)
    intensity = Intensity(params)
    for t in range(SLOTS):
        u_goal = rng.random((2, n_sims))
        u_red = rng.random((2, n_sims))
        if t >= REGULAR and surv[t] <= 0:
            break
        active = (v < surv[t])[None, :] & (t >= t0)[:, None]  # (F, S)
        if t >= REGULAR:
            length += (v < surv[t])[None, :]
        if not active.any():
            continue
        new_goals = []
        new_reds = []
        for k in (0, 1):
            delta = goals[k] - goals[1 - k]
            eta = intensity.log_lambda(log_nom[:, k], t, delta, reds[k], reds[1 - k])
            lam = np.exp(eta) * eps[None, :] * active
            new_goals.append(_poisson(lam, u_goal[k][None, :]))
            if params.reds_on:
                h = np.exp(
                    params.red_h0
                    + third_effect[min(t // 30, 2)]
                    + params.red_gamma * np.clip(delta, -2, 2)
                )
                new_reds.append(((u_red[k][None, :] < 1 - np.exp(-h)) & active).astype(np.int64))
        first = (next_goal < 0) & ((new_goals[0] > 0) | (new_goals[1] > 0))
        next_goal = np.where(first, np.where(new_goals[0] > 0, 0, 1), next_goal)
        goals[0] += new_goals[0]
        goals[1] += new_goals[1]
        if goal_slots is not None:
            goal_slots[:, :, t, 0] = new_goals[0]
            goal_slots[:, :, t, 1] = new_goals[1]
        if params.reds_on:
            reds[0] += new_reds[0]
            reds[1] += new_reds[1]
            if red_slots is not None:
                red_slots[:, :, t, 0] = new_reds[0]
                red_slots[:, :, t, 1] = new_reds[1]
    return TeamSim(goals[0], goals[1], next_goal, goal_slots, red_slots, length)


def score_grid(sim: TeamSim, cap: int = 10) -> Array:
    """(F, cap+1, cap+1) scoreline probabilities; goals above ``cap`` pooled into it."""
    h = np.minimum(sim.home, cap)
    a = np.minimum(sim.away, cap)
    f, s = h.shape
    grid = np.zeros((f, cap + 1, cap + 1))
    idx = np.repeat(np.arange(f), s)
    np.add.at(grid, (idx, h.ravel(), a.ravel()), 1.0)
    out: Array = grid / s
    return out
