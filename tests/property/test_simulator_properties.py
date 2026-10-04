from __future__ import annotations

import numpy as np
import pytest
import yaml
from hypothesis import given, settings
from hypothesis import strategies as st
from scipy.stats import chi2_contingency

from fplh.models.bonus import BonusModel
from fplh.models.gk import SaveModel
from fplh.models.goal_process import GoalProcessParams
from fplh.rules.config import load_rules
from fplh.rules.engine import score_arrays
from fplh.settings import get_settings
from fplh.sim.simulator import (
    FULL,
    Components,
    FixtureInputs,
    League,
    SideInputs,
    TimingModel,
    _capped,
    simulate_fixture,
    summarise,
    systematic_sample,
)
from fplh.sim.team import simulate_team

PARAMS = GoalProcessParams.from_dict(
    yaml.safe_load((get_settings().configs_dir / "models" / "goal_process.yaml").read_text())[
        "levels"
    ]["G6"]
)
RULES = load_rules("2024/25")


def side(prefix: str, seed: int, n_out: int = 20) -> SideInputs:
    rng = np.random.default_rng(seed)
    pos = np.array(["GK", "GK", *rng.choice(["DEF", "MID", "FWD"], n_out)])
    n = len(pos)
    return SideInputs(
        np.array([f"{prefix}{i}" for i in range(n)]),
        pos,
        np.r_[0.9, 0.1, rng.uniform(0.02, 0.98, n - 2)],
        rng.uniform(0.3, 0.99, n),
        rng.uniform(0.0, 0.6, n),
        rng.uniform(0.0, 0.6, n),
        rng.uniform(0.0, 0.3, n),
        np.r_[np.zeros(n - 1), 2.0],
    )


@settings(max_examples=25, deadline=None)
@given(
    st.integers(0, 10_000),
    st.floats(0.3, 3.5),
    st.floats(0.3, 3.5),
    st.sampled_from([3, 5]),
    st.integers(14, 24),
)
def test_match_invariants(seed: int, lh: float, la: float, limit: int, n_out: int) -> None:
    fx = FixtureInputs(
        "s:h:a", (lh, la), (side("h", seed, n_out), side("a", seed + 1, n_out)), limit
    )
    r = simulate_fixture(fx, PARAMS, RULES, League(), TimingModel(), 400, seed=seed)
    for k, sd in enumerate(r.sides):
        ev = sd.events
        team = r.home_goals if k == 0 else r.away_goals
        # player goals + opponents' own goals = the team's goals
        assert (ev["goals_scored"].sum(1) + r.sides[1 - k].events["own_goals"].sum(1) == team).all()
        assert ev["minutes"].min() >= 0 and ev["minutes"].max() <= FULL
        assert (ev["assists"].sum(1) <= ev["goals_scored"].sum(1)).all()
        # exactly eleven start, one of them a goalkeeper
        assert (sd.started.sum(1) == 11).all()
        assert (sd.started[:, sd.position == "GK"].sum(1) == 1).all()
        # substitutes within the limit; never more than eleven on the pitch
        subs = ((ev["minutes"] > 0) & ~sd.started).sum(1)
        assert (subs <= limit).all()
        on_at_end = ((ev["minutes"] > 0) & sd.started).sum(1)
        assert (on_at_end + subs <= 11 + limit).all()
        # conceded only counts goals by the other side
        assert (ev["goals_conceded"].max(1) <= (r.away_goals if k == 0 else r.home_goals)).all()


def test_replays_are_identical() -> None:
    fx = FixtureInputs("s:h:a", (1.4, 1.1), (side("h", 1), side("a", 2)), 5)
    a = simulate_fixture(fx, PARAMS, RULES, League(), TimingModel(), 500, seed=7)
    b = simulate_fixture(fx, PARAMS, RULES, League(), TimingModel(), 500, seed=7)
    assert summarise(a).equals(summarise(b))


@pytest.mark.parametrize("total", [1, 3, 10])
def test_systematic_sampling_has_exact_inclusion_probabilities(total: int) -> None:
    rng = np.random.default_rng(0)
    w = _capped(rng.uniform(0.05, 1.0, 15), float(total))
    assert w.max() <= 1.0 and abs(w.sum() - total) < 1e-9
    picks = systematic_sample(np.tile(w, (40_000, 1)), rng)
    assert (picks.sum(1) == total).all()
    assert np.abs(picks.mean(0) - w).max() < 0.012


def test_simulated_minutes_match_their_targets() -> None:
    h, a = side("h", 3), side("a", 4)
    fx = FixtureInputs("s:h:a", (1.4, 1.1), (h, a), 5)
    r = simulate_fixture(fx, PARAMS, RULES, League(), TimingModel(), 20_000, seed=3)
    sd = r.sides[0]
    gk = h.position == "GK"
    target = np.where(gk, h.p_start / h.p_start[gk].sum(), h.p_start * 10 / h.p_start[~gk].sum())
    target = np.minimum(target, 1.0)
    assert np.abs(sd.started.mean(0) - target).max() < 0.03
    started = sd.started
    p60 = ((sd.events["minutes"] >= 60) & started).sum(0) / np.maximum(started.sum(0), 1)
    busy = started.sum(0) > 2000
    # exits beyond the substitution limit stay on, so 60+ can only be higher than π^60
    assert (p60[busy] - h.p_full[busy] > -0.02).all()


def test_team_goals_follow_the_team_only_simulation() -> None:
    """§8.5: the player layer leaves the team-goal distribution of the goal process intact."""
    fx = FixtureInputs("s:h:a", (1.6, 1.0), (side("h", 5), side("a", 6)), 5)
    r = simulate_fixture(fx, PARAMS, RULES, League(), TimingModel(), 20_000, seed=11)
    ref = simulate_team(PARAMS, np.array([[1.6, 1.0]]), 20_000, seed=99)
    for full, team in ((r.home_goals, ref.home[0]), (r.away_goals, ref.away[0])):
        cells = np.minimum(np.r_[full, team], 5)
        table = np.array(
            [np.bincount(cells[:20_000], minlength=6), np.bincount(cells[20_000:], minlength=6)]
        )
        assert chi2_contingency(table).pvalue > 1e-3


def full_side(prefix: str, seed: int) -> SideInputs:
    s = side(prefix, seed)
    rng = np.random.default_rng(seed + 100)
    n = len(s)
    s.yellow_rate = rng.uniform(0.0, 0.4, n)
    s.save_mult = np.where(s.position == "GK", rng.uniform(0.8, 1.2, n), 1.0)
    s.dc_rate = rng.uniform(0.5, 6.0, (n, 3))
    s.dc_size = np.full(n, 6.0)
    return s


BONUS = BonusModel(
    weights={
        "minutes_lt_60": 3,
        "minutes_gte_60": 6,
        "goal_FWD": 24,
        "goal_MID": 18,
        "goal_DEF": 12,
        "assist": 9,
        "clean_sheet": 12,
        "save": 2,
    },
    sigma={f"{p}:{b}": 4.0 for p in ("GK", "DEF", "MID", "FWD") for b in ("lt60", "60")},
)
COMPONENTS = Components(SaveModel(), BONUS)


@settings(max_examples=15, deadline=None)
@given(st.integers(0, 10_000), st.floats(0.3, 3.5), st.floats(0.3, 3.5), st.booleans())
def test_component_events_are_consistent(seed: int, lh: float, la: float, dc: bool) -> None:
    rules = load_rules("2025/26" if dc else "2024/25")
    fx = FixtureInputs("s:h:a", (lh, la), (full_side("h", seed), full_side("a", seed + 1)), 5)
    r = simulate_fixture(fx, PARAMS, rules, League(), TimingModel(), 300, seed, COMPONENTS)
    bonus_total = np.zeros(300, dtype=np.int64)
    for k, sd in enumerate(r.sides):
        ev = sd.events
        played = ev["minutes"] > 0
        gk = sd.position == "GK"
        # cards: at most one yellow, never with a red; only players who played
        assert ev["yellow_cards"].max() <= 1
        assert not ((ev["yellow_cards"] > 0) & (ev["red_cards"] > 0)).any()
        assert not (ev["yellow_cards"][~played] > 0).any()
        # saves only by goalkeepers who played; saved penalties are the opponent's misses
        assert (ev["saves"][:, ~gk] == 0).all() and not (ev["saves"][~played] > 0).any()
        opp_missed = r.sides[1 - k].events["penalties_missed"].sum(1)
        assert (ev["penalties_saved"].sum(1) <= opp_missed).all()
        # defensive actions only when the rules score them, and only for players who played
        acts = sum(ev[a] for a in ("clearances_blocks_interceptions", "tackles", "recoveries"))
        assert (acts.sum() > 0) == dc
        assert not (acts[~played] > 0).any()
        # bonus never goes to players who did not play
        assert not (ev["bonus"][~played] > 0).any()
        bonus_total += ev["bonus"].sum(1)
        # points equal the sum of the scoring components
        parts = score_arrays(ev, np.broadcast_to(sd.position, ev["minutes"].shape), rules)
        assert (sum(parts.values()) == sd.points).all()
    # someone wins the three-point bonus in every match (ties can only add)
    assert (bonus_total >= 6).all()


def test_saves_and_cards_match_their_rates() -> None:
    h, a = full_side("h", 8), full_side("a", 9)
    fx = FixtureInputs("s:h:a", (1.4, 1.1), (h, a), 5, mean_goals=(1.5, 1.2))
    r = simulate_fixture(fx, PARAMS, RULES, League(), TimingModel(), 20_000, 5, COMPONENTS)
    sd = r.sides[0]
    ev = sd.events
    m = ev["minutes"] / 90
    gk = h.position == "GK"
    expected = COMPONENTS.saves.mean(1.2, ev["minutes"], h.save_mult[None, :])[:, gk]
    assert abs(ev["saves"][:, gk].mean() - expected.mean()) < 0.03
    p_yellow = (1 - np.exp(-h.yellow_rate[None, :] * m)) * (ev["red_cards"] == 0)
    assert abs(ev["yellow_cards"].mean() - p_yellow.mean()) < 0.003
