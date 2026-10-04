from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
import pytest
from scipy.stats import poisson

from fplh.models.goal_process import (
    BUCKETS,
    DELTAS,
    N_G,
    REGULAR,
    SLOTS,
    GoalProcessParams,
    Stoppage,
    build_cells,
    estimate_stoppage,
    fit_goal_process,
)
from fplh.sim.emulator import Emulator
from fplh.sim.team import StartState, simulate_team

NO_STOPPAGE = Stoppage((0.0,) * 15)


def null_params(**kw: object) -> GoalProcessParams:
    base: dict[str, object] = {
        "level": "G4",
        "log_g": (0.0,) * N_G,
        "beta": ((0.0,) * len(BUCKETS),) * len(DELTAS),
        "stoppage": NO_STOPPAGE,
    }
    base.update(kw)
    return GoalProcessParams(**base)  # type: ignore[arg-type]


def poisson_grid(mh: float, ma: float) -> np.ndarray:
    g = np.outer(poisson.pmf(np.arange(11), mh), poisson.pmf(np.arange(11), ma))
    return g / g.sum()


def test_null_process_is_independent_poisson() -> None:
    sim = simulate_team(null_params(), np.array([[1.6, 1.1]]), 100_000, seed=3)
    assert abs(sim.home.mean() - 1.6) < 0.02 and abs(sim.away.mean() - 1.1) < 0.02
    grid = np.zeros((11, 11))
    np.add.at(grid, (np.minimum(sim.home[0], 10), np.minimum(sim.away[0], 10)), 1)
    assert np.abs(grid / grid.sum() - poisson_grid(1.6, 1.1)).max() < 0.004


def simulate_events(params: GoalProcessParams, mu: np.ndarray, seed: int) -> pd.DataFrame:
    """Matches from the process with goal minutes, for parameter recovery."""
    rng = np.random.default_rng(seed)
    n = len(mu)
    goals = np.zeros((n, 2), dtype=int)
    rows = []
    beta = np.zeros((5, len(BUCKETS)))
    for i, d in enumerate(DELTAS):
        beta[d + 2] = params.beta_table()[i]
    for t in range(REGULAR):
        b = int(np.searchsorted(np.array(BUCKETS), t, side="right") - 1)
        for k in (0, 1):
            delta = np.clip(goals[:, k] - goals[:, 1 - k], -2, 2)
            lam = mu[:, k] / REGULAR * np.exp(params.log_g[t // 5] + beta[delta + 2, b])
            scored = rng.poisson(lam)
            for m in np.flatnonzero(scored):
                rows.extend(
                    [{"understat_match_id": m, "minute": t, "side": "ha"[k], "kind": "goal"}]
                    * int(scored[m])
                )
            goals[:, k] += scored
    return pd.DataFrame(rows, columns=["understat_match_id", "minute", "side", "kind"])


def test_fit_recovers_game_state_effects() -> None:
    rng = np.random.default_rng(0)
    beta = np.zeros((len(DELTAS), len(BUCKETS)))
    beta[DELTAS.index(-1), 3] = 0.5  # trailing by one late: scores more
    beta[DELTAS.index(1), 3] = -0.4  # leading by one late: scores less
    log_g = np.r_[np.linspace(-0.3, 0.2, N_G - 1), 0.0]
    truth = null_params(log_g=tuple(log_g), beta=tuple(tuple(r) for r in beta))
    mu = rng.uniform(0.8, 2.2, size=(6000, 2))
    events = simulate_events(truth, mu, seed=1)
    matches = pd.DataFrame(
        {"understat_match_id": range(len(mu)), "mu_home": mu[:, 0], "mu_away": mu[:, 1]}
    )
    shots = pd.DataFrame(columns=["understat_match_id", "minute", "side", "result", "situation"])
    cells = build_cells(matches, events, shots, NO_STOPPAGE)
    fitted = fit_goal_process(cells, "G4", NO_STOPPAGE)
    b = fitted.beta_table()
    assert abs(b[DELTAS.index(-1), 3] - 0.5) < 0.15
    assert abs(b[DELTAS.index(1), 3] + 0.4) < 0.15
    assert np.corrcoef(fitted.log_g[: N_G - 1], log_g[: N_G - 1])[0, 1] > 0.8


def test_stoppage_survival_is_monotone_and_bounded() -> None:
    minutes = np.r_[
        np.repeat(np.arange(80, 90), 100),
        np.repeat(90, 100),
        np.repeat(91, 120),
        np.repeat(93, 60),
        95,
    ]
    s = estimate_stoppage(pd.DataFrame({"minute": minutes})).survival
    assert s[0] == 1.0 and all(a >= b for a, b in itertools.pairwise(s))
    assert s[2] == 0.0  # slot 92 had no shots, so later slots cannot survive either


def test_in_play_start_late_in_a_level_game() -> None:
    nominal = np.array([[1.8, 0.9]])
    start = StartState(np.array([80]), np.array([1]), np.array([1]), np.array([0]), np.array([0]))
    sim = simulate_team(null_params(), nominal, 60_000, seed=5, start=start)
    assert (sim.home >= 1).all() and (sim.away >= 1).all()
    expected_none = np.exp(-(1.8 + 0.9) * 10 / REGULAR)
    assert abs((sim.next_goal == -1).mean() - expected_none) < 0.01


def test_red_cards_and_frailty_run() -> None:
    p = null_params(
        level="G5",
        red_own=-0.5,
        red_opp=0.5,
        red_h0=np.log(0.003),
        reds_on=True,
        log_frailty_shape=np.log(10.0),
        stoppage=Stoppage((1.0, 0.8, 0.5) + (0.0,) * 12),
    )
    sim = simulate_team(p, np.array([[1.4, 1.2], [1.4, 1.2]]), 20_000, seed=2)
    assert sim.home.shape == (2, 20_000)
    # common random numbers: identical inputs give identical draws
    assert (sim.home[0] == sim.home[1]).all()
    assert SLOTS == REGULAR + 15


@pytest.fixture(scope="module")
def emulator() -> Emulator:
    return Emulator.build(null_params(), grid=10, lo=0.3, hi=4.0, n_sims=20_000)


def test_emulator_reproduces_the_null_model(emulator: Emulator) -> None:
    for mh, ma in ((1.5, 1.1), (0.7, 2.6)):
        g = emulator.grid(mh, ma)
        assert abs(g.sum() - 1) < 1e-9
        assert np.abs(g - poisson_grid(mh, ma)).max() < 0.01
        xh, xa = emulator.nominal_for_means([mh], [ma])
        h, a = emulator.means_at(xh, xa)
        assert abs(h[0] - mh) < 1e-6 and abs(a[0] - ma) < 1e-6


def test_emulator_goal_model_and_round_trip(emulator: Emulator) -> None:
    model = emulator.as_goal_model()
    target = poisson_grid(1.4, 1.0)
    from fplh.models.goal_benchmarks import markets

    mk = markets(target)
    lh, la = model.invert([mk["home"], mk["draw"], mk["away"]])
    assert abs(lh - 1.4) < 0.08 and abs(la - 1.0) < 0.08
    again = Emulator.from_bytes(emulator.to_bytes())
    np.testing.assert_allclose(again.grid(1.2, 1.3), emulator.grid(1.2, 1.3))
    assert again.params == emulator.params


def test_params_round_trip() -> None:
    p = null_params(level="G5", red_own=-0.5, reds_on=True, log_frailty_shape=2.0)
    assert GoalProcessParams.from_dict(p.to_dict()) == p
    q = null_params()
    assert GoalProcessParams.from_dict(q.to_dict()) == q
