from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
import pytest
from scipy.optimize import minimize

from fplh.models.team_strength import (
    TeamStrengthFilter,
    TeamStrengthParams,
    fit_hyperparameters,
    run_filter,
)

STATIC = TeamStrengthParams(
    eta=0.25,
    phi_year=1.0,
    sigma_a=0.02,
    sigma_d=0.02,
    season_regress=1.0,
    season_var=0.0,
    promoted_a=0.0,
    promoted_d=0.0,
    promoted_var=0.2,
    omega=0.0,
)


def simulate(
    n: int = 12, seasons: int = 3, seed: int = 0
) -> tuple[pd.DataFrame, list[str], np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    teams = [f"t{i:02d}" for i in range(n)]
    ta, td = rng.normal(0, 0.3, n), rng.normal(0, 0.25, n)
    ta, td = ta - ta.mean(), td - td.mean()
    rows = []
    start = pd.Timestamp("2020-08-01", tz="UTC")
    pairs = list(itertools.permutations(range(n), 2))
    for s in range(seasons):
        rng.shuffle(pairs)
        for k, (i, j) in enumerate(pairs):
            mh, ma = np.exp(0.3 + 0.25 + ta[i] - td[j]), np.exp(0.3 + ta[j] - td[i])
            rows.append(
                {
                    "season": f"s{s}",
                    "kickoff_at": start + pd.Timedelta(days=365 * s + k // (n // 2) * 3),
                    "home_team": teams[i],
                    "away_team": teams[j],
                    "home_goals": rng.poisson(mh),
                    "away_goals": rng.poisson(ma),
                    "home_xg": rng.gamma(4, mh / 4),
                    "away_xg": rng.gamma(4, ma / 4),
                }
            )
    return pd.DataFrame(rows), teams, ta, td


def test_static_filter_matches_batch_poisson_mle() -> None:
    m, teams, _, _ = simulate()
    f = run_filter(m, STATIC, {s: teams for s in m["season"].unique()})
    idx = {t: i for i, t in enumerate(teams)}
    hi, ai = m["home_team"].map(idx).to_numpy(), m["away_team"].map(idx).to_numpy()
    hg, ag = m["home_goals"].to_numpy(), m["away_goals"].to_numpy()

    def nll(x: np.ndarray) -> float:
        a, d = x[:12] - x[:12].mean(), x[12:24] - x[12:24].mean()
        lh, la = x[24] + x[25] + a[hi] - d[ai], x[24] + a[ai] - d[hi]
        return float(-(hg * lh - np.exp(lh)).sum() - (ag * la - np.exp(la)).sum())

    x = minimize(nll, np.zeros(26), method="L-BFGS-B").x
    r = f.ratings().set_index("team").loc[teams]
    assert np.corrcoef(r["attack"], x[:12] - x[:12].mean())[0, 1] > 0.999
    assert np.abs(r["attack"] - (x[:12] - x[:12].mean())).max() < 0.05


def test_xg_improves_recovery() -> None:
    m, teams, ta, td = simulate(seed=3)
    st = {s: teams for s in m["season"].unique()}
    goals = run_filter(m.head(132), STATIC, st).ratings().set_index("team").loc[teams]
    xg = (
        run_filter(m.head(132), STATIC.with_(omega=1.0, kappa=4.0), st)
        .ratings()
        .set_index("team")
        .loc[teams]
    )
    err_goals = np.abs(goals["attack"] - ta).mean() + np.abs(goals["defence"] - td).mean()
    err_xg = np.abs(xg["attack"] - ta).mean() + np.abs(xg["defence"] - td).mean()
    assert err_xg < err_goals


def test_uncertainty_grows_with_horizon() -> None:
    m, teams, _, _ = simulate(seasons=1)
    f = run_filter(m, TeamStrengthParams(), {"s0": teams})
    last = m["kickoff_at"].max()
    _, s1 = f.predict_eta("t00", "t01", last + pd.Timedelta(days=3))
    _, s6 = f.predict_eta("t00", "t01", last + pd.Timedelta(days=45))
    assert s6[0, 0] > s1[0, 0]
    assert f.t == last  # prediction does not move the filter


def test_sum_to_zero_and_promotion() -> None:
    m, teams, _, _ = simulate(seasons=1)
    p = TeamStrengthParams(promoted_a=-0.3, promoted_d=-0.2)
    f = run_filter(m, p, {"s0": teams}, teams=[*teams, "new"])
    r = f.ratings()
    active = r[r["active"]]
    assert abs(active["attack"].sum()) < 1e-9 and abs(active["defence"].sum()) < 1e-9
    f.start_season("s1", [*teams[1:], "new"])
    r2 = f.ratings().set_index("team")
    assert r2.loc["new", "attack"] < r2.loc[teams[1:], "attack"].mean()
    assert not r2.loc[teams[0], "active"]


def test_records_pre_update_predictions() -> None:
    m, teams, _, _ = simulate(seasons=1)
    f = run_filter(m, TeamStrengthParams(), {"s0": teams}, record=True)
    h = pd.DataFrame(f.history)
    assert len(h) == len(m)
    assert (h[["pred_home", "pred_away"]] > 0).all().all()
    assert f.n_scored == 0  # first season is burn-in for the likelihood


def test_filter_rejects_unknown_team() -> None:
    f = TeamStrengthFilter(TeamStrengthParams(), ["a", "b"])
    with pytest.raises(KeyError):
        f.start_season("s", ["a", "zzz"])


def test_championship_prior_orders_promoted_teams() -> None:
    from fplh.models.team_strength import championship_priors

    m, teams, ta, _ = simulate(n=8, seasons=1, seed=5)
    priors = championship_priors(m.assign(season="e1"), {"e1": "s1"})
    strongest = teams[int(np.argmax(ta))]
    weakest = teams[int(np.argmin(ta))]
    assert priors[("s1", strongest)][0] > priors[("s1", weakest)][0]
    p = TeamStrengthParams(promoted_a=-0.3, promoted_slope_a=0.5)
    f = TeamStrengthFilter(p, teams, championship=priors)
    f.start_season("s1", teams)
    r = f.ratings().set_index("team")
    assert r.loc[strongest, "attack"] > r.loc[weakest, "attack"]


def test_restarts_never_worsen_the_fit() -> None:
    matches, teams, _, _ = simulate(n=6, seasons=2)
    by_season = {s: teams for s in matches["season"].unique()}
    tune = ("sigma_a", "omega", "eta")
    _, obj_once = fit_hyperparameters(matches, by_season, tune=tune, maxiter=8)
    again, obj_again = fit_hyperparameters(matches, by_season, tune=tune, maxiter=8, restarts=2)
    default = run_filter(matches, TeamStrengthParams(), by_season)
    assert obj_again <= obj_once <= -default.log_lik / default.n_scored
    f = run_filter(matches, again, by_season)
    assert abs(-f.log_lik / f.n_scored - obj_again) < 1e-12  # reported objective is real
