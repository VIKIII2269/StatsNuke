"""NUTS reference model: recovery and agreement with the filter (ticket 2.4)."""

from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
import pytest

from fplh.models.team_strength import TeamStrengthParams, run_filter

pytest.importorskip("numpyro", reason="install the [bayes] extra for the NUTS reference model")
from fplh.models.team_strength_nuts import fit_nuts


def test_nuts_recovers_ratings_and_agrees_with_filter() -> None:
    rng = np.random.default_rng(1)
    n = 8
    teams = [f"t{i}" for i in range(n)]
    ta, td = rng.normal(0, 0.3, n), rng.normal(0, 0.25, n)
    ta, td = ta - ta.mean(), td - td.mean()
    sched = list(itertools.permutations(range(n), 2))
    rng.shuffle(sched)
    start = pd.Timestamp("2020-08-01", tz="UTC")
    rows = []
    for k, (i, j) in enumerate(sched):
        r = k // 4
        mh, ma = np.exp(0.3 + 0.25 + ta[i] - td[j]), np.exp(0.3 + ta[j] - td[i])
        rows.append(
            {
                "season": "s0",
                "round": r,
                "kickoff_at": start + pd.Timedelta(days=7 * r),
                "home_team": teams[i],
                "away_team": teams[j],
                "home_goals": rng.poisson(mh),
                "away_goals": rng.poisson(ma),
                "home_xg": rng.gamma(4, mh / 4),
                "away_xg": rng.gamma(4, ma / 4),
            }
        )
    m = pd.DataFrame(rows)
    fit = fit_nuts(m, warmup=300, samples=300)
    assert np.corrcoef(fit.attack, ta)[0, 1] > 0.8
    assert np.corrcoef(fit.defence, td)[0, 1] > 0.8
    f = run_filter(
        m,
        TeamStrengthParams(
            phi_year=1.0,
            sigma_a=0.05,
            sigma_d=0.05,
            promoted_a=0,
            promoted_d=0,
            promoted_var=0.16,
            omega=0.5,
        ),
        {"s0": teams},
    )
    r = f.ratings().set_index("team").loc[fit.teams]
    assert np.corrcoef(r["attack"], fit.attack)[0, 1] > 0.95
    assert np.corrcoef(r["defence"], fit.defence)[0, 1] > 0.95
    assert np.abs(r["attack"] - fit.attack).max() < 0.1
