"""M1 reference model: full-posterior dynamic ratings with NumPyro NUTS (§7.2).

The production M1 is the Laplace filter in ``team_strength``: fast enough to refit at
every walk-forward deadline. This model is its ground truth. It samples the whole
rating path (non-centred Gaussian random walk over rounds, sum-to-zero) with the same
likelihood (Poisson goals + ω-weighted Gamma xG), and is used to

* check the model recovers known ratings from simulated data, and
* check the filter's ratings agree with a full refit on the same data (ticket 2.4).

Needs the optional ``[bayes]`` extra (NumPyro + JAX): ``uv sync --extra bayes``.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
import pandas as pd
from numpyro.infer import MCMC, NUTS


@dataclass
class NutsFit:
    teams: list[str]
    attack: np.ndarray  # posterior mean at the last round
    defence: np.ndarray
    home: float
    samples: dict[str, np.ndarray]


def _model(hi, ai, rnd, n_teams, n_rounds, hg, ag, hx, ax, omega, kappa):  # type: ignore[no-untyped-def]
    c = numpyro.sample("c", dist.Normal(0.3, 0.3))
    eta = numpyro.sample("eta", dist.Normal(0.2, 0.2))
    s_a = numpyro.sample("sigma_a", dist.HalfNormal(0.1))
    s_d = numpyro.sample("sigma_d", dist.HalfNormal(0.1))
    a0 = numpyro.sample("a0", dist.Normal(0, 0.4).expand([n_teams]))
    d0 = numpyro.sample("d0", dist.Normal(0, 0.4).expand([n_teams]))
    za = numpyro.sample("za", dist.Normal(0, 1).expand([n_rounds - 1, n_teams]))
    zd = numpyro.sample("zd", dist.Normal(0, 1).expand([n_rounds - 1, n_teams]))
    a = jnp.concatenate([a0[None], a0[None] + jnp.cumsum(s_a * za, axis=0)])
    d = jnp.concatenate([d0[None], d0[None] + jnp.cumsum(s_d * zd, axis=0)])
    a = a - a.mean(axis=1, keepdims=True)
    d = d - d.mean(axis=1, keepdims=True)
    numpyro.deterministic("attack", a)
    numpyro.deterministic("defence", d)
    lh = c + eta + a[rnd, hi] - d[rnd, ai]
    la = c + a[rnd, ai] - d[rnd, hi]
    numpyro.sample("hg", dist.Poisson(jnp.exp(lh)), obs=hg)
    numpyro.sample("ag", dist.Poisson(jnp.exp(la)), obs=ag)
    if omega > 0:
        gx = dist.Gamma(kappa, kappa / jnp.exp(lh)).log_prob(hx) + dist.Gamma(
            kappa, kappa / jnp.exp(la)
        ).log_prob(ax)
        numpyro.factor("xg", omega * jnp.sum(gx))


def fit_nuts(
    matches: pd.DataFrame,
    *,
    omega: float = 0.5,
    kappa: float = 4.0,
    warmup: int = 400,
    samples: int = 400,
    seed: int = 0,
) -> NutsFit:
    """``matches`` needs home_team, away_team, round (0-based), goals and xG columns."""
    teams = sorted(set(matches["home_team"]) | set(matches["away_team"]))
    idx = {t: i for i, t in enumerate(teams)}
    rounds = matches["round"].to_numpy()
    kernel = NUTS(_model, target_accept_prob=0.9)
    mcmc = MCMC(kernel, num_warmup=warmup, num_samples=samples, progress_bar=False)
    mcmc.run(
        jax.random.PRNGKey(seed),
        hi=jnp.asarray(matches["home_team"].map(idx).to_numpy()),
        ai=jnp.asarray(matches["away_team"].map(idx).to_numpy()),
        rnd=jnp.asarray(rounds),
        n_teams=len(teams),
        n_rounds=int(rounds.max()) + 1,
        hg=jnp.asarray(matches["home_goals"].to_numpy()),
        ag=jnp.asarray(matches["away_goals"].to_numpy()),
        hx=jnp.asarray(np.clip(matches["home_xg"].to_numpy(), 1e-3, None)),
        ax=jnp.asarray(np.clip(matches["away_xg"].to_numpy(), 1e-3, None)),
        omega=omega,
        kappa=kappa,
    )
    s = {k: np.asarray(v) for k, v in mcmc.get_samples().items()}
    return NutsFit(
        teams, s["attack"][:, -1].mean(0), s["defence"][:, -1].mean(0), float(s["eta"].mean()), s
    )
