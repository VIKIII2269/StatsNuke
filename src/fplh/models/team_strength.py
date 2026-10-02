"""M1: dynamic, mean-reverting team attack/defence ratings (ARCHITECTURE.md §7.2).

    log μ_home = c + η + a_home − d_away        log μ_away = c + a_away − d_home

State ``θ = (c, a_1..a_N, d_1..d_N)`` is Gaussian. Between matches ratings revert toward 0
by φ per year (``φ^(Δ/365)``) and diffuse with variance ``σ² Δ/365``; the league scoring
level ``c`` drifts slowly. Each match updates the state with a Laplace step in its
2-D log-rate space ``η = Hθ``:

* likelihood: Poisson(goals | e^η) + ω · Gamma(xG | shape κ, mean e^η);
* Newton iterations find the posterior mode η* and curvature, giving η ~ N(η*, V);
* the full state is then conditioned exactly on that Gaussian (θ | η is linear-Gaussian),
  so correlations between teams propagate.

Season transitions regress returning teams toward 0, add variance, and start promoted
teams from a learned "promoted" prior. Sum-to-zero over the active teams identifies the
ratings. Forecasts h rounds ahead propagate the state to the kickoff, so uncertainty
grows with distance exactly as in §7.2's horizon formula.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass, field, replace

import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy.optimize import minimize
from scipy.special import gammaln
from threadpoolctl import threadpool_limits

Array = npt.NDArray[np.float64]
DAY = 1.0 / 365.0


@dataclass(frozen=True)
class TeamStrengthParams:
    c0: float = 0.30  # initial league log-goal level
    eta: float = 0.20  # home advantage (log scale)
    phi_year: float = 0.80  # mean reversion per year
    sigma_a: float = 0.20  # attack diffusion per √year
    sigma_d: float = 0.20  # defence diffusion per √year
    sigma_c: float = 0.05  # league-level drift per √year
    omega: float = 0.5  # weight of the xG likelihood (0 = goals only)
    kappa: float = 4.0  # Gamma shape of xG given μ (Var = μ²/κ)
    season_regress: float = 0.8  # summer shrink of returning teams
    season_var: float = 0.02  # summer variance added (squad churn)
    promoted_a: float = -0.25  # promoted-team prior means
    promoted_d: float = -0.20
    promoted_var: float = 0.03
    promoted_slope_a: float = 0.0  # weight on the team's Championship attack rating
    promoted_slope_d: float = 0.0  # (0 = one shared promoted prior)
    init_var: float = 0.1

    def with_(self, **kw: float) -> TeamStrengthParams:
        return replace(self, **kw)


# Hyper-parameters tuned by predictive likelihood, with bounds (in natural units).
TUNABLE: dict[str, tuple[float, float]] = {
    "eta": (0.0, 0.5),
    "phi_year": (0.2, 1.0),
    "sigma_a": (0.02, 0.8),
    "sigma_d": (0.02, 0.8),
    "omega": (0.0, 3.0),
    "kappa": (0.5, 20.0),
    "season_regress": (0.3, 1.0),
    "season_var": (0.0, 0.2),
    "promoted_a": (-0.8, 0.2),
    "promoted_d": (-0.8, 0.2),
    "promoted_slope_a": (0.0, 1.5),
    "promoted_slope_d": (0.0, 1.5),
}


@dataclass
class TeamStrengthFilter:
    params: TeamStrengthParams
    teams: list[str]
    # (season, team) → Championship (attack, defence) from the season before promotion
    championship: Mapping[tuple[str, str], tuple[float, float]] = field(default_factory=dict)
    m: Array = field(init=False)
    P: Array = field(init=False)
    t: pd.Timestamp | None = field(default=None, init=False)
    season: str | None = field(default=None, init=False)
    active: npt.NDArray[np.bool_] = field(init=False)
    log_lik: float = field(default=0.0, init=False)
    n_scored: int = field(default=0, init=False)
    record: bool = field(default=False)
    history: list[dict[str, object]] = field(default_factory=list, init=False)

    def __post_init__(self) -> None:
        n = len(self.teams)
        self.index = {t: i for i, t in enumerate(self.teams)}
        self.m = np.zeros(1 + 2 * n)
        self.m[0] = self.params.c0
        self.P = np.eye(1 + 2 * n) * self.params.init_var
        self.P[0, 0] = 0.05
        self.active = np.zeros(n, dtype=bool)

    # -- state indexing -------------------------------------------------------------------
    def ia(self, team: str) -> int:
        return 1 + self.index[team]

    def id_(self, team: str) -> int:
        return 1 + len(self.teams) + self.index[team]

    def _H(self, home: str, away: str) -> Array:
        h = np.zeros((2, len(self.m)))
        h[0, 0] = h[1, 0] = 1.0
        h[0, self.ia(home)] = 1.0
        h[0, self.id_(away)] = -1.0
        h[1, self.ia(away)] = 1.0
        h[1, self.id_(home)] = -1.0
        return h

    # -- dynamics -------------------------------------------------------------------------
    def _center(self) -> None:
        n = len(self.teams)
        act = np.flatnonzero(self.active)
        if len(act) == 0:
            return
        c = np.eye(len(self.m))
        for off in (1, 1 + n):
            idx = off + act
            c[np.ix_(idx, idx)] -= 1.0 / len(act)
        self.m = c @ self.m
        self.P = c @ self.P @ c.T

    def start_season(self, season: str, teams: Iterable[str]) -> None:
        p = self.params
        n = len(self.teams)
        new_active = np.zeros(n, dtype=bool)
        for t in teams:
            new_active[self.index[t]] = True
        for i in np.flatnonzero(new_active):
            ia, id_ = 1 + i, 1 + n + i
            if self.active[i]:  # returning team: shrink toward 0, add churn variance
                for j in (ia, id_):
                    self.m[j] *= p.season_regress
                    self.P[j, :] *= p.season_regress
                    self.P[:, j] *= p.season_regress
                    self.P[j, j] += p.season_var
            else:  # promoted (or first appearance): prior from the Championship if known
                e1_a, e1_d = self.championship.get((season, self.teams[i]), (0.0, 0.0))
                prior = (
                    (ia, p.promoted_a + p.promoted_slope_a * e1_a),
                    (id_, p.promoted_d + p.promoted_slope_d * e1_d),
                )
                for j, mean in prior:
                    self.P[j, :] = 0.0
                    self.P[:, j] = 0.0
                    self.m[j] = mean
                    self.P[j, j] = p.promoted_var
        self.active = new_active
        self.season = season
        self._center()

    def propagate(self, when: pd.Timestamp) -> None:
        if self.t is None:
            self.t = when
            return
        days = max((when - self.t).total_seconds() / 86400.0, 0.0)
        if days == 0:
            return
        p = self.params
        n = len(self.teams)
        f = np.ones(len(self.m))
        q = np.zeros(len(self.m))
        decay = p.phi_year ** (days * DAY)
        act = np.flatnonzero(self.active)
        f[1 + act] = decay
        f[1 + n + act] = decay
        q[0] = p.sigma_c**2 * days * DAY
        q[1 + act] = p.sigma_a**2 * days * DAY
        q[1 + n + act] = p.sigma_d**2 * days * DAY
        self.m = f * self.m
        self.P = (f[:, None] * self.P * f[None, :]) + np.diag(q)
        self.t = when

    # -- prediction and update ---------------------------------------------------------------
    def predict_eta(
        self, home: str, away: str, when: pd.Timestamp | None = None
    ) -> tuple[Array, Array]:
        """Mean and covariance of (log μ_home, log μ_away) at ``when`` (no state change)."""
        if when is not None and self.t is not None and when > self.t:
            saved = (self.m.copy(), self.P.copy(), self.t)
            self.propagate(when)
            h = self._H(home, away)
            out = (h @ self.m + np.array([self.params.eta, 0.0]), h @ self.P @ h.T)
            self.m, self.P, self.t = saved
            return out
        h = self._H(home, away)
        return h @ self.m + np.array([self.params.eta, 0.0]), h @ self.P @ h.T

    def expected_goals(
        self, home: str, away: str, when: pd.Timestamp | None = None
    ) -> tuple[float, float]:
        """E[μ] under the lognormal predictive (includes rating uncertainty)."""
        eta0, s = self.predict_eta(home, away, when)
        return float(math.exp(eta0[0] + s[0, 0] / 2)), float(math.exp(eta0[1] + s[1, 1] / 2))

    def update(
        self,
        home: str,
        away: str,
        when: pd.Timestamp,
        goals: tuple[int, int],
        xg: tuple[float, float] | None = None,
        *,
        score: bool = True,
    ) -> None:
        self.propagate(when)
        p = self.params
        h = self._H(home, away)
        eta0 = h @ self.m + np.array([p.eta, 0.0])
        s = h @ self.P @ h.T
        y = np.asarray(goals, dtype=float)
        x = np.asarray(xg if xg is not None else (np.nan, np.nan), dtype=float)
        use_x = np.isfinite(x) & (x > 0) & (p.omega > 0)

        mu_pred = np.exp(eta0 + np.diag(s) / 2)
        if score:  # one-step-ahead predictive log-likelihood (before seeing the result)
            self.log_lik += float(np.sum(y * np.log(mu_pred) - mu_pred - gammaln(y + 1)))
            self.n_scored += 1
        if self.record:  # pre-update predictions: leak-free training data downstream
            self.history.append(
                {
                    "season": self.season,
                    "kickoff_at": when,
                    "home_team": home,
                    "away_team": away,
                    "pred_home": float(mu_pred[0]),
                    "pred_away": float(mu_pred[1]),
                    "home_goals": int(y[0]),
                    "away_goals": int(y[1]),
                }
            )

        s_inv = np.linalg.inv(s)
        eta = eta0.copy()
        for _ in range(20):
            mu = np.exp(eta)
            grad = y - mu
            w = mu.copy()
            if use_x.any():
                inv = np.where(use_x, x / mu, 0.0)
                grad = grad + np.where(use_x, p.omega * p.kappa * (inv - 1.0), 0.0)
                w = w + np.where(use_x, p.omega * p.kappa * inv, 0.0)
            hess = s_inv + np.diag(w)
            step = np.linalg.solve(hess, grad - s_inv @ (eta - eta0))
            eta = eta + step
            if np.max(np.abs(step)) < 1e-9:
                break
        mu = np.exp(eta)
        w = mu + np.where(use_x, p.omega * p.kappa * np.where(use_x, x / mu, 0.0), 0.0)
        v = np.linalg.inv(s_inv + np.diag(w))
        gain = self.P @ h.T @ s_inv  # D × 2
        self.m = self.m + gain @ (eta - eta0)
        self.P = self.P - gain @ (s - v) @ gain.T
        self.P = (self.P + self.P.T) / 2
        self._center()

    def ratings(self) -> pd.DataFrame:
        n = len(self.teams)
        sd = np.sqrt(np.diag(self.P))
        return pd.DataFrame(
            {
                "team": self.teams,
                "attack": self.m[1 : 1 + n],
                "defence": self.m[1 + n :],
                "attack_sd": sd[1 : 1 + n],
                "defence_sd": sd[1 + n :],
                "active": self.active,
            }
        )


MATCH_COLUMNS = [
    "season",
    "kickoff_at",
    "home_team",
    "away_team",
    "home_goals",
    "away_goals",
    "home_xg",
    "away_xg",
]


def run_filter(
    matches: pd.DataFrame,
    params: TeamStrengthParams,
    season_teams: Mapping[str, Iterable[str]],
    *,
    teams: list[str] | None = None,
    score_from_season: str | None = None,
    record: bool = False,
    championship: Mapping[tuple[str, str], tuple[float, float]] | None = None,
) -> TeamStrengthFilter:
    """Filter through ``matches`` (sorted by kickoff). Scores the one-step-ahead
    likelihood from ``score_from_season`` on (default: after the first season)."""
    df = matches.sort_values(["kickoff_at", "home_team"]).reset_index(drop=True)
    all_teams = teams or sorted(
        set(df["home_team"])
        | set(df["away_team"])
        | {t for ts in season_teams.values() for t in ts}
    )
    f = TeamStrengthFilter(params, all_teams, championship=championship or {}, record=record)
    first = str(df["season"].min()) if len(df) else ""
    score_from = score_from_season or first
    rows = zip(
        df["season"].astype(str),
        df["kickoff_at"],
        df["home_team"].astype(str),
        df["away_team"].astype(str),
        df["home_goals"].astype(int),
        df["away_goals"].astype(int),
        df["home_xg"].astype(float),
        df["away_xg"].astype(float),
        strict=True,
    )
    # Thousands of tiny linear-algebra calls: BLAS threads only contend (25x slower on 4
    # cores), so the filter runs single-threaded.
    with threadpool_limits(limits=1):
        for season, kickoff, home, away, hg, ag, hx, ax in rows:
            if season != f.season:
                f.start_season(season, season_teams[season])
            f.update(
                home,
                away,
                pd.Timestamp(kickoff),
                (hg, ag),
                (hx, ax),
                score=season >= score_from and season != first,
            )
    return f


def fit_hyperparameters(
    matches: pd.DataFrame,
    season_teams: Mapping[str, Iterable[str]],
    *,
    championship: Mapping[tuple[str, str], tuple[float, float]] | None = None,
    init: TeamStrengthParams | None = None,
    tune: Iterable[str] = tuple(TUNABLE),
    fixed: Mapping[str, float] | None = None,
    maxiter: int = 300,
    restarts: int = 0,
) -> tuple[TeamStrengthParams, float]:
    """Maximise the one-step-ahead predictive log-likelihood of goals (per match).

    Nelder–Mead's simplex collapses in this many dimensions before converging, so it is
    restarted from its own optimum up to ``restarts`` times while that still improves
    the objective by more than 1e-5."""
    base = (init or TeamStrengthParams()).with_(**(fixed or {}))
    names = [n for n in tune if n not in (fixed or {})]
    if not championship:  # slopes are unidentified without Championship ratings
        names = [n for n in names if not n.startswith("promoted_slope")]
    lo = np.array([TUNABLE[n][0] for n in names])
    hi = np.array([TUNABLE[n][1] for n in names])

    def unpack(z: Array) -> TeamStrengthParams:
        vals = lo + (hi - lo) / (1 + np.exp(-z))
        return base.with_(**dict(zip(names, map(float, vals), strict=True)))

    def objective(z: Array) -> float:
        f = run_filter(matches, unpack(z), season_teams, championship=championship)
        return -f.log_lik / max(f.n_scored, 1)

    start = np.array([getattr(base, n) for n in names])
    frac = np.clip((start - lo) / (hi - lo), 1e-3, 1 - 1e-3)
    z0 = np.log(frac / (1 - frac))
    best_z, best = z0, objective(z0)
    for _ in range(restarts + 1):
        res = minimize(
            objective,
            best_z,
            method="Nelder-Mead",
            options={"maxiter": maxiter, "xatol": 1e-3, "fatol": 1e-5},
        )
        improved = best - float(res.fun)
        if improved > 0:
            best_z, best = res.x, float(res.fun)
        if improved <= 1e-5:
            break
    return unpack(best_z), best


def params_to_dict(p: TeamStrengthParams) -> dict[str, float]:
    return {k: float(v) for k, v in asdict(p).items()}


def static_ratings(matches: pd.DataFrame) -> dict[str, tuple[float, float]]:
    """Season-level Poisson (attack, defence) ratings, sum-to-zero, with home advantage."""
    teams = sorted(set(matches["home_team"]) | set(matches["away_team"]))
    idx = {t: i for i, t in enumerate(teams)}
    n = len(teams)
    hi = matches["home_team"].map(idx).to_numpy()
    ai = matches["away_team"].map(idx).to_numpy()
    hg = matches["home_goals"].to_numpy(float)
    ag = matches["away_goals"].to_numpy(float)

    def nll(x: Array) -> float:
        a, d = x[:n] - x[:n].mean(), x[n : 2 * n] - x[n : 2 * n].mean()
        lh = x[-2] + x[-1] + a[hi] - d[ai]
        la = x[-2] + a[ai] - d[hi]
        return float(-(hg * lh - np.exp(lh)).sum() - (ag * la - np.exp(la)).sum()) + 1e-3 * float(
            x[: 2 * n] @ x[: 2 * n]
        )

    x = minimize(nll, np.zeros(2 * n + 2), method="L-BFGS-B").x
    a, d = x[:n] - x[:n].mean(), x[n : 2 * n] - x[n : 2 * n].mean()
    return {t: (float(a[i]), float(d[i])) for t, i in idx.items()}


def championship_priors(
    e1_matches: pd.DataFrame, next_season: Mapping[str, str]
) -> dict[tuple[str, str], tuple[float, float]]:
    """(EPL season, team) → Championship ratings from the season before.

    ``e1_matches`` holds E1 results (season, home_team, away_team, home_goals,
    away_goals); ``next_season`` maps an E1 season label to the following one.
    """
    out: dict[tuple[str, str], tuple[float, float]] = {}
    for season, g in e1_matches.dropna(subset=["home_goals", "away_goals"]).groupby("season"):
        target = next_season.get(str(season))
        if target is None:
            continue
        for team, ratings in static_ratings(g).items():
            out[(target, team)] = ratings
    return out
