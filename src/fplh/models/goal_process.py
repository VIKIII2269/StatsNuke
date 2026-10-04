"""In-match goal process G4–G6 (ARCHITECTURE.md §7.5, ladder §11.7).

Per side *k* and minute slot *t* the goal intensity is

    λ_k(t) = μ_k/90 · g(t) · exp(β(Δ_k(t), b(t)) + β_own·R_k(t) + β_opp·R_{−k}(t)) · ε

* μ_k: the match's pre-match expected goals (M1's pre-update prediction when fitting;
  the emulator's nominal rate when simulating);
* g(t): time profile, 18 five-minute bins plus one stoppage-time bin (log scale, with a
  random-walk penalty). First-half stoppage is recorded as minutes 45–50 and is absorbed
  by g; second-half stoppage slots 90+u exist with probability S(u) (``Stoppage``);
* β(Δ, b): game-state effect of goal difference Δ ∈ {−2..2} (clipped, Δ = 0 is the
  reference) in time buckets 0–30, 30–60, 60–75, 75+;
* R: red cards shown to each side (capped at 2);
* ε: match frailty shared by both sides, ε ~ Gamma(a, a) (mean 1, variance 1/a). The
  spec's log-normal frailty has the same role; the Gamma form integrates out exactly, so
  the likelihood and its gradient are closed-form (no quadrature).

Ladder (each step kept only if it beats the previous one on scoreline log loss):
G4 = g + β; G5 = G4 + red cards + frailty + red-card hazard; G6 = G5 with the game-state
effects estimated from shots (≈10× the events of goals) plus a shrunk per-state
conversion offset.

The red-card hazard (G5+) is ``h0 · g_R(third) · exp(γ·Δ)`` per side and slot, fitted on
the red-card minutes the Understat rosters give (silver ``fact_match_event``).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Literal

import numpy as np
import numpy.typing as npt
import pandas as pd
import yaml
from scipy.optimize import minimize
from scipy.special import digamma, gammaln

Array = npt.NDArray[np.float64]
Level = Literal["G4", "G5", "G6"]

REGULAR = 90
STOPPAGE_MAX = 15
SLOTS = REGULAR + STOPPAGE_MAX
N_G = REGULAR // 5 + 1  # 18 five-minute bins + stoppage
DELTAS = (-2, -1, 1, 2)
BUCKETS = (0, 30, 60, 75)  # lower edges; the last bucket runs to the end
RED_THIRDS = (0, 30, 60)


def g_bin(t: npt.NDArray[np.int64]) -> npt.NDArray[np.int64]:
    out: npt.NDArray[np.int64] = np.where(t < REGULAR, t // 5, N_G - 1)
    return out


def bucket(t: npt.NDArray[np.int64]) -> npt.NDArray[np.int64]:
    out: npt.NDArray[np.int64] = np.searchsorted(np.array(BUCKETS), t, side="right") - 1
    return out


@dataclass(frozen=True)
class Stoppage:
    """P(second-half stoppage slot 90+u is played), u = 0..STOPPAGE_MAX−1."""

    survival: tuple[float, ...]

    @property
    def mean_slots(self) -> float:
        return float(sum(self.survival))


def estimate_stoppage(shots: pd.DataFrame) -> Stoppage:
    """Shots per minute in slot 90+u relative to minutes 80–89: a slot only produces
    shots in matches that reach it, so the ratio estimates its survival."""
    minute = shots["minute"].to_numpy()
    base = np.mean([np.sum(minute == m) for m in range(80, 90)])
    s = [
        min(1.0, np.sum(minute == REGULAR + u) / base) if base else 0.0 for u in range(STOPPAGE_MAX)
    ]
    s = list(np.minimum.accumulate(s))  # survival cannot increase
    return Stoppage(tuple(round(float(x), 4) for x in s))


@dataclass
class Cells:
    """One row per (match, side, minute slot) with the state *before* that slot."""

    match: npt.NDArray[np.int64]
    side: npt.NDArray[np.int64]  # 0 home, 1 away
    gbin: npt.NDArray[np.int64]
    delta: npt.NDArray[np.int64]  # −2..2 from this side's view
    bucket: npt.NDArray[np.int64]
    third: npt.NDArray[np.int64]
    red_own: npt.NDArray[np.int64]
    red_opp: npt.NDArray[np.int64]
    log_exposure: Array
    log_mu: Array
    goals: Array
    shots: Array
    reds: Array
    n_matches: int


def build_cells(
    matches: pd.DataFrame, events: pd.DataFrame, shots: pd.DataFrame, stoppage: Stoppage
) -> Cells:
    """``matches``: understat_match_id, mu_home, mu_away (pre-match expected goals).
    ``events``: fact_match_event rows; ``shots``: fact_shot rows (non-penalty shots are
    counted for G6)."""
    mids = matches["understat_match_id"].to_numpy()
    index = {int(m): i for i, m in enumerate(mids)}
    n = len(mids)

    def per_slot(df: pd.DataFrame, minute: str, side_col: str) -> npt.NDArray[np.int64]:
        arr = np.zeros((n, 2, SLOTS), dtype=np.int64)
        keep = df["understat_match_id"].map(index).notna()
        d = df[keep]
        m = d["understat_match_id"].map(index).to_numpy(dtype=np.int64)
        s = (d[side_col] == "a").to_numpy(dtype=np.int64)
        t = np.clip(d[minute].to_numpy(dtype=np.int64), 0, SLOTS - 1)
        np.add.at(arr, (m, s, t), 1)
        return arr

    goals = per_slot(events[events["kind"].isin(["goal", "own_goal"])], "minute", "side")
    reds = per_slot(events[events["kind"] == "red"], "minute", "side")
    shot_rows = shots[(shots["result"] != "OwnGoal") & (shots["situation"] != "Penalty")]
    shot_counts = per_slot(shot_rows, "minute", "side")

    before = np.cumsum(goals, axis=2) - goals  # goals strictly before each slot
    reds_before = np.cumsum(reds, axis=2) - reds
    t = np.arange(SLOTS)
    exposure = np.where(t < REGULAR, 1.0, np.r_[np.ones(REGULAR), stoppage.survival][t])

    m_idx, s_idx, t_idx = np.meshgrid(np.arange(n), np.arange(2), t, indexing="ij")
    own, opp = s_idx, 1 - s_idx
    delta = before[m_idx, own, t_idx] - before[m_idx, opp, t_idx]
    mu = np.stack([matches["mu_home"].to_numpy(float), matches["mu_away"].to_numpy(float)], 1)
    keep = exposure[t_idx] > 0
    flat = {
        "match": m_idx[keep],
        "side": s_idx[keep],
        "t": t_idx[keep],
        "delta": np.clip(delta[keep], -2, 2),
        "red_own": np.minimum(reds_before[m_idx, own, t_idx][keep], 2),
        "red_opp": np.minimum(reds_before[m_idx, opp, t_idx][keep], 2),
        "goals": goals[m_idx, s_idx, t_idx][keep],
        "shots": shot_counts[m_idx, s_idx, t_idx][keep],
        "reds": reds[m_idx, s_idx, t_idx][keep],
    }
    tt = flat["t"]
    return Cells(
        match=flat["match"],
        side=flat["side"],
        gbin=g_bin(tt),
        delta=flat["delta"],
        bucket=bucket(tt),
        third=np.minimum(tt // 30, 2),
        red_own=flat["red_own"],
        red_opp=flat["red_opp"],
        log_exposure=np.log(exposure[tt]),
        log_mu=np.log(mu[flat["match"], flat["side"]] / REGULAR),
        goals=flat["goals"].astype(float),
        shots=flat["shots"].astype(float),
        reds=flat["reds"].astype(float),
        n_matches=n,
    )


def _beta_index(delta: npt.NDArray[np.int64], b: npt.NDArray[np.int64]) -> npt.NDArray[np.int64]:
    """Index into the (len(DELTAS) × len(BUCKETS)) β table; −1 for Δ = 0."""
    d_idx = np.searchsorted(np.array(DELTAS), delta)
    out: npt.NDArray[np.int64] = np.where(delta == 0, -1, d_idx * len(BUCKETS) + b)
    return out


@dataclass
class _Fit:
    """Parameter vector layout and the penalised negative log-likelihood."""

    cells: Cells
    counts: Array
    frailty: bool
    reds: bool
    fixed_beta: Array | None = None  # G6: β from the shot model, plus a per-Δ offset
    offset: bool = True  # use log μ
    rw: float = 50.0  # random-walk penalty on log g (regular bins)
    ridge: float = 1.0

    def __post_init__(self) -> None:
        c = self.cells
        self.bidx = _beta_index(c.delta, c.bucket)
        self.d_idx = np.where(c.delta == 0, -1, np.searchsorted(np.array(DELTAS), c.delta))
        self.n_beta = len(DELTAS) if self.fixed_beta is not None else len(DELTAS) * len(BUCKETS)
        self.sizes = [N_G, self.n_beta, 2 if self.reds else 0, 1 if self.frailty else 0]
        self.base = c.log_exposure + (c.log_mu if self.offset else 0.0)
        if self.fixed_beta is not None:
            fb = np.r_[self.fixed_beta, 0.0]
            self.base = self.base + fb[self.bidx]  # bidx −1 → appended 0
        self.n_match_counts = np.bincount(c.match, weights=self.counts, minlength=c.n_matches)

    def split(self, x: Array) -> tuple[Array, Array, Array, float]:
        i = np.cumsum([0, *self.sizes])
        lg, beta, red = x[i[0] : i[1]], x[i[1] : i[2]], x[i[2] : i[3]]
        log_a = float(x[i[3]]) if self.frailty else np.inf
        return lg, beta, red, log_a

    def eta(self, x: Array) -> Array:
        lg, beta, red, _ = self.split(x)
        c = self.cells
        e = self.base + lg[c.gbin]
        idx = self.d_idx if self.fixed_beta is not None else self.bidx
        e = e + np.where(idx >= 0, np.r_[beta, 0.0][idx], 0.0)
        if self.reds:
            e = e + red[0] * c.red_own + red[1] * c.red_opp
        out: Array = e
        return out

    def nll(self, x: Array) -> tuple[float, Array]:
        c = self.cells
        lg, beta, _, log_a = self.split(x)
        eta = self.eta(x)
        lam = np.exp(eta)
        y = self.counts
        if self.frailty:
            a = np.exp(log_a)
            big_l = np.bincount(c.match, weights=lam, minlength=c.n_matches)
            nm = self.n_match_counts
            ll = float(
                np.sum(y * eta)
                + np.sum(
                    gammaln(nm + a) - gammaln(a) + a * np.log(a) - (nm + a) * np.log(a + big_l)
                )
            )
            w = ((nm + a) / (a + big_l))[c.match]
            d_a = float(
                np.sum(
                    digamma(nm + a)
                    - digamma(a)
                    + np.log(a)
                    + 1
                    - np.log(a + big_l)
                    - (nm + a) / (a + big_l)
                )
            )
            g_log_a = -d_a * a
        else:
            ll = float(np.sum(y * eta - lam))
            w = np.ones_like(lam)
            g_log_a = 0.0
        resid = y - w * lam  # dℓ/dη
        grad = [np.bincount(c.gbin, weights=resid, minlength=N_G)]
        idx = self.d_idx if self.fixed_beta is not None else self.bidx
        ok = idx >= 0
        grad.append(np.bincount(idx[ok], weights=resid[ok], minlength=self.n_beta))
        if self.reds:
            grad.append(np.array([np.sum(resid * c.red_own), np.sum(resid * c.red_opp)]))
        if self.frailty:
            grad.append(np.array([-g_log_a]))
        g = -np.concatenate(grad)
        # penalties
        d = np.diff(lg[: N_G - 1])
        pen = self.rw * float(np.sum(d**2)) + self.ridge * float(np.sum(beta**2))
        g_lg = np.zeros(N_G)
        g_lg[: N_G - 1] += 2 * self.rw * (np.r_[0.0, d] - np.r_[d, 0.0])
        g[:N_G] += g_lg
        g[N_G : N_G + self.n_beta] += 2 * self.ridge * beta
        if self.frailty:
            g[-1] += 1e-3 * 2 * log_a  # weak pull towards a = 1 keeps a finite if σ → 0
            pen += 1e-3 * log_a**2
        return -ll + pen, g

    def run(self, x0: Array | None = None) -> Array:
        x = np.zeros(sum(self.sizes)) if x0 is None else x0
        if x0 is None and self.frailty:
            x[-1] = np.log(20.0)
        res = minimize(self.nll, x, jac=True, method="L-BFGS-B", options={"maxiter": 2000})
        out: Array = res.x
        return out


@dataclass(frozen=True)
class GoalProcessParams:
    level: Level
    log_g: tuple[float, ...]
    beta: tuple[tuple[float, ...], ...]  # [Δ index][bucket]
    red_own: float = 0.0
    red_opp: float = 0.0
    log_frailty_shape: float = float("inf")  # log a; inf = no frailty
    red_h0: float = 0.0  # log hazard per slot (G5+); 0 with red_gamma 0 = no red process
    red_third: tuple[float, ...] = (0.0, 0.0, 0.0)
    red_gamma: float = 0.0
    reds_on: bool = False
    stoppage: Stoppage = field(default_factory=lambda: Stoppage((1.0,) * 3))
    trained_before: str = ""

    def beta_table(self) -> Array:
        out: Array = np.array(self.beta, dtype=float)
        return out

    @property
    def frailty_variance(self) -> float:
        return (
            0.0
            if not np.isfinite(self.log_frailty_shape)
            else float(np.exp(-self.log_frailty_shape))
        )

    def to_dict(self) -> dict[str, object]:
        d = asdict(self)
        d["stoppage"] = list(self.stoppage.survival)
        d["log_g"], d["red_third"] = list(self.log_g), list(self.red_third)
        d["beta"] = [list(r) for r in self.beta]
        d["log_frailty_shape"] = (
            None if not np.isfinite(self.log_frailty_shape) else self.log_frailty_shape
        )
        return d

    @classmethod
    def from_dict(cls, d: dict[str, object]) -> GoalProcessParams:
        x = dict(d)
        x["stoppage"] = Stoppage(tuple(float(v) for v in x["stoppage"]))  # type: ignore[attr-defined]
        x["log_g"] = tuple(float(v) for v in x["log_g"])  # type: ignore[attr-defined]
        x["red_third"] = tuple(float(v) for v in x["red_third"])  # type: ignore[attr-defined]
        x["beta"] = tuple(tuple(float(v) for v in r) for r in x["beta"])  # type: ignore[attr-defined]
        lfs = x.get("log_frailty_shape")
        x["log_frailty_shape"] = float("inf") if lfs is None else float(lfs)  # type: ignore[arg-type]
        return cls(**x)  # type: ignore[arg-type]

    def dump(self) -> str:
        return yaml.safe_dump(self.to_dict(), sort_keys=False)


def fit_red_hazard(cells: Cells) -> tuple[float, tuple[float, ...], float]:
    """Poisson GLM for red cards per side and slot: log h = h0 + third effect + γ·Δ."""
    c = cells

    def nll(x: Array) -> tuple[float, Array]:
        h0, t1, t2, gam = x
        th = np.array([0.0, t1, t2])[c.third]
        eta = c.log_exposure + h0 + th + gam * c.delta
        lam = np.exp(eta)
        r = c.reds - lam
        ll = float(np.sum(c.reds * eta - lam))
        g = -np.array([r.sum(), r[c.third == 1].sum(), r[c.third == 2].sum(), np.sum(r * c.delta)])
        return -ll + 0.5 * float(np.sum(x[1:] ** 2)), g + np.r_[0.0, x[1:]]

    x = minimize(nll, np.array([np.log(0.0015), 0.0, 0.0, 0.0]), jac=True, method="L-BFGS-B").x
    return float(x[0]), (0.0, float(x[1]), float(x[2])), float(x[3])


def fit_goal_process(
    cells: Cells, level: Level, stoppage: Stoppage, trained_before: str = ""
) -> GoalProcessParams:
    reds = level in ("G5", "G6")
    if level == "G6":
        shot = _Fit(cells, cells.shots, frailty=True, reds=True, offset=True)
        xs = shot.run()
        _, beta_shot, _, _ = shot.split(xs)
        fit = _Fit(cells, cells.goals, frailty=True, reds=True, fixed_beta=beta_shot, ridge=20.0)
        x = fit.run()
        lg, d_off, red, log_a = fit.split(x)
        beta = beta_shot.reshape(len(DELTAS), len(BUCKETS)) + d_off[:, None]
    else:
        fit = _Fit(cells, cells.goals, frailty=reds, reds=reds)
        x = fit.run()
        lg, b, red, log_a = fit.split(x)
        beta = b.reshape(len(DELTAS), len(BUCKETS))
    params: dict[str, object] = {
        "level": level,
        "log_g": tuple(float(v) for v in lg),
        "beta": tuple(tuple(float(v) for v in row) for row in beta),
        "stoppage": stoppage,
        "trained_before": trained_before,
    }
    if reds:
        h0, third, gam = fit_red_hazard(cells)
        params.update(
            red_own=float(red[0]),
            red_opp=float(red[1]),
            log_frailty_shape=float(log_a),
            red_h0=h0,
            red_third=third,
            red_gamma=gam,
            reds_on=True,
        )
    return GoalProcessParams(**params)  # type: ignore[arg-type]


def dispersion(cells: Cells, seasons: npt.NDArray[np.str_]) -> pd.DataFrame:
    """Conditional Pearson φ̂ of total goals per match against the pre-match Poisson mean
    (§11.6): φ̂ > 1 over-dispersed, < 1 under-dispersed. ``seasons`` per match."""
    c = cells
    mu = np.exp(c.log_mu + c.log_exposure)
    expected = np.bincount(c.match, weights=mu, minlength=c.n_matches)
    observed = np.bincount(c.match, weights=c.goals, minlength=c.n_matches)
    # scale expectation to the observed total (the offset's per-90 base ignores g)
    expected *= observed.sum() / expected.sum()
    df = pd.DataFrame({"season": seasons, "z2": (observed - expected) ** 2 / expected})
    out: pd.DataFrame = df.groupby("season")["z2"].mean().rename("phi").reset_index()
    return out
