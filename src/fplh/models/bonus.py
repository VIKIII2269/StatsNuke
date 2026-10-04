"""M10 bonus (ARCHITECTURE.md §7.10): BPS reconstructed from simulated events, then the
official bonus allocation.

    BPS = Σ_c w_c · x_c(events) + μ_g + δ_p + σ_g · ε,   ε ~ N(0, 1), rounded

* x_c: counts of the events the simulator generates (minutes bands, goals by position,
  assists, clean sheets and goals conceded for GK/DEF, saves, penalty saves and misses,
  cards, own goals);
* w_c: *effective* weights, least squares of official BPS on x over the last
  ``window_seasons`` seasons observable at the deadline (BPS tables change between
  seasons). They exceed the official table where an event brings correlated actions the
  table scores separately (a goal usually also brings a shot on target);
* μ_g, σ_g: residual mean and spread by group g = position × (60+ minutes);
* δ_p: the player's decayed mean residual (passing, key passes, recoveries … that the
  simulator does not generate), shrunk with n₀ = σ²/τ² pseudo-matches (τ² the between-player
  variance, method of moments).

Bonus comes from ``rules.bonus.assign_bonus_array`` over both sides' players with
``eligible = minutes > 0``: players who did not play never get bonus.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt
import pandas as pd

from fplh.features.information_set import InformationSet
from fplh.models.shrinkage import decay

Array = npt.NDArray[np.float64]
POSITIONS = ("GK", "DEF", "MID", "FWD")
DESIGN = (
    "minutes_lt_60",
    "minutes_gte_60",
    "goal_GK",
    "goal_DEF",
    "goal_MID",
    "goal_FWD",
    "assist",
    "clean_sheet",
    "goal_conceded",
    "save",
    "penalty_save",
    "penalty_miss",
    "yellow_card",
    "red_card",
    "own_goal",
)


def bps_design(events: Mapping[str, npt.ArrayLike], position: npt.ArrayLike) -> dict[str, Array]:
    """Design columns for events of any shape (e.g. sims × players). ``clean_sheet`` is
    FPL's: 60+ minutes and nothing conceded while on the pitch, for GK and DEF."""
    pos = np.asarray(position).astype(str)
    minutes = np.asarray(events["minutes"], dtype=float)
    pos = np.broadcast_to(pos, minutes.shape)
    gk_def = np.isin(pos, ["GK", "DEF"])
    conceded = np.asarray(events["goals_conceded"], dtype=float)
    goals = np.asarray(events["goals_scored"], dtype=float)
    out = {
        "minutes_lt_60": ((minutes > 0) & (minutes < 60)).astype(float),
        "minutes_gte_60": (minutes >= 60).astype(float),
        **{f"goal_{p}": goals * (pos == p) for p in ("GK", "DEF", "MID", "FWD")},
        "assist": np.asarray(events["assists"], dtype=float),
        "clean_sheet": ((minutes >= 60) & (conceded == 0) & gk_def).astype(float),
        "goal_conceded": conceded * gk_def,
        "save": np.asarray(events["saves"], dtype=float),
        "penalty_save": np.asarray(events["penalties_saved"], dtype=float),
        "penalty_miss": np.asarray(events["penalties_missed"], dtype=float),
        "yellow_card": np.asarray(events["yellow_cards"], dtype=float),
        "red_card": np.asarray(events["red_cards"], dtype=float),
        "own_goal": np.asarray(events["own_goals"], dtype=float),
    }
    return {c: np.broadcast_to(out[c], minutes.shape).astype(float) for c in DESIGN}


def _group(position: npt.ArrayLike, minutes: npt.ArrayLike) -> npt.NDArray[np.str_]:
    band = np.where(np.asarray(minutes) >= 60, "60", "lt60")
    out: npt.NDArray[np.str_] = np.char.add(
        np.char.add(np.asarray(position).astype(str), ":"), band
    )
    return out


@dataclass
class BonusModel:
    weights: dict[str, float] = field(default_factory=dict)
    mu: dict[str, float] = field(default_factory=dict)  # by group
    sigma: dict[str, float] = field(default_factory=dict)
    players: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))  # δ_p

    def player_effect(self, player_uids: npt.ArrayLike) -> Array:
        out: Array = self.players.reindex(np.asarray(player_uids)).fillna(0.0).to_numpy(dtype=float)
        return out

    def _table(self, values: dict[str, float], default: float) -> Array:
        """(4 positions, 2 bands: < 60, 60+) lookup table."""
        out: Array = np.array(
            [[values.get(f"{p}:{b}", default) for b in ("lt60", "60")] for p in POSITIONS]
        )
        return out

    def expected_bps(
        self,
        events: Mapping[str, npt.ArrayLike],
        position: npt.ArrayLike,
        player_uids: npt.ArrayLike,
    ) -> tuple[Array, Array]:
        """Mean BPS and residual spread for events of shape (..., players)."""
        x = bps_design(events, position)
        minutes = np.asarray(events["minutes"])
        mean: Array = sum(
            (self.weights.get(c, 0.0) * x[c] for c in DESIGN), np.zeros(minutes.shape)
        )
        pos = np.broadcast_to(np.asarray(position).astype(str), minutes.shape)
        p_idx = np.select([pos == p for p in POSITIONS], range(len(POSITIONS)), 2)
        band = (minutes >= 60).astype(int)
        mean = mean + self._table(self.mu, 0.0)[p_idx, band] + self.player_effect(player_uids)
        sd: Array = self._table(self.sigma, 5.0)[p_idx, band]
        return mean, sd

    def sample_bps(
        self,
        events: Mapping[str, npt.ArrayLike],
        position: npt.ArrayLike,
        player_uids: npt.ArrayLike,
        rng: np.random.Generator,
    ) -> Array:
        mean, sd = self.expected_bps(events, position, player_uids)
        out: Array = np.round(mean + sd * rng.standard_normal(mean.shape))
        return out


def bps_rows(info: InformationSet) -> pd.DataFrame:
    pm = info.table("fact_player_match")
    if pm.empty:
        return pm
    out: pd.DataFrame = pm[(pm["minutes"] > 0) & pm["player_uid"].notna()].copy()
    return out


def _design_frame(rows: pd.DataFrame) -> pd.DataFrame:
    events = {c: rows[c].to_numpy() for c in rows.columns if c != "position"}
    x = bps_design(events, rows["position"].to_numpy())
    return pd.DataFrame(x, index=rows.index)


def fit_bonus(
    info: InformationSet, window_seasons: int = 2, half_life_days: float = 365.0
) -> BonusModel:
    rows = bps_rows(info)
    if len(rows) < 200:
        return BonusModel()
    seasons = sorted(rows["season"].unique())[-window_seasons:]
    recent = rows[rows["season"].isin(seasons)]
    x = _design_frame(recent)
    keep = [c for c in DESIGN if x[c].abs().sum() > 0]
    beta, *_ = np.linalg.lstsq(x[keep].to_numpy(), recent["bps"].to_numpy(float), rcond=None)
    weights = {c: float(b) for c, b in zip(keep, beta, strict=True)}

    xa = _design_frame(rows)
    fitted = sum((weights.get(c, 0.0) * xa[c] for c in DESIGN), pd.Series(0.0, index=rows.index))
    resid = rows["bps"].astype(float) - fitted
    g = pd.Series(_group(rows["position"].to_numpy(), rows["minutes"].to_numpy()), index=rows.index)
    in_window = rows["season"].isin(seasons)
    mu = resid[in_window].groupby(g[in_window]).mean()
    dev = resid - g.map(mu).fillna(0.0)
    w = decay(info, rows["kickoff_at"], half_life_days)
    per = pd.DataFrame({"wd": w * dev, "w": w, "uid": rows["player_uid"]}).groupby("uid").sum()
    sigma2 = float(dev[in_window].var())
    n_eff = per["w"]
    means = per["wd"] / n_eff.clip(lower=1e-9)
    big = n_eff >= 10
    tau2 = float(means[big].var() - (sigma2 / n_eff[big]).mean()) if big.sum() > 5 else 0.0
    n0 = sigma2 / tau2 if tau2 > 0 else np.inf
    delta = per["wd"] / (n_eff + n0) if np.isfinite(n0) else per["wd"] * 0.0
    within = dev - rows["player_uid"].map(delta).fillna(0.0)
    sigma = within[in_window].groupby(g[in_window]).std()
    return BonusModel(
        weights,
        {str(k): float(v) for k, v in mu.items()},
        {str(k): float(v) for k, v in sigma.items()},
        delta,
    )
