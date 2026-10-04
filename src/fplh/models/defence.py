"""M7 defensive actions (ARCHITECTURE.md §7.7b): counts behind FPL's defensive contribution.

FPL publishes clearances/blocks/interceptions (CBI), tackles and recoveries only for
2016/17–2018/19 and from 2025/26; defensive contribution scores only from 2025/26, so the
model is validated walk-forward within 2018/19 and within 2025/26 (open question 1c, the
documented exception to the untouched holdout) and is active in the simulator only when
the season's rules award DC points.

* **Definition drift.** When both eras are observable, each (position, action) rate per 90
  is compared; if 2025/26+ differs from 2016/17–2018/19 by more than 10 %, the old counts
  are rescaled by the ratio (recoveries fell by about 30 %; midfield and forward CBI+T rose
  by 24–43 %).
* **Player rates.** Per action, a decayed Gamma–Poisson rate per 90, shrunk towards the
  player's position × role group.
* **Opponent.** A shrunk multiplier for the defensive actions teams make against each
  opponent (possession-heavy opponents force more).
* **Dispersion.** A per-player-match Gamma frailty shared by the three actions, so the
  group total is NegBin with size k per position (method of moments).

Not modelled (documented): game state, since action timing is not in our sources.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.stats import nbinom

from fplh.features.information_set import InformationSet
from fplh.models.shrinkage import decay, gamma_poisson, player_groups

ACTIONS = ("clearances_blocks_interceptions", "tackles", "recoveries")
OLD_ERA_END = "2018-19"
NEW_ERA_START = "2025-26"
DRIFT_TOLERANCE = 0.10
OPP_PSEUDO_MATCHES = 5.0
# the action group each position is scored on (FPL 2025/26 rules)
GROUP = {
    "GK": ACTIONS[:2],
    "DEF": ACTIONS[:2],
    "MID": ACTIONS,
    "FWD": ACTIONS,
}


@dataclass
class DefenceModel:
    players: pd.DataFrame = field(  # index player_uid: position, role, rate per action
        default_factory=lambda: pd.DataFrame(columns=["position", "role", *ACTIONS])
    )
    position_rate: dict[str, dict[str, float]] = field(default_factory=dict)
    opponent: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    size: dict[str, float] = field(default_factory=dict)  # NegBin k of the group total
    drift: dict[str, float] = field(default_factory=dict)  # "position:action" → factor

    def rates(self, player_uids: pd.Series, positions: pd.Series) -> np.ndarray:
        """(n, 3) per-90 rates; unseen players get their position's mean."""
        p = self.players.reindex(player_uids.to_numpy())
        fill = np.array(
            [[self.position_rate.get(str(x), {}).get(a, 0.0) for a in ACTIONS] for x in positions]
        )
        vals = p[list(ACTIONS)].to_numpy(dtype=float) if len(p) else np.empty((0, 3))
        out: np.ndarray = np.where(np.isnan(vals), fill, vals)
        return out

    def opponent_factor(self, opponents: pd.Series) -> np.ndarray:
        out: np.ndarray = self.opponent.reindex(opponents.to_numpy()).fillna(1.0).to_numpy()
        return out

    def group_size(self, positions: pd.Series) -> np.ndarray:
        return np.array([self.size.get(str(x), 20.0) for x in positions])

    def threshold_prob(
        self,
        player_uids: pd.Series,
        positions: pd.Series,
        opponents: pd.Series,
        minutes: np.ndarray,
        thresholds: np.ndarray,
    ) -> np.ndarray:
        """P(group total ≥ threshold) for players who played ``minutes``."""
        mean = self.group_mean(player_uids, positions, opponents, minutes)
        k = self.group_size(positions)
        p = k / (k + np.clip(mean, 1e-9, None))
        out: np.ndarray = nbinom.sf(np.asarray(thresholds) - 1, k, p)
        return out

    def group_mean(
        self,
        player_uids: pd.Series,
        positions: pd.Series,
        opponents: pd.Series,
        minutes: np.ndarray,
    ) -> np.ndarray:
        r = self.rates(player_uids, positions)
        mask = np.array([[a in GROUP.get(str(x), ACTIONS) for a in ACTIONS] for x in positions])
        scale = np.asarray(minutes) / 90 * self.opponent_factor(opponents)
        out: np.ndarray = (r * mask).sum(axis=1) * scale
        return out


def defensive_rows(info: InformationSet) -> pd.DataFrame:
    pm = info.table("fact_player_match")
    if pm.empty or ACTIONS[0] not in pm:
        return pd.DataFrame()
    rows = pm[(pm["minutes"] > 0) & pm["player_uid"].notna() & pm[ACTIONS[0]].notna()].copy()
    for a in ACTIONS:
        rows[a] = rows[a].astype(float)
    return rows


def drift_factors(rows: pd.DataFrame) -> dict[str, float]:
    """position:action → factor for old-era counts (1 when within tolerance)."""
    old = rows[rows["season"] <= OLD_ERA_END]
    new = rows[rows["season"] >= NEW_ERA_START]
    out: dict[str, float] = {}
    if old.empty or new.empty:
        return out
    for pos in sorted(set(old["position"]) & set(new["position"])):
        o, n = old[old["position"] == pos], new[new["position"] == pos]
        for a in ACTIONS:
            r_old = o[a].sum() / max(o["minutes"].sum(), 1)
            r_new = n[a].sum() / max(n["minutes"].sum(), 1)
            if r_old > 0 and abs(r_new / r_old - 1) > DRIFT_TOLERANCE:
                out[f"{pos}:{a}"] = float(r_new / r_old)
    return out


def fit_defence(info: InformationSet, half_life_days: float = 365.0) -> DefenceModel:
    rows = defensive_rows(info)
    if rows.empty:
        return DefenceModel()
    drift = drift_factors(rows)
    old = (rows["season"] <= OLD_ERA_END).to_numpy()
    for key, factor in drift.items():
        pos, a = key.split(":")
        sel = old & (rows["position"] == pos).to_numpy()
        rows.loc[sel, a] = rows.loc[sel, a] * factor
    w = decay(info, rows["kickoff_at"], half_life_days)
    e = w * rows["minutes"] / 90
    agg = pd.DataFrame({"e": e, **{a: w * rows[a] for a in ACTIONS}, "uid": rows["player_uid"]})
    agg = agg.groupby("uid")[["e", *ACTIONS]].sum()
    groups = player_groups(info).reindex(agg.index)
    groups["position"] = groups["position"].fillna("MID")
    groups["role"] = groups["role"].fillna(groups["position"])
    group = groups["position"] + ":" + groups["role"]
    players = groups.copy()
    for a in ACTIONS:
        players[a] = gamma_poisson(agg[a], agg["e"], group)
    by_pos = agg.groupby(groups["position"])[["e", *ACTIONS]].sum()
    position_rate = {
        str(pos): {a: float(r[a] / max(r["e"], 1e-9)) for a in ACTIONS}
        for pos, r in by_pos.iterrows()
    }

    # opponent multiplier: actions made against each opponent relative to the expected
    expected_rows = (
        players.reindex(rows["player_uid"].to_numpy())[list(ACTIONS)].to_numpy()
        * (rows["minutes"].to_numpy() / 90)[:, None]
    ).sum(axis=1)
    made = rows[list(ACTIONS)].sum(axis=1).to_numpy()
    per_opp = (
        pd.DataFrame(
            {
                "made": w.to_numpy() * made,
                "exp": w.to_numpy() * expected_rows,
                "opp": rows["opponent"],
            }
        )
        .groupby("opp")[["made", "exp"]]
        .sum()
    )
    per_match = float(np.nansum(expected_rows) / max(rows["fixture_uid"].nunique() * 2, 1))
    prior = OPP_PSEUDO_MATCHES * per_match
    opponent = (per_opp["made"] + prior) / (per_opp["exp"] + prior)

    # NegBin size of the position's group total, beyond Poisson
    size: dict[str, float] = {}
    for grp, d in rows.groupby("position"):
        cols = list(GROUP.get(str(grp), ACTIONS))
        r = players.reindex(d["player_uid"].to_numpy())[cols].to_numpy().sum(axis=1)
        mean = (
            r
            * d["minutes"].to_numpy()
            / 90
            * opponent.reindex(d["opponent"]).fillna(1.0).to_numpy()
        )
        y = d[cols].sum(axis=1).to_numpy()
        excess = float(((y - mean) ** 2 - mean).sum())
        size[str(grp)] = (
            float(np.clip((mean**2).sum() / excess, 1.0, 200.0)) if excess > 0 else 200.0
        )
    return DefenceModel(players, position_rate, opponent, size, drift)
