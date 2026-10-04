"""M9 cards (ARCHITECTURE.md §7.9): per-player yellow-card propensity.

From FPL player-matches observable at the deadline, time-decayed with half-life
``half_life_days``: a Gamma–Poisson yellow rate per 90, shrunk towards the mean of the
player's position × role group. The simulator draws P(yellow) = 1 − exp(−rate·minutes/90).

FPL records at most one yellow per match and never a yellow together with a red (a second
booking is scored as the red only), so a simulated red removes the yellow. Red-card
*timing* comes from the team process; the sent-off player is drawn ∝ yellow rate.

Not modelled (documented): game state (yellow minutes are not identifiable from our
sources) and the referee (observed only after the match).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from fplh.features.information_set import InformationSet
from fplh.models.shrinkage import decay, gamma_poisson, player_groups


@dataclass
class CardRates:
    players: pd.DataFrame  # index player_uid: position, role, exposure, yellow_rate
    position_rate: dict[str, float] = field(default_factory=dict)

    def rates_for(self, player_uids: pd.Series, positions: pd.Series) -> np.ndarray:
        """Yellow rate per 90 for given players; unseen players get their position's."""
        p = self.players["yellow_rate"].reindex(player_uids.to_numpy()).to_numpy()
        fill = np.array([self.position_rate.get(str(x), 0.0) for x in positions])
        out: np.ndarray = np.where(np.isnan(p), fill, p)
        return out


def fit_cards(info: InformationSet, half_life_days: float = 365.0) -> CardRates:
    pm = info.table("fact_player_match")
    pm = pm[(pm["minutes"] > 0) & pm["player_uid"].notna()] if not pm.empty else pm
    if pm.empty:
        return CardRates(pd.DataFrame(columns=["position", "role", "exposure", "yellow_rate"]))
    w = decay(info, pm["kickoff_at"], half_life_days)
    agg = (
        pm.assign(e=w * pm["minutes"] / 90, y=w * pm["yellow_cards"])
        .groupby("player_uid")[["e", "y"]]
        .sum()
    )
    groups = player_groups(info).reindex(agg.index)
    groups["position"] = groups["position"].fillna("MID")
    groups["role"] = groups["role"].fillna(groups["position"])
    group = groups["position"] + ":" + groups["role"]
    players = groups.assign(exposure=agg["e"], yellow_rate=gamma_poisson(agg["y"], agg["e"], group))
    by_pos = agg.groupby(groups["position"])[["y", "e"]].sum()
    position_rate = (by_pos["y"] / by_pos["e"].clip(lower=1e-9)).to_dict()
    return CardRates(players, {str(k): float(v) for k, v in position_rate.items()})
