"""M8 goalkeeper saves (ARCHITECTURE.md §7.8).

Saves of a goalkeeper on the pitch for a fraction m = minutes/90 of a match, given the
opponent's pre-match expected goals μ (M1, point in time):

    saves ~ NegBin(mean = m · g_k · (a + b·μ), size k)

* a, b: least squares over all goalkeeper matches observable at the deadline. Given μ,
  the match's realised goals add nothing (coefficient 0.04 on 2016–2026), so saves are
  drawn independently of the simulated goals;
* g_k: the goalkeeper's multiplier (his defence's shot profile and his own shot-stopping),
  Gamma–Poisson shrunk towards 1 on decayed saves against decayed expected saves;
* k: method of moments on the residual variance beyond Poisson.

Penalty saves: a missed opposition penalty is saved by the goalkeeper on the pitch with
the league share ``penalty_save_share`` (FPL counts saved penalties as missed too).

``prematch`` is a frame of fixture_uid, pred_home, pred_away: M1's pre-update rates for
every match (``evaluate.phase2.training_predictions``), so each μ predates its match.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.stats import nbinom

from fplh.features.information_set import InformationSet
from fplh.models.shrinkage import decay, gamma_poisson


@dataclass
class SaveModel:
    a: float = 1.18
    b: float = 1.23
    size: float = 20.0  # NegBin k
    keepers: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))  # g_k
    penalty_save_share: float = 0.74

    def multiplier(self, player_uids: pd.Series) -> np.ndarray:
        out: np.ndarray = self.keepers.reindex(player_uids.to_numpy()).fillna(1.0).to_numpy()
        return out

    def mean(
        self, mu_opp: np.ndarray | float, minutes: np.ndarray, mult: np.ndarray | float
    ) -> np.ndarray:
        out: np.ndarray = (np.asarray(minutes) / 90) * mult * (self.a + self.b * np.asarray(mu_opp))
        return out


def keeper_rows(info: InformationSet, prematch: pd.DataFrame) -> pd.DataFrame:
    """Goalkeeper matches with minutes > 0 and the opponent's pre-match μ."""
    pm = info.table("fact_player_match")
    gk = pm[(pm["position"] == "GK") & (pm["minutes"] > 0) & pm["player_uid"].notna()]
    rates = prematch[["fixture_uid", "pred_home", "pred_away"]]
    gk = gk.merge(rates, on="fixture_uid")
    out: pd.DataFrame = gk.assign(
        mu_opp=np.where(gk["was_home"], gk["pred_away"], gk["pred_home"]),
        m=gk["minutes"] / 90,
    )
    return out


def fit_saves(
    info: InformationSet, prematch: pd.DataFrame, half_life_days: float = 365.0
) -> SaveModel:
    pm = info.table("fact_player_match")
    missed = float(pm["penalties_missed"].sum()) if not pm.empty else 0.0
    share = float(pm["penalties_saved"].sum()) / missed if missed else 0.74
    gk = keeper_rows(info, prematch)
    if len(gk) < 50:
        return SaveModel(penalty_save_share=share)
    m, mu, y = gk["m"].to_numpy(), gk["mu_opp"].to_numpy(), gk["saves"].to_numpy(float)
    (a, b), *_ = np.linalg.lstsq(np.c_[m, m * mu], y, rcond=None)
    expected = m * (a + b * mu)
    w = decay(info, gk["kickoff_at"], half_life_days).to_numpy()
    agg = pd.DataFrame({"y": w * y, "e": w * expected, "uid": gk["player_uid"].to_numpy()})
    agg = agg.groupby("uid")[["y", "e"]].sum()
    keepers = gamma_poisson(agg["y"], agg["e"], pd.Series("GK", index=agg.index))
    mean = expected * keepers.reindex(gk["player_uid"].to_numpy()).to_numpy()
    excess = float(((y - mean) ** 2 - mean).sum())
    size = float(np.clip((mean**2).sum() / excess, 2.0, 200.0)) if excess > 0 else 200.0
    return SaveModel(float(a), float(b), size, keepers, share)


def negbin_pmf(mean: np.ndarray, size: float, upto: int) -> np.ndarray:
    """(n, upto+1) NegBin(mean, size) pmf; the last column holds the upper tail."""
    mean = np.clip(np.asarray(mean, dtype=float), 1e-9, None)
    p = size / (size + mean)
    ks = np.arange(upto + 1)
    pmf = nbinom.pmf(ks[None, :], size, p[:, None])
    pmf[:, -1] = nbinom.sf(upto - 1, size, p)
    out: np.ndarray = pmf
    return out
