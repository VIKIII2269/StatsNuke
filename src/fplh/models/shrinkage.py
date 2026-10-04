"""Shared shrinkage pieces for the per-player component models (M5–M10).

* ``gamma_poisson``: posterior mean rate per player, shrunk towards their group's mean
  with a prior strength from the method of moments (between-player variance beyond
  Poisson noise);
* ``player_groups``: each player's FPL position (latest) and Understat role (mode over
  starts), the grouping every component model shrinks within;
* ``decay``: weights 2^(−age/h½) of rows observed before the deadline.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from fplh.features.information_set import InformationSet

ROLE = {
    "GK": "GK",
    "DC": "CB",
    "DR": "FB",
    "DL": "FB",
    "DMR": "FB",
    "DML": "FB",
    "DMC": "DM",
    "MC": "CM",
    "MR": "WM",
    "ML": "WM",
    "AMC": "AM",
    "AMR": "W",
    "AML": "W",
    "FW": "FW",
    "FWR": "FW",
    "FWL": "FW",
}


def gamma_poisson(count: pd.Series, exposure: pd.Series, group: pd.Series) -> pd.Series:
    """Posterior mean rate per player: (α_g + count) / (α_g/μ_g + exposure)."""
    df = pd.DataFrame({"c": count, "e": exposure, "g": group})
    out = pd.Series(np.nan, index=df.index)
    for _, d in df.groupby("g"):
        mu = d["c"].sum() / max(d["e"].sum(), 1e-9)
        big = d[d["e"] >= 5]
        if mu <= 0:
            out[d.index] = 0.0
            continue
        if len(big) >= 5:
            rates = big["c"] / big["e"]
            between = rates.var() - (mu / big["e"]).mean()
            alpha = float(np.clip(mu**2 / max(between, 1e-6 * mu**2), 0.5, 200.0))
        else:
            alpha = 5.0 * mu  # five matches of pseudo-exposure
        out[d.index] = (alpha + d["c"]) / (alpha / mu + d["e"])
    return out


def decay(info: InformationSet, at: pd.Series, half_life_days: float) -> pd.Series:
    age = (info.deadline - at).dt.total_seconds() / 86400
    out: pd.Series = np.power(2.0, -age / half_life_days)
    return out


def player_groups(info: InformationSet) -> pd.DataFrame:
    """index player_uid: position (latest FPL), role (Understat mode, else position)."""
    pm = info.table("fact_player_match")
    if pm.empty:
        return pd.DataFrame(columns=["position", "role"])
    position = (
        pm.sort_values("kickoff_at")
        .drop_duplicates("player_uid", keep="last")
        .set_index("player_uid")["position"]
    )
    us = info.table("fact_player_match_understat")
    out = pd.DataFrame({"position": position})
    if not us.empty and "player_uid" in us:
        roles = us[us["player_uid"].notna()].assign(role=lambda d: d["position"].map(ROLE))
        roles = roles[roles["role"].notna()]
        mode = roles.groupby("player_uid")["role"].agg(lambda r: r.mode().iloc[0])
        out["role"] = mode.reindex(out.index)
    else:
        out["role"] = np.nan
    out["role"] = out["role"].fillna(out["position"])
    return out
