"""M5/M6 attack (ARCHITECTURE.md §7.7): per-player goal and assist propensities.

From Understat player-matches and shots observable at the deadline, time-decayed with
half-life ``half_life_days`` (weights 2^(−age/h½)):

* shot rate r_p (non-penalty shots per 90): Gamma–Poisson, shrunk towards the mean of
  the player's group (FPL position × Understat role); the prior strength α_g comes from
  the method of moments (between-player variance beyond Poisson noise);
* shot quality q̄_p (non-penalty xG per shot): pseudo-count shrinkage towards the group;
* finishing f_p (non-penalty goals per xG): heavy shrinkage towards 1 (M6);
* non-penalty goal rate r_p·q̄_p·f_p and assist weight a_p (xA per 90, shrunk the same way);
* penalty takers: decayed penalty attempts; league constants: the share of goals that are
  penalties and own goals, penalty conversion, and the share of goals with an FPL assist.

The simulator allocates each simulated team goal to the players on the pitch in proportion
to these rates (``sim/simulator.py``), so team goal totals stay those of the goal process.
"""

from __future__ import annotations

from dataclasses import dataclass, field

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
QUALITY_PSEUDO_SHOTS = 20.0
FINISHING_PSEUDO_XG = 30.0


def _gamma_poisson(count: pd.Series, exposure: pd.Series, group: pd.Series) -> pd.Series:
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


@dataclass
class AttackRates:
    players: pd.DataFrame  # index player_uid: goal_rate, assist_rate, pen_weight, exposure, …
    league: dict[str, float] = field(default_factory=dict)

    def rates_for(self, player_uids: pd.Series, positions: pd.Series) -> pd.DataFrame:
        """Rates for given players; unseen players get their position's mean."""
        p = self.players.reindex(player_uids.to_numpy())
        by_pos = self.players.groupby("position")[["goal_rate", "assist_rate"]].mean()
        fill = by_pos.reindex(positions.to_numpy())
        out = pd.DataFrame(
            {
                "goal_rate": p["goal_rate"].fillna(
                    pd.Series(fill["goal_rate"].to_numpy(), index=p.index)
                ),
                "assist_rate": p["assist_rate"].fillna(
                    pd.Series(fill["assist_rate"].to_numpy(), index=p.index)
                ),
                "pen_weight": p["pen_weight"].fillna(0.0),
            }
        )
        return out.fillna(0.0).reset_index(drop=True)


def fit_attack(info: InformationSet, half_life_days: float = 365.0) -> AttackRates:
    us = info.table("fact_player_match_understat")
    shots = info.table("fact_shot")
    pm = info.table("fact_player_match")
    if us.empty or "player_uid" not in us:
        return AttackRates(
            pd.DataFrame(columns=["goal_rate", "assist_rate", "pen_weight", "position"])
        )
    us = us[us["player_uid"].notna()].copy()
    if "xa" not in us:
        us["xa"] = 0.0
    age = (info.deadline - us["event_at"]).dt.total_seconds() / 86400
    us["w"] = np.power(2.0, -age / half_life_days)

    s = shots[shots["player_uid"].notna()].copy()
    s["pen"] = s["situation"] == "Penalty"
    s["goal"] = s["result"] == "Goal"
    np_shots = s[~s["pen"] & (s["result"] != "OwnGoal")]
    per_match = np_shots.groupby(["player_uid", "fixture_uid"]).agg(
        np_shots=("shot_id", "size"), npxg=("xg", "sum"), np_goals=("goal", "sum")
    )
    pens = s[s["pen"]].groupby(["player_uid", "fixture_uid"]).size().rename("pens")
    us = us.join(per_match, on=["player_uid", "fixture_uid"]).join(
        pens, on=["player_uid", "fixture_uid"]
    )
    for c in ("np_shots", "npxg", "np_goals", "pens"):
        us[c] = us[c].fillna(0.0)
    us["role"] = us["position"].map(ROLE)

    position = (
        pm.sort_values("kickoff_at")
        .drop_duplicates("player_uid", keep="last")
        .set_index("player_uid")["position"]
    )
    agg = (
        us.assign(
            e=us["w"] * us["minutes"] / 90,
            s=us["w"] * us["np_shots"],
            x=us["w"] * us["npxg"],
            g=us["w"] * us["np_goals"],
            a=us["w"] * us["xa"],
            pw=us["w"] * us["pens"],
        )
        .groupby("player_uid")[["e", "s", "x", "g", "a", "pw"]]
        .sum()
    )
    starters = us[us["role"].notna()]
    role = starters.groupby("player_uid")["role"].agg(lambda r: r.mode().iloc[0])
    agg["position"] = position.reindex(agg.index).fillna("MID")
    agg["role"] = role.reindex(agg.index).fillna(agg["position"])
    group = agg["position"] + ":" + agg["role"]

    shot_rate = _gamma_poisson(agg["s"], agg["e"], group)
    q_group = agg.groupby(group)["x"].transform("sum") / agg.groupby(group)["s"].transform(
        "sum"
    ).clip(lower=1e-9)
    quality = (agg["x"] + QUALITY_PSEUDO_SHOTS * q_group) / (agg["s"] + QUALITY_PSEUDO_SHOTS)
    finishing = (agg["g"] + FINISHING_PSEUDO_XG) / (agg["x"] + FINISHING_PSEUDO_XG)
    players = pd.DataFrame(
        {
            "position": agg["position"],
            "role": agg["role"],
            "exposure": agg["e"],
            "shot_rate": shot_rate,
            "xg_per_shot": quality,
            "finishing": finishing,
            "goal_rate": shot_rate * quality * finishing,
            "assist_rate": _gamma_poisson(agg["a"], agg["e"], group),
            "pen_weight": agg["pw"],
        }
    )

    events = info.table("fact_match_event")
    matches = max(int(events["understat_match_id"].nunique()) if not events.empty else 0, 1)
    goals = events[events["kind"].isin(["goal", "own_goal"])] if not events.empty else events
    n_goals = max(len(goals), 1)
    pen_shots = s[s["pen"]]
    fpl_goals = float(pm["goals_scored"].sum()) + float(pm["own_goals"].sum())
    league = {
        "penalty_share": float(goals["is_penalty"].sum() / n_goals) if len(goals) else 0.1,
        "own_goal_share": float((goals["kind"] == "own_goal").sum() / n_goals)
        if len(goals)
        else 0.03,
        "penalty_conversion": float(pen_shots["goal"].mean()) if len(pen_shots) else 0.78,
        "penalty_misses_per_side": float((~pen_shots["goal"]).sum() / (2 * matches))
        if len(pen_shots)
        else 0.03,
        "assist_share": float(pm["assists"].sum() / fpl_goals) if fpl_goals else 0.7,
    }
    return AttackRates(players, league)
