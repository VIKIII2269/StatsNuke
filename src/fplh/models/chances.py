"""Model v2.1: assists from chance creation (experiment P5.9).

v2's assists come from each player's share of his team's goals. A model of chance
creation predicts FPL assists better. It is trained on Understat xA per appearance, with
each player's and team's state after their last match strictly before the forecast time:
- rolling xG, xA, shots, key passes and xG chain per 90 over 3, 6, 15 and 38 appearances;
- minutes and start shares;
- the team's xG for and the opponent's xG against over 6 and 19 matches;
- home or away.

The forecast for a player-fixture is

    assists = rate · P(play) · scale,

where the rate is the predicted xA per appearance, P(play) is v2's, and the scale
converts Understat xA to FPL assists (fitted on the training appearances). It is blended
into v2's expected points:

    E' = E + 3 · ASSIST_WEIGHT · (assists − v2 assists).

The weight 0.75 was chosen on 2021/22. On 2022/23–2024/25, over all player-fixtures,
points MSE fell from 3.536 to 3.522: −0.014 [−0.018, −0.011] (gameweek-block CI),
in each test season. Goals stay with v2, which predicts them better.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import xgboost as xgb

from fplh.features.information_set import SilverStore

ASSIST_WEIGHT = 0.75
ASSIST_POINTS = 3
NAME = "v2.1 (assists from chance creation)"
STATS = ("xg", "xa", "shots", "key_passes", "xg_chain")
WINDOWS = (3, 6, 15, 38)
TEAM_WINDOWS = (6, 19)
PARAMS: dict[str, Any] = {
    "max_depth": 5,
    "eta": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "objective": "reg:tweedie",
    "tweedie_variance_power": 1.3,
    "nthread": 4,
    "verbosity": 0,
    "seed": 0,
}
ROUNDS = 400
FEATURES = (
    [f"{c}90_{w}" for w in WINDOWS for c in STATS]
    + [f"mins_{w}" for w in WINDOWS]
    + [f"start_{w}" for w in WINDOWS]
    + [f"team_{k}_{w}" for w in TEAM_WINDOWS for k in ("xg", "xga")]
    + [f"opp_{k}_{w}" for w in TEAM_WINDOWS for k in ("xg", "xga")]
    + ["home"]
)


def appearances(store: SilverStore) -> pd.DataFrame:
    """Understat appearances with kickoff time and the player's team (dim_fixture uids)."""
    u = store.get("fact_player_match_understat").dropna(subset=["player_uid", "fixture_uid"])
    u = u.drop_duplicates(["player_uid", "fixture_uid"]).copy()
    for c in (*STATS, "minutes"):
        u[c] = pd.to_numeric(u[c], errors="coerce").fillna(0.0)
    u["started"] = u["started"].astype(float)
    dim = store.get("dim_fixture")[["fixture_uid", "home_team", "away_team", "kickoff_at"]]
    u = u.merge(dim, on="fixture_uid")
    u["team"] = np.where(u["side"] == "h", u["home_team"], u["away_team"])
    out: pd.DataFrame = u.sort_values("kickoff_at").reset_index(drop=True)
    return out


def _roll(g: Any, col: str, w: int, how: str, index: pd.Index) -> pd.Series:
    r = getattr(g[col].rolling(w, min_periods=1), how)()
    out: pd.Series = r.reset_index(level=0, drop=True).reindex(index)
    return out


def player_states(u: pd.DataFrame) -> pd.DataFrame:
    """Each player's rolling state after each appearance (that match included)."""
    g = u.groupby("player_uid")
    out = u[["player_uid", "kickoff_at"]].copy()
    for w in WINDOWS:
        mins = _roll(g, "minutes", w, "sum", u.index)
        per90 = mins.clip(lower=1) / 90
        for c in STATS:
            out[f"{c}90_{w}"] = _roll(g, c, w, "sum", u.index) / per90
        out[f"mins_{w}"] = mins / w
        out[f"start_{w}"] = _roll(g, "started", w, "sum", u.index) / w
    return out


def team_states(u: pd.DataFrame) -> pd.DataFrame:
    """Each team's rolling xG for / against after each match (that match included)."""
    tm = (
        u.groupby(["fixture_uid", "side"])
        .agg(t_xg=("xg", "sum"), team=("team", "first"), kickoff_at=("kickoff_at", "first"))
        .reset_index()
    )
    other = tm[["fixture_uid", "side", "t_xg"]].assign(side=tm["side"].map({"h": "a", "a": "h"}))
    tm = tm.merge(other.rename(columns={"t_xg": "t_xga"}), on=["fixture_uid", "side"])
    tm = tm.sort_values("kickoff_at")
    g = tm.groupby("team")
    out = tm[["team", "kickoff_at"]].copy()
    for w in TEAM_WINDOWS:
        out[f"xg_{w}"] = _roll(g, "t_xg", w, "mean", tm.index)
        out[f"xga_{w}"] = _roll(g, "t_xga", w, "mean", tm.index)
    return out


def _ns(t: pd.Series) -> pd.Series:
    """UTC timestamps at one resolution (sources mix microseconds and nanoseconds)."""
    out: pd.Series = pd.to_datetime(t, utc=True).astype("datetime64[ns, UTC]")
    return out


def attach(rows: pd.DataFrame, ps: pd.DataFrame, ts: pd.DataFrame) -> pd.DataFrame:
    """``rows``: player_uid, team, opp, home, at (the forecast time). Adds the features
    as of strictly before ``at``."""
    r = rows.reset_index(drop=False).rename(columns={"index": "_row"})
    r["at"] = _ns(r["at"])
    ps = ps.assign(kickoff_at=_ns(ps["kickoff_at"]))
    ts = ts.assign(kickoff_at=_ns(ts["kickoff_at"]))
    r = r.sort_values("at")
    r = pd.merge_asof(
        r,
        ps.rename(columns={"kickoff_at": "at"}).sort_values("at"),
        on="at",
        by="player_uid",
        allow_exact_matches=False,
    )
    team = ts.rename(
        columns={
            "kickoff_at": "at",
            **{c: f"team_{c}" for c in map(str, ts.columns) if c.startswith("xg")},
        }
    )
    r = pd.merge_asof(r, team.sort_values("at"), on="at", by="team", allow_exact_matches=False)
    opp = ts.rename(
        columns={
            "kickoff_at": "at",
            "team": "opp",
            **{c: f"opp_{c}" for c in map(str, ts.columns) if c.startswith("xg")},
        }
    )
    r = pd.merge_asof(r, opp.sort_values("at"), on="at", by="opp", allow_exact_matches=False)
    out: pd.DataFrame = r.sort_values("_row").set_index("_row").rename_axis(None)
    return out


@dataclass
class AssistModel:
    booster: xgb.Booster
    scale: float  # FPL assists per Understat xA

    @classmethod
    def fit(cls, store: SilverStore, before: pd.Timestamp) -> AssistModel:
        """Train on every Understat appearance that kicked off before ``before``."""
        u = appearances(store)
        u = u[u["kickoff_at"] < before]
        ps, ts = player_states(u), team_states(u)
        rows = u.assign(
            opp=np.where(u["side"] == "h", u["away_team"], u["home_team"]),
            home=(u["side"] == "h").astype(float),
            at=u["kickoff_at"],
        )
        x = attach(rows[["player_uid", "team", "opp", "home", "at", "xa"]], ps, ts)
        booster = xgb.train(PARAMS, xgb.DMatrix(x[FEATURES], label=x["xa"]), ROUNDS)
        pm = store.get("fact_player_match")[["player_uid", "fixture_uid", "assists"]]
        pm = pm.drop_duplicates(["player_uid", "fixture_uid"]).rename(
            columns={"assists": "fpl_assists"}
        )
        both = u[["player_uid", "fixture_uid", "xa"]].merge(pm, on=["player_uid", "fixture_uid"])
        scale = float(both["fpl_assists"].sum() / max(both["xa"].sum(), 1e-9)) if len(both) else 1.0
        return cls(booster, scale)

    def rate(self, x: pd.DataFrame) -> np.ndarray:
        """Expected Understat xA per appearance."""
        out: np.ndarray = self.booster.predict(xgb.DMatrix(x[FEATURES]))
        return out


def player_teams(store: SilverStore, pred: pd.DataFrame, at: pd.Timestamp) -> pd.Series:
    """Each forecast row's team: FPL's record for that fixture if it exists, else the
    latest FPL snapshot at or before ``at``, else the player's latest recorded team."""
    pm = store.get("fact_player_match")[["player_uid", "fixture_uid", "team", "kickoff_at"]]
    by_fixture = pm.drop_duplicates(["player_uid", "fixture_uid"]).set_index(
        ["player_uid", "fixture_uid"]
    )["team"]
    team = pd.Series(
        [
            by_fixture.get((p, f))
            for p, f in zip(pred["player_uid"], pred["fixture_uid"], strict=True)
        ],
        index=pred.index,
        dtype=object,
    )
    snap = store.get("snap_fpl_player")
    if len(snap):
        snap = snap[snap["observed_at"] <= at]
        snap = snap.sort_values("observed_at").drop_duplicates("code", keep="last")
        latest = dict(
            zip("fpl:" + snap["code"].astype("int64").astype(str), snap["team"], strict=True)
        )
        team = team.fillna(pred["player_uid"].map(latest))
    last = (
        pm.sort_values("kickoff_at")
        .drop_duplicates("player_uid", keep="last")
        .set_index("player_uid")["team"]
    )
    out: pd.Series = team.fillna(pred["player_uid"].map(last))
    return out


def adjust(
    pred: pd.DataFrame,
    store: SilverStore,
    at: pd.Timestamp | pd.Series,
    *,
    model: AssistModel | None = None,
    weight: float = ASSIST_WEIGHT,
) -> pd.DataFrame:
    """Blend the chance model's assists into v2's forecast. ``at`` is the forecast time:
    one deadline (live) or a per-row series (backtests); features are as of strictly
    before it. ``pred``: player_uid, fixture_uid, expected_points, e_assists, p_play.
    Adds e_assists_v2 and assists_chance, and updates e_assists and expected_points."""
    if pred.empty:
        return pred
    at_s = at if isinstance(at, pd.Series) else pd.Series(at, index=pred.index)
    at_s = pd.to_datetime(at_s, utc=True)
    model = model or AssistModel.fit(store, at_s.min())
    dim = store.get("dim_fixture").set_index("fixture_uid")
    out = pred.copy()
    team = player_teams(store, out, at_s.max())
    home_team = out["fixture_uid"].map(dim["home_team"])
    away_team = out["fixture_uid"].map(dim["away_team"])
    rows = pd.DataFrame(
        {
            "player_uid": out["player_uid"].to_numpy(),
            "team": team.to_numpy(),
            "opp": np.where(team == home_team, away_team, home_team),
            "home": (team == home_team).astype(float).to_numpy(),
            "at": at_s.to_numpy(),
        },
        index=out.index,
    )
    u = appearances(store)  # states use only matches strictly before each row's time
    x = attach(rows, player_states(u), team_states(u))
    rate = pd.Series(model.rate(x), index=out.index)
    assists = rate * out["p_play"].clip(0, 1) * model.scale
    known = rows["team"].notna() & home_team.notna() & away_team.notna()
    assists = assists.where(known, out["e_assists"])  # no team or fixture: keep v2
    out["e_assists_v2"] = out["e_assists"]
    out["assists_chance"] = assists
    out["e_assists"] = (1 - weight) * out["e_assists"] + weight * assists
    out["expected_points"] = out["expected_points"] + ASSIST_POINTS * (
        out["e_assists"] - out["e_assists_v2"]
    )
    return out
