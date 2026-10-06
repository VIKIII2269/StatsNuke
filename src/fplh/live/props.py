"""Anytime-scorer forward test (A11): our P(score) against the bookmakers' prices.

No historical prop prices exist, so this test only runs forward. For each finished
fixture with captured anytime-scorer prices:

* our forecast is the one ``live advise`` stored at that gameweek's deadline, as P(score |
  plays) = p_goal / p_play (a scorer bet is void if the player does not play);
* the market's probability is the median implied probability over books at the last
  capture before kickoff, times a margin factor c. c is learned walk-forward, as goals
  scored ÷ implied probability over earlier gameweeks (0.85 before any);
* the outcome is whether the player scored, for players who played.

Log loss of ours, the market's and their equal blend is reported per gameweek and for the
season. A fused attack input is adopted only once the blend beats both over enough weeks.
"""

from __future__ import annotations

import re
import unicodedata
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from fplh.features.information_set import SilverStore
    from fplh.lake.storage import Lake

FORECAST_KEY = "state/forecast/{season}/gw{gw}.parquet"
PRIOR_MARGIN = 0.85


def name_key(name: str) -> str:
    s = unicodedata.normalize("NFKD", str(name)).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z]", "", s.lower())


def match_players(props: pd.DataFrame, players: pd.DataFrame) -> pd.DataFrame:
    """``players``: player_uid, team, first_name, second_name, web_name. Matches each prop
    row's player name within its fixture's two teams: full name, then web name, then a
    surname unique within the fixture."""
    out = props.copy()
    out["player_uid"] = None
    for pair, idx in out.groupby(["home_team", "away_team"]).groups.items():
        teams = [str(t) for t in pair] if isinstance(pair, tuple) else [str(pair)]
        cands = players[players["team"].isin(teams)]
        lookup: dict[str, str | None] = {}
        surname: dict[str, list[str]] = {}
        for _, p in cands.iterrows():
            lookup[name_key(f"{p['first_name']} {p['second_name']}")] = p["player_uid"]
            lookup.setdefault(name_key(p["web_name"]), p["player_uid"])
            last = name_key(str(p["second_name"]).split(" ")[-1])
            surname.setdefault(last, []).append(p["player_uid"])
        for i in idx:
            key = name_key(str(out.at[i, "player_name"]))
            uid = lookup.get(key)
            if uid is None:
                last = name_key(str(out.at[i, "player_name"]).split(" ")[-1])
                found = surname.get(last, [])
                uid = found[0] if len(found) == 1 else None
            out.at[i, "player_uid"] = uid
    return out


def _logloss(p: np.ndarray, y: np.ndarray) -> float:
    q = np.clip(p, 1e-4, 1 - 1e-4)
    return float(-(y * np.log(q) + (1 - y) * np.log(1 - q)).mean())


def evaluate(
    props: pd.DataFrame, forecasts: pd.DataFrame, outcomes: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(per player-fixture rows, per-gameweek summary).

    ``props``: matched ``snap_props`` rows; ``forecasts``: gw, player_uid, fixture_uid,
    p_goal, p_play; ``outcomes``: player_uid, fixture_uid, minutes, goals_scored."""
    pre = props[props["observed_at"] < props["kickoff_at"]].dropna(subset=["player_uid"])
    last = pre.sort_values("observed_at").drop_duplicates(
        ["fixture_uid", "player_uid", "bookmaker"], keep="last"
    )
    mkt = (
        last.assign(implied=1 / last["price"])
        .groupby(["fixture_uid", "player_uid"])["implied"]
        .median()
        .rename("implied")
        .reset_index()
    )
    rows = mkt.merge(forecasts, on=["fixture_uid", "player_uid"], how="inner").merge(
        outcomes, on=["fixture_uid", "player_uid"], how="inner"
    )
    rows = rows[rows["minutes"] > 0].copy()
    if rows.empty:
        return rows, pd.DataFrame()
    rows["ours"] = (rows["p_goal"] / rows["p_play"].clip(lower=1e-3)).clip(0, 1)
    rows["scored"] = (rows["goals_scored"] >= 1).astype(float)
    margin, out = PRIOR_MARGIN, []
    for gw in sorted(rows["gw"].unique()):
        g = rows["gw"] == gw
        rows.loc[g, "c"] = margin
        past = rows[rows["gw"] <= gw]
        margin = float(past["scored"].sum() / past["implied"].sum()) if len(past) else margin
    rows["market"] = (rows["implied"] * rows["c"]).clip(0, 1)
    rows["blend"] = 0.5 * rows["ours"] + 0.5 * rows["market"]
    for gw, g in rows.groupby("gw"):
        y = g["scored"].to_numpy()
        out.append(
            {
                "gw": int(str(gw)),
                "n": len(g),
                "ours": _logloss(g["ours"].to_numpy(), y),
                "market": _logloss(g["market"].to_numpy(), y),
                "blend": _logloss(g["blend"].to_numpy(), y),
            }
        )
    return rows, pd.DataFrame(out)


def props_section(summary: pd.DataFrame, rows: pd.DataFrame) -> str:
    if summary.empty:
        return ""
    y = rows["scored"].to_numpy()
    lines = [
        "### Anytime-scorer forward test (A11)",
        "| GW | Players | Ours | Market | Blend |",
        "|---|---|---|---|---|",
    ]
    for _, r in summary.iterrows():
        lines.append(
            f"| {int(r['gw'])} | {int(r['n'])} | {r['ours']:.4f} | {r['market']:.4f} | "
            f"{r['blend']:.4f} |"
        )
    lines.append(
        f"- **Season** ({len(rows)} player-fixtures, log loss, lower is better): ours "
        f"{_logloss(rows['ours'].to_numpy(), y):.4f}, market "
        f"{_logloss(rows['market'].to_numpy(), y):.4f}, blend "
        f"{_logloss(rows['blend'].to_numpy(), y):.4f}"
    )
    return "\n".join(lines)


def season_eval(
    lake: Lake, store: SilverStore, season: str
) -> tuple[pd.DataFrame, pd.DataFrame] | str:
    """The forward test over every stored deadline forecast of ``season``: (rows, summary),
    or the reason it cannot run yet."""
    from fplh.lake.parquet import read_parquet

    props = store.get("snap_props")
    if props.empty or (props["season"] == season).sum() == 0:
        return "no anytime-scorer captures yet"
    props = props[props["season"] == season]
    snap = store.get("snap_fpl_player")
    snap = snap[snap["season"] == season].sort_values("observed_at")
    snap = snap.drop_duplicates("code", keep="last")
    names = store.get("dim_player")
    players = snap.assign(player_uid="fpl:" + snap["code"].astype(str))[["player_uid", "team"]]
    players = players.merge(names, on="player_uid", how="left")
    matched = match_players(props, players)
    keys = [k for k in lake.list(f"state/forecast/{season}/") if re.search(r"/gw\d+\.parquet$", k)]
    if not keys:
        return "no stored forecasts yet"
    forecasts = pd.concat([read_parquet(lake, k) for k in keys], ignore_index=True)
    if "p_play" not in forecasts or "p_goal" not in forecasts:
        return "stored forecasts lack p_goal / p_play"
    forecasts = forecasts[forecasts["horizon"] == 1] if "horizon" in forecasts else forecasts
    pm = store.get("fact_player_match")
    outcomes = pm[pm["season"] == season][["player_uid", "fixture_uid", "minutes", "goals_scored"]]
    return evaluate(matched, forecasts, outcomes)
