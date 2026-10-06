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

import numpy as np
import pandas as pd

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
