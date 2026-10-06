"""The website's snapshot: everything the Stats Nuke site shows, as one JSON document.

Built at the end of every Live run (``fplh web export``) and uploaded by the workflow to
the site's database, where only approved members can read it. Sections:

* ``gameweeks``, ``fixtures``: the season's schedule and FPL's average and highest scores;
* ``players``: every current player with model v2's expected points for the next five
  gameweeks, P(start), P(play), P(score), price, ownership and FPL's availability flags;
* ``plan``: the model team's decision for the next deadline (or the provisional plan
  before the decision is due);
* ``team``: the model team's season, week by week, with each pick's points once final;
* ``bets``, ``props``: the paper betting and anytime-scorer forward tests;
* ``benchmarks``: manager panels (p50, p90, p99), the live OpenFPL replica comparison and
  the backtests;
* ``health``, ``data``: GitHub Actions runs, lake feeds and the pipeline's checks;
* ``lab``: the experiment log.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from fplh.features.information_set import SilverStore
from fplh.lake.storage import Lake
from fplh.live.team import (
    NEXT_FORECAST_KEY,
    NEXT_KEY,
    REPLICA_KEY,
    LiveTeam,
    events,
    load,
    next_deadline,
)
from fplh.web import benchmarks as bench
from fplh.web.checks import bronze_feeds, pipeline_checks, silver_tables
from fplh.web.lab import load_log

SNAPSHOT_VERSION = 1
FORECAST_KEY = "state/forecast/{season}/gw{gw}.parquet"
SHORT = {
    "arsenal": "ARS",
    "aston-villa": "AVL",
    "bournemouth": "BOU",
    "brentford": "BRE",
    "brighton": "BHA",
    "burnley": "BUR",
    "chelsea": "CHE",
    "coventry": "COV",
    "crystal-palace": "CRY",
    "everton": "EVE",
    "fulham": "FUL",
    "hull": "HUL",
    "ipswich": "IPS",
    "leeds": "LEE",
    "leicester": "LEI",
    "liverpool": "LIV",
    "luton": "LUT",
    "man-city": "MCI",
    "man-united": "MUN",
    "newcastle": "NEW",
    "nottm-forest": "NFO",
    "sheffield-united": "SHU",
    "southampton": "SOU",
    "sunderland": "SUN",
    "tottenham": "TOT",
    "west-ham": "WHU",
    "wolves": "WOL",
}

# Frozen backtests (docs/EXPERIMENTS_V2.md): 2022/23–2024/25, chosen on 2021/22, 2025/26
# untouched.
BACKTEST: dict[str, Any] = {
    "seasons": "2022/23–2024/25",
    "forecast": [
        {"model": "Model v2", "mse": 3.536, "mae": 0.942, "spearman": 0.722, "top10": 0.447},
        {"model": "v1 simulator", "mse": 3.633, "mae": 0.970, "spearman": 0.700, "top10": 0.441},
        {"model": "OpenFPL replica", "mse": 3.661, "mae": 0.996, "spearman": 0.696, "top10": 0.429},
        {"model": "A0", "mse": 4.015, "mae": 1.001, "spearman": 0.714, "top10": 0.410},
        {"model": "Last 5", "mse": 4.354, "mae": 1.050, "spearman": 0.681, "top10": 0.384},
    ],
    "replay": [
        {"strategy": "Model v2", "seasons": [2391, 2339, 2377], "total": 7107, "hits": 39},
        {"strategy": "v1 simulator", "seasons": [2272, 2384, 2300], "total": 6956, "hits": 39},
        {"strategy": "OpenFPL replica", "seasons": [2185, 2250, 2264], "total": 6699, "hits": 117},
        {"strategy": "A0", "seasons": [2050, 1958, 2212], "total": 6220, "hits": 100},
        {"strategy": "Last 5", "seasons": [1977, 2005, 2018], "total": 6000, "hits": 158},
    ],
    "replay_vs_openfpl": {"per_gw": 3.71, "low": 0.25, "high": 7.19, "p": 0.034},
    "mse_vs_openfpl": {"delta": -0.126, "low": -0.146, "high": -0.107},
    "betting": {"bets": 466, "clv": 0.029, "low": 0.016, "high": 0.042, "span": "2016/17–2024/25"},
}


def short(team: object) -> str:
    t = str(team)
    return SHORT.get(t, t[:3].upper())


def _num(x: Any, nd: int = 2) -> float | None:
    if x is None:
        return None
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(v) or math.isinf(v) else round(v, nd)


def _iso(x: Any) -> str | None:
    if x is None or (not isinstance(x, str) and pd.isna(x)):
        return None
    return pd.Timestamp(x).isoformat()


def clean(obj: Any) -> Any:
    """JSON-safe: numpy scalars to Python, NaN and infinities to null, timestamps to ISO."""
    if isinstance(obj, Mapping):
        return {str(k): clean(v) for k, v in obj.items()}
    if isinstance(obj, list | tuple):
        return [clean(v) for v in obj]
    if isinstance(obj, np.generic):
        obj = obj.item()
    if isinstance(obj, float):
        return None if math.isnan(obj) or math.isinf(obj) else obj
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    return obj


def gameweeks(ev: pd.DataFrame) -> list[dict[str, Any]]:
    return [
        {
            "gw": int(str(gw)),
            "deadline": _iso(r["deadline_at"]),
            "finished": bool(r["finished"]),
            "final": bool(r["data_checked"]),
            "average": _num(r["average_entry_score"], 0) if bool(r["finished"]) else None,
            "highest": _num(r["highest_score"], 0) if bool(r["finished"]) else None,
        }
        for gw, r in ev.iterrows()
    ]


def fixtures(dim: pd.DataFrame, season: str) -> list[dict[str, Any]]:
    fx = dim[dim["season"] == season].sort_values("kickoff_at")
    fx = fx.reindex(
        columns=[*fx.columns.difference(["home_goals", "away_goals"]), "home_goals", "away_goals"]
    )
    return [
        {
            "id": r["fixture_uid"],
            "gw": None if pd.isna(r["round"]) else int(r["round"]),
            "home": short(r["home_team"]),
            "away": short(r["away_team"]),
            "kickoff": _iso(r["kickoff_at"]),
            "hg": None if pd.isna(r["home_goals"]) else int(r["home_goals"]),
            "ag": None if pd.isna(r["away_goals"]) else int(r["away_goals"]),
        }
        for _, r in fx.iterrows()
    ]


def latest_snapshot(store: SilverStore, season: str, now: pd.Timestamp) -> pd.DataFrame:
    snap = store.get("snap_fpl_player")
    snap = snap[(snap["season"] == season) & (snap["observed_at"] <= now)]
    if snap.empty:
        return snap
    last = snap[snap["observed_at"] == snap["observed_at"].max()]
    out: pd.DataFrame = last.assign(player_uid="fpl:" + last["code"].astype("int64").astype(str))
    return out


def players(
    snap: pd.DataFrame,
    names: pd.DataFrame,
    pred: pd.DataFrame | None,
    rounds: pd.Series,
    first_gw: int | None,
) -> tuple[list[int], list[dict[str, Any]]]:
    """(the forecast's gameweeks, one row per current player)."""
    gws: list[int] = []
    xp = pd.DataFrame()
    h1 = pd.DataFrame()
    if pred is not None and not pred.empty:
        p = pred.assign(gw=pred["fixture_uid"].map(rounds)).dropna(subset=["gw"])
        p["gw"] = p["gw"].astype(int)
        if first_gw is not None:
            p = p[p["gw"] >= first_gw]
        gws = sorted(p["gw"].unique().tolist())[:5]
        p = p[p["gw"].isin(gws)]
        xp = p.groupby(["player_uid", "gw"])["expected_points"].sum().unstack()
        xp = xp.reindex(columns=gws).fillna(0.0)
        g1 = p[p["gw"] == gws[0]] if gws else p.iloc[0:0]
        agg: dict[str, Any] = {}
        if "p_start" in g1:
            agg["p_start"] = ("p_start", "max")
        if "p_play" in g1:
            agg["p_play"] = ("p_play", "max")
        if "p_goal" in g1:
            agg["p_goal"] = ("p_goal", lambda s: 1 - float(np.prod(1 - s.to_numpy(float))))
        h1 = g1.groupby("player_uid").agg(**agg) if agg else pd.DataFrame()
    cols = ["player_uid", "first_name", "second_name", "web_name"]
    nm = names.reindex(columns=cols).set_index("player_uid")
    rows = []
    for _, r in snap.iterrows():
        uid = r["player_uid"]
        xs = [float(str(xp.at[uid, g])) for g in gws] if uid in xp.index else [0.0] * len(gws)
        row = {
            "id": uid,
            "name": str(nm.at[uid, "web_name"]) if uid in nm.index else uid,
            "full": (
                f"{nm.at[uid, 'first_name']} {nm.at[uid, 'second_name']}" if uid in nm.index else ""
            ),
            "team": short(r["team"]),
            "pos": str(r["position"]),
            "price": _num(float(r["now_cost"]) / 10, 1),
            "own": _num(r.get("selected_by_percent"), 1),
            "status": str(r.get("status") or "a"),
            "chance": _num(r.get("chance_of_playing_next_round"), 0),
            "news": str(r.get("news") or ""),
            "xp": [round(x, 2) for x in xs],
            "xp5": round(sum(xs), 2),
        }
        for col in ("p_start", "p_play", "p_goal"):
            row[col] = _num(h1.at[uid, col], 3) if col in h1 and uid in h1.index else None
        rows.append(row)
    rows.sort(key=lambda x: -(x["xp"][0] if x["xp"] else 0))
    return gws, rows


def _read(lake: Lake, key: str) -> pd.DataFrame | None:
    from fplh.lake.parquet import read_parquet

    return read_parquet(lake, key) if lake.exists(key) else None


def team_section(team: LiveTeam | None, pm: pd.DataFrame) -> list[dict[str, Any]]:
    if team is None:
        return []
    out = []
    for key, w in sorted(team.weeks.items(), key=lambda kv: int(kv[0])):
        gw = int(key)
        week = {"gw": gw, **{k: v for k, v in w.items() if k != "squad"}}
        if "points" in w:
            rows = pm[pm["round"] == gw]
            pts = rows.groupby("player_uid")["total_points"].sum()
            mins = rows.groupby("player_uid")["minutes"].sum()
            week["picks"] = {
                p: {"points": int(pts.get(p, 0)), "minutes": int(mins.get(p, 0))}
                for p in [*w["xi"], *w["bench"]]
            }
        out.append(week)
    return out


def bets_section(lake: Lake) -> dict[str, Any]:
    from fplh.delivery.live_consensus import BETS_KEY, summary

    book = _read(lake, BETS_KEY)
    if book is None or book.empty:
        return {"summary": {"bets": 0}, "bets": []}
    rows = []
    for _, b in book.sort_values("observed_at", ascending=False).iterrows():
        rows.append(
            {
                "home": short(b["home_team"]),
                "away": short(b["away_team"]),
                "kickoff": _iso(b["kickoff_at"]),
                "market": b["market"],
                "outcome": b["outcome"],
                "book": b["bookmaker"],
                "at": _iso(b["observed_at"]),
                "price": _num(b["price"]),
                "fair": _num(b["fair"], 4),
                "ev": _num(b["ev"], 4),
                "fair_close": _num(b.get("fair_close"), 4),
                "clv": _num(b.get("clv"), 4),
                "settled": bool(b.get("settled", False)),
                "won": None if pd.isna(b.get("won")) else bool(b.get("won")),
                "profit": _num(b.get("profit"), 3),
            }
        )
    return {"summary": summary(book), "bets": rows}


def props_section(lake: Lake, store: SilverStore, season: str) -> dict[str, Any]:
    from fplh.live.props import _logloss, season_eval

    try:
        result = season_eval(lake, store, season)
    except Exception as e:  # the site must still build
        return {"note": f"not available: {e}", "weeks": []}
    if isinstance(result, str):
        return {"note": result, "weeks": []}
    rows, summary = result
    if summary.empty:
        return {"note": "no finished fixture with both a forecast and prices yet", "weeks": []}
    y = rows["scored"].to_numpy()
    return {
        "weeks": summary.to_dict("records"),
        "season": {
            "n": len(rows),
            **{c: _logloss(rows[c].to_numpy(), y) for c in ("ours", "market", "blend")},
        },
    }


def health_section(repo: str | None, token: str | None) -> dict[str, Any]:
    """Recent GitHub Actions runs (the site also refreshes these live in the browser)."""
    if not repo:
        return {"runs": [], "note": "no repository configured"}
    import httpx

    headers = {"Accept": "application/vnd.github+json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    try:
        r = httpx.get(
            f"https://api.github.com/repos/{repo}/actions/runs",
            params={"per_page": 60},
            headers=headers,
            timeout=30,
        )
        r.raise_for_status()
    except Exception as e:
        return {"runs": [], "note": f"GitHub API: {e}"}
    runs = []
    for run in r.json().get("workflow_runs", []):
        runs.append(
            {
                "workflow": run.get("name"),
                "event": run.get("event"),
                "status": run.get("status"),
                "conclusion": run.get("conclusion"),
                "started": run.get("run_started_at"),
                "updated": run.get("updated_at"),
                "branch": run.get("head_branch"),
                "sha": str(run.get("head_sha", ""))[:7],
                "url": run.get("html_url"),
            }
        )
    return {"runs": runs, "repo": repo}


def benchmarks_section(
    lake: Lake,
    store: SilverStore,
    season: str,
    now: pd.Timestamp,
    ev: pd.DataFrame,
    snap: pd.DataFrame,
    team: LiveTeam | None,
    pm: pd.DataFrame,
    fpl_get: bench.Getter | None,
) -> dict[str, Any]:
    note = None
    state = bench.load_state(lake, season)
    if fpl_get is not None and not snap.empty:
        total = int(pd.to_numeric(snap["total_players"]).max())
        final = [int(g) for g in ev.index[ev["data_checked"].astype(bool)]]
        try:
            state = bench.update_panels(lake, season, total, final, fpl_get, now)
        except Exception as e:  # FPL down or rate-limited: keep the stored panels
            note = f"manager panels not refreshed: {e}"
    ours = [
        {"gw": int(k), "points": w["points"], "hits": w.get("hits", 0)}
        for k, w in sorted((team.weeks if team else {}).items(), key=lambda kv: int(kv[0]))
        if "points" in w
    ]
    live: list[dict[str, Any]] = []
    outcomes = pm.reindex(columns=["player_uid", "fixture_uid", "total_points"])
    for gw in [int(g) for g in ev.index[ev["data_checked"].astype(bool)]]:
        v2 = _read(lake, FORECAST_KEY.format(season=season, gw=gw))
        rep = _read(lake, REPLICA_KEY.format(season=season, gw=gw))
        if v2 is None or rep is None:
            continue
        if "horizon" in v2:
            v2 = v2[v2["horizon"] == 1]
        live += bench.openfpl_live(v2.assign(gw=gw), rep.assign(gw=gw), outcomes)
    return {
        "model": ours,
        "panels": bench.panel_medians(state),
        "panel_meta": {
            k: {"rank": v["rank"], "share": v["share"], "chosen_at": v["chosen_at"]}
            for k, v in state.get("panels", {}).items()
        },
        "openfpl_live": live,
        "backtest": BACKTEST,
        "note": note,
    }


def build(
    lake: Lake,
    store: SilverStore,
    season: str,
    now: pd.Timestamp,
    *,
    log_path: Path,
    repo: str | None = None,
    token: str | None = None,
    fpl_get: bench.Getter | None = None,
) -> dict[str, Any]:
    ev = events(store, season)
    dim = store.get("dim_fixture")
    rounds = dim.set_index("fixture_uid")["round"]
    names = store.get("dim_player")
    pm = store.get("fact_player_match")
    pm = pm[pm["season"] == season]
    snap = latest_snapshot(store, season, now)
    team = load(lake, season)
    weeks = team.weeks if team else {}

    try:
        next_gw, next_dl = next_deadline(store, season, now)
    except LookupError:
        next_gw, next_dl = None, None
    plan: dict[str, Any] | None = None
    pred: pd.DataFrame | None = None
    made: str | None = None
    if next_gw is not None and weeks.get(str(next_gw), {}).get("advised"):
        plan = {"gameweek": next_gw, "provisional": False, **weeks[str(next_gw)]}
        pred = _read(lake, FORECAST_KEY.format(season=season, gw=next_gw))
        made = str(plan["advised"])
    elif lake.exists(NEXT_KEY.format(season=season)):
        prov = json.loads(lake.get_bytes(NEXT_KEY.format(season=season)))
        if int(prov.get("gameweek", -1)) == next_gw:
            plan = prov
            pred = _read(lake, NEXT_FORECAST_KEY.format(season=season))
            made = str(prov["advised"])
    gws, rows = players(snap, names, pred, rounds, next_gw)

    feeds = bronze_feeds(lake, now)
    snapshot = {
        "version": SNAPSHOT_VERSION,
        "generated_at": now.isoformat(),
        "season": season,
        "next": {"gw": next_gw, "deadline": _iso(next_dl)},
        "gameweeks": gameweeks(ev),
        "fixtures": fixtures(dim, season),
        "forecast": {"gws": gws, "made_at": made},
        "players": rows,
        "plan": plan,
        "team": team_section(team, pm),
        "bets": bets_section(lake),
        "props": props_section(lake, store, season),
        "benchmarks": benchmarks_section(lake, store, season, now, ev, snap, team, pm, fpl_get),
        "health": health_section(repo, token),
        "data": {
            "feeds": feeds,
            "silver": silver_tables(lake),
            "checks": pipeline_checks(lake, season, now, ev, feeds, dim, weeks, made),
        },
        "lab": load_log(log_path),
    }
    out: dict[str, Any] = clean(snapshot)
    return out
