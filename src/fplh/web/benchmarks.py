"""Benchmarks against real FPL managers and the OpenFPL replica, for the website.

* **Manager panels.** FPL publishes the overall standings (classic league 314) and every
  manager's gameweek history. Once per season, at the first run, the site picks a panel of
  managers at three overall ranks: the median manager (p50: rank N/2), the top 10 % (p90:
  rank N/10) and the top 1 % (p99: rank N/100), ``PANEL_SIZE`` consecutive entries each.
  Their gameweek points net of hits are fetched once per finalised gameweek, and the panel
  median is the benchmark. The panel is chosen when the model team starts, so the
  comparison from then on is forward-looking.
* **OpenFPL.** The live model team stores the OpenFPL replica's forecast at each deadline
  next to model v2's (``live.team.save_replica``). After the gameweek, both are scored on
  the same player-fixtures (MSE and MAE of expected against actual points).
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Iterable
from typing import Any

import numpy as np
import pandas as pd

from fplh.lake.storage import Lake

API = "https://fantasy.premierleague.com/api"
STANDINGS = "/leagues-classic/314/standings/?page_standings={page}"
HISTORY = "/entry/{entry}/history/"
PANELS = {"p50": 0.5, "p90": 0.1, "p99": 0.01}  # share of managers ranked above the panel
PANEL_SIZE = 20
PAGE = 50
STATE_KEY = "state/benchmarks/{season}.json"

Getter = Callable[[str], Any]


def http_getter(user_agent: str, pause_s: float = 0.25) -> Getter:
    """A polite JSON getter for the FPL API (one request at a time)."""
    import httpx

    client = httpx.Client(
        headers={"User-Agent": user_agent, "Accept": "application/json"},
        timeout=30.0,
        follow_redirects=True,
    )

    def get(path: str) -> Any:
        time.sleep(pause_s)
        r = client.get(API + path)
        r.raise_for_status()
        return r.json()

    return get


def load_state(lake: Lake, season: str) -> dict[str, Any]:
    key = STATE_KEY.format(season=season)
    if lake.exists(key):
        state: dict[str, Any] = json.loads(lake.get_bytes(key))
        return state
    return {"panels": {}, "points": {}, "fetched": []}


def update_panels(
    lake: Lake,
    season: str,
    total_players: int,
    finalised: Iterable[int],
    get: Getter,
    now: pd.Timestamp,
) -> dict[str, Any]:
    """Choose the panels (first run only), then fetch every panel manager's history when a
    gameweek has been finalised since the last fetch. Returns the stored state."""
    state = load_state(lake, season)
    changed = False
    for name, share in PANELS.items():
        if name in state["panels"] or total_players <= 0:
            continue
        rank = max(1, round(total_players * share))
        page = (rank - 1) // PAGE + 1
        start = (rank - 1) % PAGE
        results = get(STANDINGS.format(page=page))["standings"]["results"]
        if start + PANEL_SIZE > len(results) == PAGE:  # the panel runs onto the next page
            results += get(STANDINGS.format(page=page + 1))["standings"]["results"]
        chosen = results[start : start + PANEL_SIZE] or results[:PANEL_SIZE]
        state["panels"][name] = {
            "rank": rank,
            "share": share,
            "chosen_at": now.isoformat(),
            "entries": [int(r["entry"]) for r in chosen],
        }
        changed = True
    done = sorted(int(g) for g in finalised)
    if done and done != state.get("fetched"):
        for panel in state["panels"].values():
            for entry in panel["entries"]:
                hist = get(HISTORY.format(entry=entry))["current"]
                state["points"][str(entry)] = {
                    str(h["event"]): int(h["points"]) - int(h.get("event_transfers_cost", 0))
                    for h in hist
                }
        state["fetched"] = done
        changed = True
    if changed:
        key = STATE_KEY.format(season=season)
        lake.put_bytes(key, json.dumps(state, sort_keys=True).encode(), overwrite=True)
    return state


def panel_medians(state: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    """Per panel, per gameweek: the median points (net of hits) and how many managers."""
    out: dict[str, list[dict[str, Any]]] = {}
    for name, panel in state.get("panels", {}).items():
        by_gw: dict[int, list[int]] = {}
        for entry in panel["entries"]:
            for gw, pts in state.get("points", {}).get(str(entry), {}).items():
                by_gw.setdefault(int(gw), []).append(int(pts))
        out[name] = [
            {"gw": gw, "points": float(np.median(v)), "n": len(v)}
            for gw, v in sorted(by_gw.items())
        ]
    return out


def openfpl_live(
    v2: pd.DataFrame, replica: pd.DataFrame, outcomes: pd.DataFrame
) -> list[dict[str, Any]]:
    """Per gameweek: MSE and MAE of model v2 and the OpenFPL replica on the same
    player-fixtures. ``v2`` and ``replica``: gw, player_uid, fixture_uid, expected_points;
    ``outcomes``: player_uid, fixture_uid, total_points."""
    keys = ["gw", "player_uid", "fixture_uid"]
    a = v2[[*keys, "expected_points"]].rename(columns={"expected_points": "v2"})
    b = replica[[*keys, "expected_points"]].rename(columns={"expected_points": "replica"})
    rows = a.merge(b, on=keys).merge(outcomes, on=["player_uid", "fixture_uid"])
    rows = rows.dropna(subset=["v2", "replica", "total_points"])
    out = []
    for gw, g in rows.groupby("gw"):
        y = g["total_points"].to_numpy(float)
        res: dict[str, Any] = {"gw": int(str(gw)), "n": len(g)}
        for col in ("v2", "replica"):
            err = g[col].to_numpy(float) - y
            res[f"mse_{col}"] = round(float((err**2).mean()), 4)
            res[f"mae_{col}"] = round(float(np.abs(err).mean()), 4)
        out.append(res)
    return out
