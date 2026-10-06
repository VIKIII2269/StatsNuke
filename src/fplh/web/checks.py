"""Data checks for the website: what the lake holds, how fresh each feed is, and whether
the live pipeline did each job it owes (captures around deadlines, post-gameweek pulls,
odds credits, a decision for every gameweek)."""

from __future__ import annotations

import json
import re
from typing import Any

import pandas as pd

from fplh.lake.storage import Lake

# feed → (warn after, fail after) in hours since the last capture
FRESHNESS: dict[str, tuple[float, float]] = {
    "fpl/bootstrap-static": (4, 12),
    "fpl/fixtures": (4, 12),
    "odds_api/odds": (96, 216),
    "odds_api/events": (96, 216),
    "understat/league": (30, 96),
    "football_data/E0": (96, 240),
    "vaastav/merged_gw": (96, 240),
}
OBS = re.compile(r"obs=(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})Z")
ODDS_MONTHLY = 500
ODDS_FLOOR = 10


def _status(age_h: float | None, limits: tuple[float, float] | None) -> str:
    if limits is None:
        return "info"
    if age_h is None:
        return "fail"
    warn, fail = limits
    return "ok" if age_h <= warn else "warn" if age_h <= fail else "fail"


def bronze_feeds(lake: Lake, now: pd.Timestamp) -> list[dict[str, Any]]:
    """One row per source/endpoint: captures, first and last capture, age and status."""
    feeds: dict[str, list[str]] = {}
    for key in lake.list("bronze/"):
        if not key.endswith(".meta.json"):
            continue
        parts = key.split("/")
        src = next((p[7:] for p in parts if p.startswith("source=")), None)
        end = next((p[9:] for p in parts if p.startswith("endpoint=")), None)
        m = OBS.search(parts[-1])
        if src and end and m:
            feeds.setdefault(f"{src}/{end}", []).append(m.group(1))
    out = []
    for feed, stamps in sorted(feeds.items()):
        first, last = min(stamps), max(stamps)
        last_ts = pd.Timestamp(last, tz="UTC")
        age = (now - last_ts).total_seconds() / 3600
        out.append(
            {
                "feed": feed,
                "captures": len(stamps),
                "first": pd.Timestamp(first, tz="UTC").isoformat(),
                "last": last_ts.isoformat(),
                "age_hours": round(age, 1),
                "status": _status(age, FRESHNESS.get(feed)),
            }
        )
    return out


def silver_tables(lake: Lake) -> list[dict[str, Any]]:
    """Rows per silver table, from the parquet footers (no data read)."""
    import pyarrow.parquet as pq

    tables: dict[str, int] = {}
    for key in lake.list("silver/"):
        if not key.endswith(".parquet"):
            continue
        name = key.split("/")[1]
        with lake.fs.open(lake.path(key), "rb") as f:
            tables[name] = tables.get(name, 0) + int(pq.ParquetFile(f).metadata.num_rows)
    return [{"table": t, "rows": n} for t, n in sorted(tables.items())]


def _check(name: str, status: str, detail: str) -> dict[str, str]:
    return {"name": name, "status": status, "detail": detail}


def pipeline_checks(
    lake: Lake,
    season: str,
    now: pd.Timestamp,
    events: pd.DataFrame,
    feeds: list[dict[str, Any]],
    fixtures: pd.DataFrame,
    team_weeks: dict[str, dict[str, Any]],
    forecast_made: str | None,
) -> list[dict[str, str]]:
    """The live pipeline's obligations, each ok / warn / fail with a plain explanation.
    ``events``: index gameweek, deadline_at, finished, data_checked."""
    checks = []
    by_feed = {f["feed"]: f for f in feeds}

    boot = by_feed.get("fpl/bootstrap-static")
    checks.append(
        _check(
            "FPL snapshot",
            boot["status"] if boot else "fail",
            f"last capture {boot['age_hours']:.1f} h ago (every 3 h expected)"
            if boot
            else "no FPL snapshot in the lake",
        )
    )

    past = events[pd.to_datetime(events["deadline_at"], utc=True) < now]
    if len(past) and boot:
        last_gw = int(past.index[-1])
        deadline = pd.Timestamp(str(past.loc[last_gw, "deadline_at"]))
        stamps = [
            pd.Timestamp(m.group(1), tz="UTC")
            for k in lake.list("bronze/source=fpl/endpoint=bootstrap-static/")
            if (m := OBS.search(k)) and k.endswith(".meta.json")
        ]
        window = [s for s in stamps if deadline - pd.Timedelta(hours=24) <= s < deadline]
        if stamps and min(stamps) > deadline - pd.Timedelta(hours=24):
            status, detail = "info", "captures began after this deadline"
        else:
            status = "ok" if len(window) >= 6 else "warn" if window else "fail"
            detail = f"{len(window)} snapshots in the 24 h before the deadline (hourly job)"
        checks.append(_check(f"Deadline captures, GW{last_gw}", status, detail))

    final = [int(g) for g in events.index[events["data_checked"].astype(bool)]]
    pulled: list[int] = []
    if lake.exists("state/fpl_post_gw.json"):
        pulled = [int(g) for g in json.loads(lake.get_bytes("state/fpl_post_gw.json"))["collected"]]
    start = min(pulled, default=None)  # the collector began mid-season
    missing = [g for g in final if g not in pulled and (start is None or g >= start)]
    checks.append(
        _check(
            "Post-gameweek pulls",
            "ok" if not missing else "warn",
            f"pulled GW {', '.join(map(str, sorted(pulled))) or 'none'}"
            + (f"; missing {', '.join(map(str, missing))}" if missing else ""),
        )
    )

    if lake.exists("state/odds_budget.json"):
        b = json.loads(lake.get_bytes("state/odds_budget.json"))
        used = int(b.get("used", 0))
        left = ODDS_MONTHLY - used
        checks.append(
            _check(
                "Odds API credits",
                "ok" if left > ODDS_FLOOR else "warn",
                f"{used} of {ODDS_MONTHLY} used in {b.get('month')}; floor {ODDS_FLOOR}",
            )
        )

    fx = fixtures[fixtures["season"] == season]
    no_round = int(fx["round"].isna().sum())
    checks.append(
        _check(
            "Fixture schedule",
            "ok" if len(fx) == 380 and no_round == 0 else "warn",
            f"{len(fx)} fixtures, {no_round} without a gameweek",
        )
    )

    decided = sorted(int(g) for g, w in team_weeks.items() if w.get("advised"))
    if decided:
        owed = [int(g) for g in past.index if int(g) >= decided[0]]
        gaps = [g for g in owed if g not in decided]
        checks.append(
            _check(
                "Model team decisions",
                "ok" if not gaps else "fail",
                f"decided GW {decided[0]}–{decided[-1]}"
                + (f"; missed {', '.join(map(str, gaps))}" if gaps else ""),
            )
        )
    else:
        checks.append(_check("Model team decisions", "info", "first decision due 30 h before GW6"))

    if forecast_made:
        age = (now - pd.Timestamp(forecast_made)).total_seconds() / 3600
        checks.append(
            _check(
                "Forecast for the next deadline",
                "ok" if age <= 26 else "warn",
                f"made {age:.0f} h ago",
            )
        )
    else:
        checks.append(_check("Forecast for the next deadline", "warn", "none stored yet"))

    hist = [by_feed.get(f) for f in ("vaastav/merged_gw", "football_data/E0", "understat/league")]
    stale = [h["feed"] for h in hist if h and h["status"] == "fail"]
    checks.append(
        _check(
            "Current-season history",
            "ok" if not stale and all(hist) else "warn",
            "vaastav, football-data and Understat refreshed"
            if not stale and all(hist)
            else f"stale: {', '.join(stale) or 'missing feeds'}",
        )
    )
    return checks
