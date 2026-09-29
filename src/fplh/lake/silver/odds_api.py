"""The Odds API bronze → live rows of ``snap_odds`` (observed at fetch time)."""

from __future__ import annotations

import json

import pandas as pd

from fplh.entities.teams import TeamResolver
from fplh.lake.bronze import list_bronze, read_bronze
from fplh.lake.silver.common import season_label
from fplh.lake.storage import Lake

SOURCE = "odds_api"


def normalise(lake: Lake, teams: TeamResolver) -> pd.DataFrame:
    rows = []
    for key in list_bronze(lake, SOURCE, "odds"):
        meta, payload = read_bronze(lake, key)
        if meta["http_status"] != 200:
            continue
        obs = pd.Timestamp(meta["observed_at_us"])
        for ev in json.loads(payload):
            kickoff = pd.Timestamp(ev["commence_time"])
            start = kickoff.year if kickoff.month >= 7 else kickoff.year - 1
            home, away = teams.uid(ev["home_team"]), teams.uid(ev["away_team"])
            for book in ev.get("bookmakers", []):
                for market in book.get("markets", []):
                    for o in market.get("outcomes", []):
                        if market["key"] == "h2h":
                            outcome = {ev["home_team"]: "home", ev["away_team"]: "away"}.get(
                                o["name"], "draw"
                            )
                            mkt, line = "1x2", float("nan")
                        elif market["key"] == "totals":
                            outcome, mkt, line = (
                                o["name"].lower(),
                                "total",
                                float(o.get("point", "nan")),
                            )
                        else:
                            continue
                        rows.append(
                            {
                                "season": season_label(start),
                                "division": "E0",
                                "kickoff_at": kickoff,
                                "home_team": home,
                                "away_team": away,
                                "bookmaker": book["key"],
                                "book_code": book["key"],
                                "market": mkt,
                                "line": line,
                                "outcome": outcome,
                                "price": float(o["price"]),
                                "is_closing": False,
                                "observed_at": obs,
                                "source": SOURCE,
                                "bronze_key": key,
                            }
                        )
    return pd.DataFrame(rows)
