"""The Odds API (v4) for live EPL prices (paper benchmark only; ARCHITECTURE.md §2.2).

``/events`` is free and gives the kickoff schedule; ``/odds`` costs markets × regions
credits. The API key travels as a query parameter and is redacted before anything is
stored. Credit counters from the response headers are kept in bronze metadata.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from fplh.collectors.backfill import FileSpec

SOURCE = "odds_api"
DEFAULT_BASE = "https://api.the-odds-api.com/v4"
SPORT = "soccer_epl"
MARKETS = ("h2h", "totals")
REGIONS = ("uk",)
CREDIT_HEADERS = ("x-requests-remaining", "x-requests-used", "x-requests-last")


def call_cost(markets: tuple[str, ...] = MARKETS, regions: tuple[str, ...] = REGIONS) -> int:
    return len(markets) * len(regions)


def events_spec(api_key: str, base: str = DEFAULT_BASE) -> FileSpec:
    return FileSpec(
        "events",
        f"{base.rstrip('/')}/sports/{SPORT}/events",
        {"sport": SPORT},
        refresh=True,
        query={"apiKey": api_key},
        redact=("apiKey",),
        keep_headers=CREDIT_HEADERS,
    )


def odds_spec(api_key: str, kind: str, base: str = DEFAULT_BASE) -> FileSpec:
    return FileSpec(
        "odds",
        f"{base.rstrip('/')}/sports/{SPORT}/odds",
        {"sport": SPORT, "markets": "-".join(MARKETS), "regions": "-".join(REGIONS), "plan": kind},
        refresh=True,
        query={
            "apiKey": api_key,
            "regions": ",".join(REGIONS),
            "markets": ",".join(MARKETS),
            "oddsFormat": "decimal",
            "dateFormat": "iso",
        },
        redact=("apiKey",),
        keep_headers=CREDIT_HEADERS,
    )


def kickoffs_from_events(events: list[dict[str, Any]]) -> list[datetime]:
    return sorted(datetime.fromisoformat(e["commence_time"].replace("Z", "+00:00")) for e in events)
