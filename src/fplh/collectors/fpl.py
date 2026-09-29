"""Official FPL API collectors (ARCHITECTURE.md §5.2).

Snapshot data (prices, status, ``chance_of_playing_*``, news, ownership, ``ep_next``)
exists only as current state: a missed snapshot is lost forever (§5.3.2).
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any

from fplh.clock import Clock
from fplh.collectors.base import fetch_record
from fplh.collectors.http import HttpClient
from fplh.lake.bronze import BronzeRecord

SOURCE = "fpl"
DEFAULT_BASE_URL = "https://fantasy.premierleague.com/api"


class CollectorError(RuntimeError):
    pass


def _parse_ts(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def next_deadline(bootstrap: dict[str, Any], now: datetime) -> datetime | None:
    """Earliest gameweek deadline strictly after ``now``, if any."""
    upcoming = [
        _parse_ts(e["deadline_time"])
        for e in bootstrap.get("events", [])
        if e.get("deadline_time") and _parse_ts(e["deadline_time"]) > now
    ]
    return min(upcoming, default=None)


def latest_checked_event(bootstrap: dict[str, Any]) -> int | None:
    """Most recent gameweek whose scores FPL has finalised (``data_checked``)."""
    done = [e["id"] for e in bootstrap.get("events", []) if e.get("data_checked")]
    return max(done, default=None)


def element_ids(bootstrap: dict[str, Any]) -> list[int]:
    return sorted(int(e["id"]) for e in bootstrap.get("elements", []))


@dataclass
class FplApi:
    base_url: str = DEFAULT_BASE_URL

    def url(self, path: str) -> str:
        return f"{self.base_url.rstrip('/')}/{path.strip('/')}/"

    async def fetch(
        self,
        client: HttpClient,
        clock: Clock,
        endpoint: str,
        path: str,
        params: dict[str, str] | None = None,
    ) -> BronzeRecord:
        return await fetch_record(
            client, clock, source=SOURCE, endpoint=endpoint, url=self.url(path), params=params
        )

    async def bootstrap(self, client: HttpClient, clock: Clock) -> BronzeRecord:
        return await self.fetch(client, clock, "bootstrap-static", "bootstrap-static")

    async def fixtures(self, client: HttpClient, clock: Clock) -> BronzeRecord:
        return await self.fetch(client, clock, "fixtures", "fixtures")

    async def event_live(self, client: HttpClient, clock: Clock, gw: int) -> BronzeRecord:
        return await self.fetch(client, clock, "event-live", f"event/{gw}/live", {"gw": str(gw)})

    async def element_summary(self, client: HttpClient, clock: Clock, element: int) -> BronzeRecord:
        return await self.fetch(
            client,
            clock,
            "element-summary",
            f"element-summary/{element}",
            {"element": str(element)},
        )


def _require_ok(rec: BronzeRecord) -> dict[str, Any]:
    if not rec.ok:
        raise CollectorError(f"{rec.url} returned HTTP {rec.http_status}")
    data = rec.json()
    if not isinstance(data, dict):
        raise CollectorError(f"{rec.url} returned non-object JSON")
    return data


@dataclass
class FplSnapshotCollector:
    """``bootstrap-static`` (optionally with ``fixtures``).

    With ``only_within_hours`` set, the bootstrap is fetched but discarded unless a
    deadline falls within that many hours, which gives the hourly deadline-day cadence.
    """

    api: FplApi
    with_fixtures: bool = False
    only_within_hours: float | None = None
    source: str = SOURCE

    async def collect(self, client: HttpClient, clock: Clock) -> list[BronzeRecord]:
        boot = await self.api.bootstrap(client, clock)
        data = _require_ok(boot)
        if self.only_within_hours is not None:
            nd = next_deadline(data, boot.observed_at)
            if nd is None or nd - boot.observed_at > timedelta(hours=self.only_within_hours):
                return []
        records = [boot]
        if self.with_fixtures:
            fx = await self.api.fixtures(client, clock)
            _require_ok_list(fx)
            records.append(fx)
        return records


def _require_ok_list(rec: BronzeRecord) -> list[Any]:
    if not rec.ok:
        raise CollectorError(f"{rec.url} returned HTTP {rec.http_status}")
    data = rec.json()
    if not isinstance(data, list):
        raise CollectorError(f"{rec.url} returned non-array JSON")
    return data


@dataclass
class FplFixturesCollector:
    api: FplApi
    source: str = SOURCE

    async def collect(self, client: HttpClient, clock: Clock) -> list[BronzeRecord]:
        fx = await self.api.fixtures(client, clock)
        _require_ok_list(fx)
        return [fx]


@dataclass
class FplPostGameweekCollector:
    """Finalised per-fixture stats: ``event/{gw}/live`` + every ``element-summary``.

    Run only after lockdown (§5.3.3). ``gw=None`` picks the latest ``data_checked`` event;
    if that gameweek is in ``skip_gws`` (already collected) nothing is returned.
    """

    api: FplApi
    gw: int | None = None
    max_concurrency: int = 4
    skip_gws: frozenset[int] = frozenset()
    source: str = SOURCE

    async def collect(self, client: HttpClient, clock: Clock) -> list[BronzeRecord]:
        boot = await self.api.bootstrap(client, clock)
        data = _require_ok(boot)
        gw = self.gw if self.gw is not None else latest_checked_event(data)
        if gw is None:
            raise CollectorError("no finalised (data_checked) gameweek yet")
        if self.gw is None and gw in self.skip_gws:
            return []
        records = [boot, await self.api.event_live(client, clock, gw)]
        _require_ok(records[-1])

        sem = asyncio.Semaphore(self.max_concurrency)

        async def one(element: int) -> BronzeRecord:
            async with sem:
                return await self.api.element_summary(client, clock, element)

        summaries = await asyncio.gather(*(one(e) for e in element_ids(data)))
        failed = [r.url for r in summaries if not r.ok]
        records.extend(summaries)
        if failed:
            # Still hand back everything so the caller persists what succeeded.
            raise PartialCollectionError(records, failed)
        return records


class PartialCollectionError(CollectorError):
    def __init__(self, records: list[BronzeRecord], failed: list[str]) -> None:
        super().__init__(f"{len(failed)} request(s) failed, first: {failed[0]}")
        self.records = records
        self.failed = failed
