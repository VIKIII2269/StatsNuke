"""Collector adapter seam (ARCHITECTURE.md §5.3.6).

Every source sits behind this protocol so a provider can be swapped without touching
downstream code. Collectors only fetch; persisting is the caller's job via
:func:`fplh.lake.bronze.write_bronze`.
"""

from __future__ import annotations

from typing import Protocol

from fplh.clock import Clock
from fplh.collectors.http import HttpClient
from fplh.lake.bronze import BronzeRecord


class Collector(Protocol):
    source: str

    async def collect(self, client: HttpClient, clock: Clock) -> list[BronzeRecord]: ...


async def fetch_record(
    client: HttpClient,
    clock: Clock,
    *,
    source: str,
    endpoint: str,
    url: str,
    params: dict[str, str] | None = None,
    query: dict[str, str] | None = None,
) -> BronzeRecord:
    """GET ``url`` and wrap the response. ``observed_at`` is stamped on receipt."""
    response = await client.get(url, params=query)
    return BronzeRecord(
        source=source,
        endpoint=endpoint,
        url=str(response.request.url),
        http_status=response.status_code,
        observed_at=clock.now(),
        payload=response.content,
        params=dict(params or {}),
        content_type=response.headers.get("Content-Type"),
    )
