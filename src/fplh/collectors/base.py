"""Collector adapter seam (ARCHITECTURE.md §5.3.6).

Every source sits behind this protocol so a provider can be swapped without touching
downstream code. Collectors only fetch; persisting is the caller's job via
:func:`fplh.lake.bronze.write_bronze`.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Protocol

import httpx

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
    headers: dict[str, str] | None = None,
    redact: Iterable[str] = (),
    keep_headers: Iterable[str] = (),
) -> BronzeRecord:
    """GET ``url`` and wrap the response. ``observed_at`` is stamped on receipt.

    Query parameters named in ``redact`` (e.g. API keys) are replaced in the stored URL;
    response headers named in ``keep_headers`` are kept in the bronze metadata.
    """
    response = await client.get(url, params=query, headers=headers)
    return BronzeRecord(
        source=source,
        endpoint=endpoint,
        url=redact_url(response.request.url, redact),
        http_status=response.status_code,
        observed_at=clock.now(),
        payload=response.content,
        params=dict(params or {}),
        content_type=response.headers.get("Content-Type"),
        response_headers={
            h.lower(): response.headers[h] for h in keep_headers if h in response.headers
        },
    )


def redact_url(url: httpx.URL, names: Iterable[str]) -> str:
    names = set(names)
    if not names:
        return str(url)
    params = [(k, "REDACTED" if k in names else v) for k, v in url.params.multi_items()]
    return str(url.copy_with(params=params))
