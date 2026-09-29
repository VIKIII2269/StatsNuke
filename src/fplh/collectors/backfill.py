"""Shared machinery for file-style backfills (one request → one bronze object).

Objects are written as they arrive, so a long backfill that dies part-way keeps its
progress, and a re-run skips parameter sets already in bronze unless ``refresh`` is set
(used for the current season, whose files keep changing).
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime

import httpx

from fplh.clock import Clock
from fplh.collectors.base import fetch_record
from fplh.collectors.http import HttpClient, RetryableStatusError
from fplh.lake.bronze import latest_by_params, write_bronze
from fplh.lake.storage import Lake


@dataclass(frozen=True)
class FileSpec:
    endpoint: str
    url: str
    params: dict[str, str]
    refresh: bool = False  # fetch even if already in bronze (e.g. current season)
    optional: bool = False  # 404 is expected for some seasons
    headers: dict[str, str] | None = None
    query: dict[str, str] | None = None
    redact: tuple[str, ...] = ()
    keep_headers: tuple[str, ...] = ()


@dataclass
class BackfillResult:
    written: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)
    failed: list[str] = field(default_factory=list)

    def summary(self) -> str:
        return (
            f"written {len(self.written)}, skipped {len(self.skipped)} (already in bronze), "
            f"missing {len(self.missing)} (optional 404), failed {len(self.failed)}"
        )


def season_start_year(now: datetime) -> int:
    """EPL seasons start in August; July belongs to the new season's pre-window."""
    return now.year if now.month >= 7 else now.year - 1


def season_label(start_year: int, sep: str = "-") -> str:
    return f"{start_year}{sep}{(start_year + 1) % 100:02d}"


async def backfill_files(
    client: HttpClient,
    clock: Clock,
    lake: Lake,
    source: str,
    specs: Sequence[FileSpec],
    *,
    max_concurrency: int = 1,
) -> BackfillResult:
    result = BackfillResult()
    existing_by_endpoint: dict[str, set[tuple[tuple[str, str], ...]]] = {}
    todo: list[FileSpec] = []
    for spec in specs:
        if spec.endpoint not in existing_by_endpoint:
            existing_by_endpoint[spec.endpoint] = set(latest_by_params(lake, source, spec.endpoint))
        if (
            not spec.refresh
            and tuple(sorted(spec.params.items())) in existing_by_endpoint[spec.endpoint]
        ):
            result.skipped.append(spec.url)
        else:
            todo.append(spec)

    sem = asyncio.Semaphore(max_concurrency)

    async def one(spec: FileSpec) -> None:
        async with sem:
            try:
                rec = await fetch_record(
                    client,
                    clock,
                    source=source,
                    endpoint=spec.endpoint,
                    url=spec.url,
                    params=spec.params,
                    query=spec.query,
                    headers=spec.headers,
                    redact=spec.redact,
                    keep_headers=spec.keep_headers,
                )
            except (RetryableStatusError, httpx.HTTPError) as exc:
                result.failed.append(f"{spec.url}: {type(exc).__name__}: {exc}")
                return
        if rec.http_status == 404 and spec.optional:
            result.missing.append(spec.url)
        elif not rec.ok:
            result.failed.append(f"{spec.url}: HTTP {rec.http_status}")
        else:
            result.written.append(write_bronze(lake, rec))

    await asyncio.gather(*(one(s) for s in todo))
    return result
