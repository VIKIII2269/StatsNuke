"""Shared polite HTTP client: identifying User-Agent, rate limit, retries with backoff."""

from __future__ import annotations

import asyncio
import email.utils
import random
import time
from datetime import UTC, datetime
from types import TracebackType
from typing import Self

import httpx
from tenacity import (
    AsyncRetrying,
    RetryCallState,
    retry_if_exception_type,
    stop_after_attempt,
)

RETRYABLE_STATUS = frozenset({408, 425, 429, 500, 502, 503, 504})


class RetryableStatusError(Exception):
    def __init__(self, response: httpx.Response) -> None:
        super().__init__(f"HTTP {response.status_code} for {response.request.url}")
        self.response = response
        self.retry_after = _parse_retry_after(response.headers.get("Retry-After"))


def _parse_retry_after(value: str | None) -> float | None:
    if not value:
        return None
    try:
        return max(0.0, float(value))
    except ValueError:
        pass
    try:
        dt = email.utils.parsedate_to_datetime(value)
    except (TypeError, ValueError):
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return max(0.0, (dt - datetime.now(UTC)).total_seconds())


class RateLimiter:
    """Minimum spacing between request starts, shared by all concurrent tasks."""

    def __init__(self, requests_per_second: float) -> None:
        self._interval = 1.0 / requests_per_second
        self._next = 0.0
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        async with self._lock:
            now = time.monotonic()
            wait = self._next - now
            self._next = max(now, self._next) + self._interval
        if wait > 0:
            await asyncio.sleep(wait)


class HttpClient:
    """Async GET with rate limiting and exponential-backoff retries.

    Retries transport errors and :data:`RETRYABLE_STATUS` responses, honouring
    ``Retry-After``. Other statuses (e.g. 404) are returned to the caller unchanged.
    """

    def __init__(
        self,
        *,
        user_agent: str,
        timeout_s: float = 30.0,
        requests_per_second: float = 1.0,
        max_attempts: int = 5,
        backoff_initial_s: float = 1.0,
        backoff_max_s: float = 60.0,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._client = httpx.AsyncClient(
            headers={"User-Agent": user_agent, "Accept": "application/json"},
            timeout=timeout_s,
            follow_redirects=True,
            transport=transport,
        )
        self._limiter = RateLimiter(requests_per_second)
        self._max_attempts = max_attempts
        self._backoff_initial = backoff_initial_s
        self._backoff_max = backoff_max_s

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        await self._client.aclose()

    def _wait(self, state: RetryCallState) -> float:
        exc = state.outcome.exception() if state.outcome else None
        if isinstance(exc, RetryableStatusError) and exc.retry_after is not None:
            return min(exc.retry_after, self._backoff_max)
        base = self._backoff_initial * 2 ** (state.attempt_number - 1)
        return float(min(self._backoff_max, base + random.uniform(0, self._backoff_initial)))

    async def get(self, url: str, params: dict[str, str] | None = None) -> httpx.Response:
        async for attempt in AsyncRetrying(
            stop=stop_after_attempt(self._max_attempts),
            wait=self._wait,
            retry=retry_if_exception_type((httpx.TransportError, RetryableStatusError)),
            reraise=True,
        ):
            with attempt:
                await self._limiter.acquire()
                response = await self._client.get(url, params=params)
                if response.status_code in RETRYABLE_STATUS:
                    raise RetryableStatusError(response)
                return response
        raise AssertionError("unreachable")  # pragma: no cover
