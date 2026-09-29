from __future__ import annotations

import time

import httpx
import pytest
import respx

from fplh.collectors.http import HttpClient, RateLimiter, RetryableStatusError, _parse_retry_after

URL = "https://example.test/api/x/"


def client(**kw: object) -> HttpClient:
    opts: dict[str, object] = {
        "user_agent": "fplh-test",
        "requests_per_second": 1000.0,
        "backoff_initial_s": 0.0,
        "max_attempts": 3,
    }
    opts.update(kw)
    return HttpClient(**opts)  # type: ignore[arg-type]


@respx.mock
async def test_retries_server_errors_then_succeeds() -> None:
    route = respx.get(URL).mock(
        side_effect=[httpx.Response(503), httpx.Response(500), httpx.Response(200, json={"ok": 1})]
    )
    async with client() as c:
        resp = await c.get(URL)
    assert resp.status_code == 200
    assert route.call_count == 3


@respx.mock
async def test_honours_retry_after() -> None:
    respx.get(URL).mock(
        side_effect=[httpx.Response(429, headers={"Retry-After": "0"}), httpx.Response(200)]
    )
    async with client() as c:
        assert (await c.get(URL)).status_code == 200


@respx.mock
async def test_gives_up_after_max_attempts() -> None:
    route = respx.get(URL).mock(return_value=httpx.Response(502))
    async with client() as c:
        with pytest.raises(RetryableStatusError):
            await c.get(URL)
    assert route.call_count == 3


@respx.mock
async def test_client_errors_are_not_retried() -> None:
    route = respx.get(URL).mock(return_value=httpx.Response(404))
    async with client() as c:
        assert (await c.get(URL)).status_code == 404
    assert route.call_count == 1


@respx.mock
async def test_transport_errors_are_retried() -> None:
    route = respx.get(URL).mock(side_effect=[httpx.ConnectError("boom"), httpx.Response(200)])
    async with client() as c:
        assert (await c.get(URL)).status_code == 200
    assert route.call_count == 2


@respx.mock
async def test_identifies_itself() -> None:
    route = respx.get(URL).mock(return_value=httpx.Response(200))
    async with client(user_agent="StatsNuke-test/1") as c:
        await c.get(URL)
    assert route.calls.last.request.headers["User-Agent"] == "StatsNuke-test/1"


async def test_rate_limiter_spaces_requests() -> None:
    limiter = RateLimiter(requests_per_second=20)
    start = time.monotonic()
    for _ in range(4):
        await limiter.acquire()
    assert time.monotonic() - start >= 0.14  # 3 gaps of 50 ms


def test_parse_retry_after() -> None:
    assert _parse_retry_after("5") == 5.0
    assert _parse_retry_after(None) is None
    assert _parse_retry_after("garbage") is None
    assert _parse_retry_after("Wed, 21 Oct 2015 07:28:00 GMT") == 0.0
