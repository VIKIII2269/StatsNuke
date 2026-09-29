from __future__ import annotations

import copy
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
import pytest
import respx
from typer.testing import CliRunner

from fplh.cli import app
from fplh.clock import FixedClock
from fplh.collectors.fpl import (
    CollectorError,
    FplApi,
    FplPostGameweekCollector,
    FplSnapshotCollector,
    PartialCollectionError,
    latest_checked_event,
    next_deadline,
)
from fplh.collectors.http import HttpClient
from fplh.lake.bronze import list_bronze, read_bronze
from fplh.lake.storage import Lake
from tests.conftest import load_sample

BASE = "https://fantasy.premierleague.com/api"
NOW = datetime(2026, 9, 11, 12, 0, tzinfo=UTC)  # GW3 deadline is 2026-09-12T10:00Z


@pytest.fixture
def bootstrap() -> dict[str, Any]:
    return load_sample("bootstrap-static.json")


def http() -> HttpClient:
    return HttpClient(user_agent="t", requests_per_second=1000, backoff_initial_s=0, max_attempts=2)


def test_next_deadline_and_latest_checked(bootstrap: dict[str, Any]) -> None:
    assert next_deadline(bootstrap, NOW) == datetime(2026, 9, 12, 10, 0, tzinfo=UTC)
    assert next_deadline(bootstrap, datetime(2027, 1, 1, tzinfo=UTC)) is None
    assert latest_checked_event(bootstrap) == 1


@respx.mock
async def test_snapshot(bootstrap: dict[str, Any]) -> None:
    respx.get(f"{BASE}/bootstrap-static/").mock(return_value=httpx.Response(200, json=bootstrap))
    respx.get(f"{BASE}/fixtures/").mock(return_value=httpx.Response(200, json=[]))
    async with http() as c:
        recs = await FplSnapshotCollector(FplApi(), with_fixtures=True).collect(c, FixedClock(NOW))
    assert [r.endpoint for r in recs] == ["bootstrap-static", "fixtures"]
    assert all(r.observed_at == NOW for r in recs)


@pytest.mark.parametrize(("hours", "stored"), [(6.0, False), (24.0, True)])
@respx.mock
async def test_deadline_window(bootstrap: dict[str, Any], hours: float, stored: bool) -> None:
    respx.get(f"{BASE}/bootstrap-static/").mock(return_value=httpx.Response(200, json=bootstrap))
    async with http() as c:
        recs = await FplSnapshotCollector(FplApi(), only_within_hours=hours).collect(
            c, FixedClock(NOW)
        )
    assert bool(recs) is stored


@respx.mock
async def test_snapshot_http_error_raises() -> None:
    respx.get(f"{BASE}/bootstrap-static/").mock(return_value=httpx.Response(403))
    async with http() as c:
        with pytest.raises(CollectorError, match="403"):
            await FplSnapshotCollector(FplApi()).collect(c, FixedClock(NOW))


def _mock_post_gw(bootstrap: dict[str, Any], failing: int | None = None) -> None:
    respx.get(f"{BASE}/bootstrap-static/").mock(return_value=httpx.Response(200, json=bootstrap))
    respx.get(f"{BASE}/event/1/live/").mock(return_value=httpx.Response(200, json={"elements": []}))
    summary = load_sample("element-summary.json")
    for e in bootstrap["elements"]:
        status = 404 if e["id"] == failing else 200
        respx.get(f"{BASE}/element-summary/{e['id']}/").mock(
            return_value=httpx.Response(status, json=summary)
        )


@respx.mock
async def test_post_gameweek_pulls_every_element(bootstrap: dict[str, Any]) -> None:
    _mock_post_gw(bootstrap)
    async with http() as c:
        recs = await FplPostGameweekCollector(FplApi()).collect(c, FixedClock(NOW))
    assert [r.endpoint for r in recs[:2]] == ["bootstrap-static", "event-live"]
    assert recs[1].params == {"gw": "1"}
    assert sorted(r.params["element"] for r in recs[2:]) == ["1", "2", "3", "4"]


@respx.mock
async def test_post_gameweek_partial_failure_keeps_successes(bootstrap: dict[str, Any]) -> None:
    _mock_post_gw(bootstrap, failing=3)
    async with http() as c:
        with pytest.raises(PartialCollectionError) as info:
            await FplPostGameweekCollector(FplApi()).collect(c, FixedClock(NOW))
    assert len(info.value.failed) == 1
    assert sum(r.ok for r in info.value.records) == 5


@respx.mock
async def test_post_gameweek_requires_finalised_event(bootstrap: dict[str, Any]) -> None:
    nothing_checked = copy.deepcopy(bootstrap)
    for e in nothing_checked["events"]:
        e["data_checked"] = False
    respx.get(f"{BASE}/bootstrap-static/").mock(
        return_value=httpx.Response(200, json=nothing_checked)
    )
    async with http() as c:
        with pytest.raises(CollectorError, match="data_checked"):
            await FplPostGameweekCollector(FplApi()).collect(c, FixedClock(NOW))


@respx.mock
def test_cli_snapshot_writes_bronze(
    bootstrap: dict[str, Any], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("FPLH_LAKE_URI", str(tmp_path / "lake"))
    respx.get(f"{BASE}/bootstrap-static/").mock(return_value=httpx.Response(200, json=bootstrap))
    respx.get(f"{BASE}/fixtures/").mock(return_value=httpx.Response(200, json=[]))
    result = CliRunner().invoke(app, ["collect", "fpl-snapshot", "--with-fixtures"])
    assert result.exit_code == 0, result.output
    lake = Lake(str(tmp_path / "lake"))
    [key] = list_bronze(lake, "fpl", "bootstrap-static")
    meta, payload = read_bronze(lake, key)
    assert meta["http_status"] == 200
    assert b"deadline_time" in payload
    assert len(list_bronze(lake, "fpl", "fixtures")) == 1


@respx.mock
def test_cli_reports_blocked_api(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FPLH_LAKE_URI", str(tmp_path / "lake"))
    respx.get(f"{BASE}/bootstrap-static/").mock(return_value=httpx.Response(403))
    result = CliRunner().invoke(app, ["collect", "fpl-snapshot"])
    assert result.exit_code == 1
    assert "403" in result.output


@respx.mock
def test_cli_post_gw_skips_already_collected(
    bootstrap: dict[str, Any], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("FPLH_LAKE_URI", str(tmp_path / "lake"))
    _mock_post_gw(bootstrap)
    runner = CliRunner()
    first = runner.invoke(app, ["collect", "fpl-post-gw"])
    assert first.exit_code == 0, first.output
    lake = Lake(str(tmp_path / "lake"))
    assert len(list_bronze(lake, "fpl", "element-summary")) == 4
    assert b"[1]" in lake.get_bytes("state/fpl_post_gw.json")

    second = runner.invoke(app, ["collect", "fpl-post-gw"])
    assert second.exit_code == 0, second.output
    assert "nothing to store" in second.output
    assert len(list_bronze(lake, "fpl", "element-summary")) == 4
