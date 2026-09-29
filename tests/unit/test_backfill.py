from __future__ import annotations

import json
from datetime import UTC, datetime

import httpx
import respx

from fplh.clock import FixedClock
from fplh.collectors import football_data, odds, understat, vaastav
from fplh.collectors.backfill import backfill_files, season_label, season_start_year
from fplh.collectors.http import HttpClient
from fplh.lake.bronze import latest_by_params, list_bronze, parse_key, read_bronze
from fplh.lake.storage import Lake

NOW = datetime(2026, 9, 29, 12, tzinfo=UTC)


def http() -> HttpClient:
    return HttpClient(user_agent="t", requests_per_second=1000, backoff_initial_s=0, max_attempts=2)


def test_season_helpers() -> None:
    assert season_start_year(datetime(2026, 9, 1, tzinfo=UTC)) == 2026
    assert season_start_year(datetime(2027, 3, 1, tzinfo=UTC)) == 2026
    assert season_label(2026) == "2026-27"
    assert football_data.season_code(1999) == "9900"
    assert football_data.season_code(2026) == "2627"


def test_spec_builders_mark_current_season_for_refresh() -> None:
    specs = vaastav.vaastav_specs([2025, 2026], current_season=2026)
    merged = {s.params["season"]: s for s in specs if s.endpoint == "merged_gw"}
    assert not merged["2025_26"].refresh and merged["2026_27"].refresh
    assert merged["2025_26"].url.endswith("/2025-26/gws/merged_gw.csv")
    assert any(s.optional for s in specs if s.endpoint == "fixtures")
    fd = football_data.football_data_specs([2025], current_season=2026)
    assert {s.url.rsplit("/", 2)[-2:][1] for s in fd} == {"E0.csv", "E1.csv"}


@respx.mock
async def test_backfill_writes_skips_and_refreshes(lake: Lake) -> None:
    specs = vaastav.vaastav_specs([2025], current_season=2026, base="https://v.test")
    respx.get("https://v.test/2025-26/fixtures.csv").mock(return_value=httpx.Response(404))
    respx.get(url__regex=r"https://v\.test/.*").mock(
        return_value=httpx.Response(200, text="a,b\n1,2\n")
    )
    async with http() as c:
        first = await backfill_files(c, FixedClock(NOW), lake, "vaastav", specs)
        second = await backfill_files(c, FixedClock(NOW.replace(hour=13)), lake, "vaastav", specs)
    assert len(first.written) == 4 and first.missing == ["https://v.test/2025-26/fixtures.csv"]
    assert second.written == [] and len(second.skipped) == 4  # completed season: skip
    current = vaastav.vaastav_specs([2026], current_season=2026, base="https://v.test")
    async with http() as c:
        third = await backfill_files(c, FixedClock(NOW.replace(hour=14)), lake, "vaastav", current)
    assert len(third.written) == 4  # current season: refresh (new observation)


@respx.mock
async def test_backfill_reports_failures(lake: Lake) -> None:
    specs = football_data.football_data_specs([2024], current_season=2026, base="https://fd.test")
    respx.get("https://fd.test/2425/E0.csv").mock(return_value=httpx.Response(200, text="x"))
    respx.get("https://fd.test/2425/E1.csv").mock(return_value=httpx.Response(403))
    async with http() as c:
        res = await backfill_files(c, FixedClock(NOW), lake, "football_data", specs)
    assert len(res.written) == 1 and res.failed == ["https://fd.test/2425/E1.csv: HTTP 403"]


@respx.mock
async def test_understat_two_stage(lake: Lake) -> None:
    league = {"datesData": [{"id": "1", "isResult": True}, {"id": "2", "isResult": False}]}
    route = respx.get("https://u.test/getLeagueData/EPL/2025").mock(
        return_value=httpx.Response(200, json=league)
    )
    respx.get("https://u.test/getMatchData/1").mock(return_value=httpx.Response(200, json={}))
    async with http() as c:
        await backfill_files(
            c,
            FixedClock(NOW),
            lake,
            "understat",
            understat.league_specs([2025], 2026, "https://u.test"),
        )
        specs = understat.match_specs(lake, "https://u.test")
        assert [s.params for s in specs] == [{"match": "1"}]  # unfinished matches wait
        await backfill_files(c, FixedClock(NOW), lake, "understat", specs)
    assert route.calls.last.request.headers["X-Requested-With"] == "XMLHttpRequest"
    assert len(list_bronze(lake, "understat", "match")) == 1


@respx.mock
async def test_odds_key_is_redacted_and_credits_kept(lake: Lake) -> None:
    respx.get("https://o.test/sports/soccer_epl/odds").mock(
        return_value=httpx.Response(
            200, json=[], headers={"x-requests-used": "42", "x-requests-remaining": "458"}
        )
    )
    async with http() as c:
        res = await backfill_files(
            c,
            FixedClock(NOW),
            lake,
            "odds_api",
            [odds.odds_spec("SECRET", "closing", "https://o.test")],
        )
    meta, _ = read_bronze(lake, res.written[0])
    assert "SECRET" not in json.dumps(meta)
    assert "apiKey=REDACTED" in meta["url"]
    assert meta["response_headers"] == {"x-requests-used": "42", "x-requests-remaining": "458"}
    assert odds.call_cost() == 2


def test_parse_key_roundtrip(lake: Lake) -> None:
    from fplh.lake.bronze import BronzeRecord, write_bronze

    key = write_bronze(lake, BronzeRecord("s", "e", "u", 200, NOW, b"{}", {"b": "2", "a": "1"}))
    pk = parse_key(key)
    assert (pk.source, pk.endpoint, pk.observed_at, pk.params) == (
        "s",
        "e",
        NOW,
        {"a": "1", "b": "2"},
    )
    assert latest_by_params(lake, "s", "e") == {(("a", "1"), ("b", "2")): key}
