"""``fplh`` command-line interface."""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from typing import Annotated

import httpx
import pandas as pd
import typer

from fplh.clock import SystemClock
from fplh.collectors.base import Collector
from fplh.collectors.fpl import (
    CollectorError,
    FplApi,
    FplFixturesCollector,
    FplPostGameweekCollector,
    FplSnapshotCollector,
    PartialCollectionError,
)
from fplh.collectors.http import HttpClient, RetryableStatusError
from fplh.lake.bronze import BronzeRecord, write_bronze
from fplh.lake.storage import Lake
from fplh.rules import load_rules
from fplh.rules.engine import score as score_points
from fplh.rules.golden import (
    check,
    golden_cache_path,
    load_known_exceptions,
    load_vaastav,
    vaastav_url,
)
from fplh.settings import REPO_ROOT, get_settings
from fplh.sources import load_sources

app = typer.Typer(no_args_is_help=True, help="EPL × FPL hybrid forecasting system.")
collect_app = typer.Typer(no_args_is_help=True, help="Fetch raw data into bronze.")
rules_app = typer.Typer(no_args_is_help=True, help="FPL scoring rules engine.")
golden_app = typer.Typer(no_args_is_help=True, help="Golden tests against official points.")
app.add_typer(collect_app, name="collect")
app.add_typer(rules_app, name="rules")
app.add_typer(golden_app, name="golden")

KNOWN_EXCEPTIONS = REPO_ROOT / "tests" / "golden" / "known_exceptions.yaml"
# Small mutable bookkeeping lives under state/ (never under bronze/, which is create-only).
POST_GW_STATE_KEY = "state/fpl_post_gw.json"


def _client(source: str) -> HttpClient:
    s = get_settings()
    cfg = load_sources()[source]
    return HttpClient(
        user_agent=s.user_agent,
        timeout_s=s.http_timeout_s,
        requests_per_second=cfg.requests_per_second,
    )


def _persist(lake: Lake, records: list[BronzeRecord]) -> None:
    for rec in records:
        key = write_bronze(lake, rec)
        typer.echo(f"wrote {key} ({len(rec.payload)} bytes, HTTP {rec.http_status})")


async def _run(collector: Collector, source: str) -> list[BronzeRecord]:
    async with _client(source) as client:
        return await collector.collect(client, SystemClock())


def _collect(collector: Collector, lake: Lake | None = None) -> list[BronzeRecord]:
    lake = lake or Lake(get_settings().lake_uri)
    try:
        records = asyncio.run(_run(collector, collector.source))
    except PartialCollectionError as exc:
        _persist(lake, [r for r in exc.records if r.ok])
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(1) from exc
    except (CollectorError, RetryableStatusError, httpx.HTTPError) as exc:
        typer.echo(f"error: {type(exc).__name__}: {exc}", err=True)
        raise typer.Exit(1) from exc
    if not records:
        typer.echo("nothing to store")
    _persist(lake, records)
    return records


def _fpl_api() -> FplApi:
    base = load_sources()["fpl"].base_url
    return FplApi(base) if base else FplApi()


@collect_app.command("fpl-snapshot")
def collect_fpl_snapshot(
    with_fixtures: Annotated[bool, typer.Option(help="Also fetch /fixtures.")] = False,
    only_within_hours: Annotated[
        float | None,
        typer.Option(help="Store only if a deadline is within this many hours."),
    ] = None,
) -> None:
    """Snapshot bootstrap-static (prices, status, news, ownership, ep_next)."""
    _collect(FplSnapshotCollector(_fpl_api(), with_fixtures, only_within_hours))


@collect_app.command("fpl-fixtures")
def collect_fpl_fixtures() -> None:
    """Fetch the fixture list (detects rescheduling, blanks and doubles)."""
    _collect(FplFixturesCollector(_fpl_api()))


@collect_app.command("fpl-post-gw")
def collect_fpl_post_gw(
    gw: Annotated[int | None, typer.Option(help="Gameweek; default: latest finalised.")] = None,
) -> None:
    """Finalised per-fixture stats for a gameweek (run after lockdown).

    Without ``--gw``, skips the latest finalised gameweek if it was already collected.
    """
    cfg = load_sources()["fpl"]
    lake = Lake(get_settings().lake_uri)
    state = json.loads(lake.get_bytes(POST_GW_STATE_KEY)) if lake.exists(POST_GW_STATE_KEY) else {}
    done = frozenset(int(g) for g in state.get("collected", []))
    records = _collect(
        FplPostGameweekCollector(_fpl_api(), gw, cfg.max_concurrency, skip_gws=done), lake
    )
    collected = [int(r.params["gw"]) for r in records if r.endpoint == "event-live"]
    if collected:
        state["collected"] = sorted(done | set(collected))
        lake.put_bytes(POST_GW_STATE_KEY, json.dumps(state).encode(), overwrite=True)


@rules_app.command("score")
def rules_score(
    season: Annotated[str, typer.Option(help="e.g. 2026/27")],
    csv: Annotated[Path, typer.Option(help="Player × fixture events (FPL column names).")],
    out: Annotated[Path | None, typer.Option(help="Write the breakdown CSV here.")] = None,
) -> None:
    """Score a CSV of match events and print (or write) the points breakdown."""
    events = pd.read_csv(csv)
    result = pd.concat(
        [events, score_points(events, load_rules(season)).add_prefix("pts_")], axis=1
    )
    if out:
        result.to_csv(out, index=False)
        typer.echo(f"wrote {out} ({len(result)} rows)")
    else:
        result.to_csv(sys.stdout, index=False)


@golden_app.command("fetch")
def golden_fetch(
    season: Annotated[list[str], typer.Option(help="Season(s), e.g. 2025/26.")],
) -> None:
    """Download vaastav merged_gw.csv for the given seasons into the local cache."""

    async def run() -> None:
        async with _client("vaastav") as client:
            for s in season:
                resp = await client.get(vaastav_url(s))
                if resp.status_code != 200:
                    raise typer.BadParameter(f"{s}: HTTP {resp.status_code} from {resp.url}")
                dest = golden_cache_path(get_settings().cache_dir, s)
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_bytes(resp.content)
                typer.echo(f"{s}: {len(resp.content)} bytes → {dest}")

    asyncio.run(run())


@golden_app.command("check")
def golden_check(
    season: Annotated[str, typer.Option(help="e.g. 2025/26")],
    csv: Annotated[Path | None, typer.Option(help="merged_gw-style CSV; default: cache.")] = None,
    show: Annotated[int, typer.Option(help="Mismatching rows to print.")] = 20,
) -> None:
    """Reproduce official total_points for a season; exits 1 on any mismatch."""
    path = csv or golden_cache_path(get_settings().cache_dir, season)
    if not path.exists():
        typer.echo(f"no data at {path}; run `fplh golden fetch --season {season}`", err=True)
        raise typer.Exit(2)
    report = check(
        load_vaastav(season, path),
        load_rules(season),
        load_known_exceptions(KNOWN_EXCEPTIONS, season),
    )
    typer.echo(report.summary())
    if not report.points_mismatches.empty:
        typer.echo(report.points_mismatches.head(show).to_string())
    if not report.ok:
        raise typer.Exit(1)


@app.command("version")
def version() -> None:
    from fplh import __version__

    typer.echo(json.dumps({"fplh": __version__}))


if __name__ == "__main__":  # pragma: no cover
    app()
