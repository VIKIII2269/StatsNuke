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
from fplh.collectors import football_data, odds, understat, vaastav
from fplh.collectors.backfill import BackfillResult, FileSpec, backfill_files, season_start_year
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
backfill_app = typer.Typer(no_args_is_help=True, help="Download historical files into bronze.")
report_app = typer.Typer(no_args_is_help=True, help="Data availability reports.")
silver_app = typer.Typer(no_args_is_help=True, help="Build validated silver tables.")
evaluate_app = typer.Typer(no_args_is_help=True, help="Walk-forward evaluation and checks.")
app.add_typer(collect_app, name="collect")
app.add_typer(backfill_app, name="backfill")
app.add_typer(report_app, name="report")
app.add_typer(silver_app, name="silver")
app.add_typer(evaluate_app, name="evaluate")
app.add_typer(rules_app, name="rules")
app.add_typer(golden_app, name="golden")

ODDS_STATE_KEY = "state/odds_budget.json"
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


def _seasons(first: int, seasons: list[int] | None) -> tuple[list[int], int]:
    current = season_start_year(SystemClock().now())
    return (seasons or list(range(first, current + 1))), current


def _backfill(source: str, specs: list[FileSpec], lake: Lake | None = None) -> BackfillResult:
    lake = lake or Lake(get_settings().lake_uri)
    cfg = load_sources()[source]

    async def run() -> BackfillResult:
        async with _client(source) as client:
            return await backfill_files(
                client, SystemClock(), lake, source, specs, max_concurrency=cfg.max_concurrency
            )

    result = asyncio.run(run())
    typer.echo(f"{source}: {result.summary()}")
    for line in result.failed[:20]:
        typer.echo(f"  failed: {line}", err=True)
    if result.failed:
        raise typer.Exit(1)
    return result


SeasonsOpt = Annotated[
    list[int] | None, typer.Option("--season", help="Season start year(s); default: all.")
]


@backfill_app.command("vaastav")
def backfill_vaastav(season: SeasonsOpt = None) -> None:
    """FPL history 2016/17→ from vaastav/Fantasy-Premier-League."""
    seasons, current = _seasons(vaastav.FIRST_SEASON, season)
    base = load_sources()["vaastav"].base_url or vaastav.DEFAULT_BASE
    _backfill(vaastav.SOURCE, vaastav.vaastav_specs(seasons, current, base))


@backfill_app.command("football-data")
def backfill_football_data(season: SeasonsOpt = None) -> None:
    """Results, match stats and odds (E0, E1) 1993/94→ from football-data.co.uk."""
    seasons, current = _seasons(football_data.FIRST_SEASON, season)
    base = load_sources()["football_data"].base_url or football_data.DEFAULT_BASE
    _backfill(football_data.SOURCE, football_data.football_data_specs(seasons, current, base=base))


@backfill_app.command("understat")
def backfill_understat(
    season: SeasonsOpt = None,
    matches: Annotated[bool, typer.Option(help="Also fetch per-match data.")] = True,
) -> None:
    """League data, then every finished match not yet in bronze (incremental)."""
    seasons, current = _seasons(understat.FIRST_SEASON, season)
    base = load_sources()["understat"].base_url or understat.DEFAULT_BASE
    lake = Lake(get_settings().lake_uri)
    _backfill(understat.SOURCE, understat.league_specs(seasons, current, base), lake)
    if matches:
        _backfill(understat.SOURCE, understat.match_specs(lake, base), lake)


@collect_app.command("odds")
def collect_odds(
    due_only: Annotated[
        bool, typer.Option("--due/--now", help="Only fire calls the budget planner has due.")
    ] = True,
) -> None:
    """EPL h2h + totals from The Odds API, within the monthly credit budget."""
    from datetime import UTC, datetime

    from fplh.collectors.odds_budget import due, plan_calls

    s = get_settings()
    if s.odds_api_key is None:
        typer.echo("FPLH_ODDS_API_KEY is not set; skipping odds collection")
        return
    key = s.odds_api_key.get_secret_value()
    base = load_sources()["odds_api"].base_url or odds.DEFAULT_BASE
    lake = Lake(s.lake_uri)
    now = SystemClock().now()
    month = f"{now:%Y-%m}"
    state = json.loads(lake.get_bytes(ODDS_STATE_KEY)) if lake.exists(ODDS_STATE_KEY) else {}
    if state.get("month") != month:
        state = {"month": month, "fired": [], "used": 0}

    ev = _backfill(odds.SOURCE, [odds.events_spec(key, base)], lake)
    from fplh.lake.bronze import read_bronze

    meta, payload = read_bronze(lake, ev.written[0])
    kickoffs = odds.kickoffs_from_events(json.loads(payload))
    used = int(meta["response_headers"].get("x-requests-used", state["used"]))
    month_start = datetime(now.year, now.month, 1, tzinfo=UTC)
    month_end = datetime(now.year + now.month // 12, now.month % 12 + 1, 1, tzinfo=UTC)
    plan = plan_calls(
        kickoffs,
        period_start=max(month_start, now),
        period_end=month_end,
        budget=s.odds_monthly_credits,
        used=used,
        cost=odds.call_cost(),
    )
    calls = due(plan, now, state["fired"]) if due_only else plan[:1]
    if not calls:
        typer.echo(f"no odds call due (credits used this month: {used})")
    for call in calls[:1]:
        res = _backfill(odds.SOURCE, [odds.odds_spec(key, call.kind, base)], lake)
        m, _ = read_bronze(lake, res.written[0])
        used = int(m["response_headers"].get("x-requests-used", used + call.credits))
        state["fired"].append(call.id)
        typer.echo(f"fired {call.id}; credits used this month: {used}")
    state["used"] = used
    lake.put_bytes(ODDS_STATE_KEY, json.dumps(state).encode(), overwrite=True)


@report_app.command("odds-columns")
def report_odds_columns(
    division: Annotated[str, typer.Option(help="E0 or E1")] = "E0",
) -> None:
    """Share of matches with each closing/pre-match odds family, per season (gap 3)."""
    from fplh.lake.bronze import latest_by_params, read_bronze
    from fplh.lake.parquet import write_parquet
    from fplh.lake.silver.football_data import odds_column_report

    lake = Lake(get_settings().lake_uri)
    frames = {}
    for params, key in sorted(latest_by_params(lake, football_data.SOURCE, division).items()):
        _, payload = read_bronze(lake, key)
        frames[dict(params)["season"]] = payload
    if not frames:
        typer.echo("no football-data in bronze; run `fplh backfill football-data`", err=True)
        raise typer.Exit(2)
    report = odds_column_report(frames)
    write_parquet(lake, f"gold/reports/odds_columns_{division}.parquet", report, ["season"])
    typer.echo(report.to_string(index=False))


@silver_app.command("build")
def silver_build(
    season: Annotated[
        list[str] | None, typer.Option("--season", help="Silver season label, e.g. 2025-26.")
    ] = None,
    dry_run: Annotated[bool, typer.Option(help="Validate and gate without writing.")] = False,
) -> None:
    """Normalise bronze → silver, resolve entities, run quality gates, write if all pass."""
    from fplh.lake.quality import QualityGateError
    from fplh.lake.silver.build import build

    lake = Lake(get_settings().lake_uri)
    try:
        report, tables = build(lake, seasons=set(season) if season else None, write=not dry_run)
    except QualityGateError as exc:
        for r in exc.results:
            typer.echo(r.line())
            if not r.passed and not r.failures.empty:
                typer.echo(r.failures.head(10).to_string())
        raise typer.Exit(1) from exc
    typer.echo(report.summary())
    cov = tables.get("entity_coverage")
    if cov is not None and not cov.empty:
        typer.echo(cov.to_string(index=False))
    typer.echo(
        f"wrote {len(report.written)} parts" if report.written else "dry run: nothing written"
    )


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


@golden_app.command("check-silver")
def golden_check_silver(
    season: Annotated[list[str] | None, typer.Option("--season", help="e.g. 2019-20")] = None,
) -> None:
    """Reproduce official points for every silver season (built by `fplh silver build`)."""
    from fplh.features.information_set import SilverStore
    from fplh.rules.golden import from_silver

    pm = SilverStore(Lake(get_settings().lake_uri)).get("fact_player_match")
    if pm.empty:
        typer.echo("no silver fact_player_match; run `fplh silver build`", err=True)
        raise typer.Exit(2)
    failed = False
    for s in season or sorted(pm["season"].unique()):
        report = check(
            from_silver(s, pm),
            load_rules(s),
            load_known_exceptions(KNOWN_EXCEPTIONS, s),
            bonus_from_bps=False,  # silver keeps every player, but not always every BPS row
        )
        typer.echo(report.summary())
        failed |= not report.ok
    if failed:
        raise typer.Exit(1)


@evaluate_app.command("a0")
def evaluate_a0(
    season: Annotated[list[str], typer.Option("--season", help="e.g. 2022-23 (repeatable)")],
) -> None:
    """A0 baseline walk-forward per season; metrics go to the gold leaderboard."""
    from fplh.evaluate.a0 import evaluate_season

    lake = Lake(get_settings().lake_uri)
    for s in season:
        r = evaluate_season(lake, s)
        typer.echo(f"== {s}  (runs: {r.run_ids})")
        for level, metrics in (("player", r.player_metrics), ("match", r.match_metrics)):
            typer.echo(f"  {level}: " + ", ".join(f"{k}={v:.4f}" for k, v in metrics.items()))


@evaluate_app.command("leakage")
def evaluate_leakage(
    season: Annotated[str, typer.Option(help="Silver season label, e.g. 2024-25")],
    every: Annotated[int, typer.Option(help="Check every n-th gameweek deadline.")] = 5,
) -> None:
    """Run the leakage checks (max_observed_at ≤ D, future-shuffle) on real silver."""
    from fplh.features.information_set import SilverStore
    from fplh.features.leakage import check_leakage
    from fplh.features.spine import historical_deadlines

    store = SilverStore(Lake(get_settings().lake_uri))
    tables = ("dim_fixture", "fact_player_match", "snap_fpl_player", "snap_odds")
    frames = {t: store.get(t) for t in tables}
    deadlines = list(historical_deadlines(frames["dim_fixture"], season)["deadline_at"])[::every]
    problems = check_leakage(frames, deadlines)
    for p in problems:
        typer.echo(p, err=True)
    typer.echo(f"{len(deadlines)} deadlines checked, {len(problems)} problems")
    if problems:
        raise typer.Exit(1)


@app.command("version")
def version() -> None:
    from fplh import __version__

    typer.echo(json.dumps({"fplh": __version__}))


if __name__ == "__main__":  # pragma: no cover
    app()
