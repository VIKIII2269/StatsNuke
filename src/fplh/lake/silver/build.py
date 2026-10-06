"""Build every silver table from bronze, resolve entities, validate, gate, then write.

Nothing is written unless every blocking quality gate passes (ARCHITECTURE.md §6.6).
Silver is fully rebuildable: the same bronze always produces byte-identical Parquet.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from fplh.entities.fixtures import build_dim_fixture, with_fixture_uid
from fplh.entities.players import build_dim_player, link_players, load_overrides
from fplh.entities.teams import TeamResolver
from fplh.lake.bronze import latest_by_params, read_bronze
from fplh.lake.parquet import to_parquet_bytes, write_parquet
from fplh.lake.quality import CheckResult, run_gates
from fplh.lake.silver import football_data as fd
from fplh.lake.silver import fpl_live, odds_api
from fplh.lake.silver import understat as us
from fplh.lake.silver import vaastav as va
from fplh.lake.silver.common import latest_payload, season_label
from fplh.lake.silver.football_data import prematch_observed_at as fd_prematch
from fplh.lake.silver.schemas import validate
from fplh.lake.silver.timeline import derive_lineups, match_events
from fplh.lake.storage import Lake
from fplh.sources import load_sources

# Sort keys make the Parquet output deterministic.
SORT_KEYS: dict[str, list[str]] = {
    "fact_player_match": ["season", "fpl_fixture_id", "element"],
    "fpl_fixture": ["season", "fpl_fixture_id"],
    "fpl_player_season": ["season", "element"],
    "fd_match": ["season", "division", "kickoff_at", "home_team"],
    "snap_odds": [
        "season",
        "division",
        "kickoff_at",
        "home_team",
        "bookmaker",
        "market",
        "is_closing",
        "outcome",
        "observed_at",
    ],
    "us_match": ["season", "understat_match_id"],
    "us_team_match": ["season", "team", "kickoff_at"],
    "us_player_season": ["season", "understat_player_id"],
    "fact_shot": ["season", "understat_match_id", "shot_id"],
    "fact_player_match_understat": ["season", "understat_match_id", "understat_player_id"],
    "fact_match_event": [
        "season",
        "understat_match_id",
        "minute",
        "kind",
        "seq",
        "side",
        "understat_player_id",
    ],
    "snap_fpl_player": ["season", "observed_at", "element"],
    "fpl_event": ["season", "observed_at", "gameweek"],
    "dim_fixture": ["fixture_uid"],
    "dim_team": ["team_uid"],
    "dim_player": ["player_uid"],
    "player_link": ["season", "team", "code"],
}
DIMENSIONS = {"dim_fixture", "dim_team", "dim_player", "player_link"}


@dataclass
class BuildReport:
    rows: dict[str, int] = field(default_factory=dict)
    notes: dict[str, Any] = field(default_factory=dict)
    gates: list[CheckResult] = field(default_factory=list)
    written: list[str] = field(default_factory=list)

    def summary(self) -> str:
        lines = [f"{t}: {n} rows" for t, n in sorted(self.rows.items())]
        lines += [g.line() for g in self.gates]
        return "\n".join(lines)


def _season_from_code(code: str) -> int:
    yy = int(code[:2])
    return (2000 if yy < 50 else 1900) + yy


def _concat(frames: list[pd.DataFrame]) -> pd.DataFrame:
    frames = [f for f in frames if f is not None and not f.empty]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def assemble(
    lake: Lake, seasons: set[str] | None = None
) -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    """Normalise every source in bronze into in-memory silver tables."""
    lags = {name: cfg.publication_lag_hours for name, cfg in load_sources().sources.items()}
    teams = TeamResolver.from_config()
    notes: dict[str, Any] = {}

    def keep(season: str) -> bool:
        return seasons is None or season in seasons

    # football-data first: its names are canonical team ids.
    fd_matches, odds = [], []
    for division in ("E0", "E1"):
        for params, key in sorted(latest_by_params(lake, fd.SOURCE, division).items()):
            year = _season_from_code(dict(params)["season"])
            if not keep(season_label(year)):
                continue
            payload = latest_payload(lake, fd.SOURCE, division, season=dict(params)["season"])
            assert payload is not None
            m, o, n = fd.normalise_file(
                payload[1],
                start_year=year,
                division=division,
                teams=teams,
                bronze_key=key,
                result_lag_hours=lags["football_data"],
            )
            notes[f"football_data/{division}/{season_label(year)}"] = n
            fd_matches.append(m)
            odds.append(o)

    fpl_fx, fpl_pm, fpl_players = [], [], []
    for params in sorted(latest_by_params(lake, va.SOURCE, "merged_gw")):
        year = int(dict(params)["season"][:4])
        if not keep(season_label(year)):
            continue
        s = va.load_season(lake, year, teams, lags["vaastav"])
        notes[f"vaastav/{s.season}"] = s.notes
        fpl_fx.append(s.fixtures)
        fpl_pm.append(s.player_match)
        fpl_players.append(s.players)
    live = fpl_live.live_season(lake, teams)
    if live is not None and keep(live.season):
        # Our own post-GW pulls replace vaastav for the season they cover.
        notes[f"fpl_live/{live.season}"] = live.notes
        fpl_fx = [f for f in fpl_fx if f["season"].iloc[0] != live.season] + [live.fixtures]
        fpl_pm = [f for f in fpl_pm if f["season"].iloc[0] != live.season] + [live.player_match]
        fpl_players = [f for f in fpl_players if f["season"].iloc[0] != live.season] + [
            live.players
        ]

    us_tables: dict[str, list[pd.DataFrame]] = {}
    matches_by_id: dict[int, pd.Series] = {}
    for params, key in sorted(latest_by_params(lake, us.SOURCE, "league").items()):
        year = int(dict(params)["season"])
        if not keep(season_label(year)):
            continue
        payload = latest_payload(lake, us.SOURCE, "league", season=str(year))
        assert payload is not None
        out = us.normalise_league(
            payload[1], year, teams, bronze_key=key, lag_hours=lags["understat"]
        )
        for name, df in out.items():
            us_tables.setdefault(name, []).append(df)
        for _, row in out["us_match"].iterrows():
            matches_by_id[int(row["understat_match_id"])] = row
    orphan = 0
    for params, key in sorted(latest_by_params(lake, us.SOURCE, "match").items()):
        mid = int(dict(params)["match"])
        match_row = matches_by_id.get(mid)
        if match_row is None:
            orphan += 1
            continue
        out = us.normalise_match(
            read_bronze(lake, key)[1], match_row, bronze_key=key, lag_hours=lags["understat"]
        )
        for name, df in out.items():
            us_tables.setdefault(name, []).append(df)
    notes["understat/orphan_match_payloads"] = orphan

    snap, events = fpl_live.snapshots(lake, teams)
    odds.append(odds_api.normalise(lake, teams))

    t: dict[str, pd.DataFrame] = {
        "fd_match": _concat(fd_matches),
        "snap_odds": _concat(odds),
        "fpl_fixture": _concat(fpl_fx),
        "fact_player_match": _concat(fpl_pm),
        "fpl_player_season": _concat(fpl_players),
        "snap_fpl_player": snap,
        "fpl_event": events,
        "snap_props": odds_api.normalise_props(lake, teams),
        **{name: _concat(frames) for name, frames in us_tables.items()},
    }
    return _resolve(t, teams, notes), notes


def _resolve(
    t: dict[str, pd.DataFrame], teams: TeamResolver, notes: dict[str, Any]
) -> dict[str, pd.DataFrame]:
    for name in ("fd_match", "snap_odds", "fpl_fixture", "us_match"):
        if not t.get(name, pd.DataFrame()).empty:
            t[name] = with_fixture_uid(t[name])
    pm = t["fact_player_match"]
    if not pm.empty:
        home = pm["team"].where(pm["was_home"], pm["opponent"])
        away = pm["opponent"].where(pm["was_home"], pm["team"])
        pm["fixture_uid"] = pm["season"] + ":" + home + ":" + away
        pm["player_uid"] = "fpl:" + pm["code"].astype(str)
    us_match = t.get("us_match", pd.DataFrame())
    if not us_match.empty:
        uid_by_mid = us_match.set_index("understat_match_id")["fixture_uid"]
        for name in ("fact_shot", "fact_player_match_understat"):
            if not t.get(name, pd.DataFrame()).empty:
                t[name]["fixture_uid"] = t[name]["understat_match_id"].map(uid_by_mid)
        _align_understat_times(t, notes)

    t["dim_fixture"] = build_dim_fixture(t["fpl_fixture"], t["fd_match"], us_match)
    _align_imputed_kickoffs(t, notes)
    all_teams = set()
    for name in ("fd_match", "fpl_fixture", "us_match"):
        df = t.get(name, pd.DataFrame())
        if not df.empty:
            all_teams |= set(df["home_team"]) | set(df["away_team"])
    t["dim_team"] = pd.DataFrame({"team_uid": sorted(all_teams)})

    roster = t.get("fact_player_match_understat", pd.DataFrame())
    fpl_players = t["fpl_player_season"]
    if not pm.empty:
        fpl_apps = pm[["season", "team", "code", "fixture_uid", "minutes"]]
        if not roster.empty:
            us_apps = roster[
                ["season", "team", "understat_player_id", "player_name", "fixture_uid", "minutes"]
            ]
        else:
            us_apps = pd.DataFrame(
                columns=[
                    "season",
                    "team",
                    "understat_player_id",
                    "player_name",
                    "fixture_uid",
                    "minutes",
                ]
            )
        result = link_players(fpl_apps, us_apps, fpl_players, load_overrides())
        t["player_link"] = result.links
        t["entity_review_queue"] = result.review
        t["entity_coverage"] = result.coverage
        t["dim_player"] = build_dim_player(fpl_players, result.links)
        notes["entity_link_methods"] = result.links["method"].value_counts().to_dict()
        if not roster.empty:
            code_by = result.links.set_index(["season", "team", "understat_player_id"])["code"]
            for name in ("fact_player_match_understat", "fact_shot"):
                df = t.get(name, pd.DataFrame())
                if df.empty:
                    continue
                keys = pd.MultiIndex.from_frame(df[["season", "team", "understat_player_id"]])
                codes = code_by.reindex(keys).to_numpy()
                df["player_uid"] = pd.Series(codes, index=df.index).map(
                    lambda c: f"fpl:{int(c)}" if pd.notna(c) else pd.NA
                )
    _timeline(t, notes)
    return t


def _timeline(t: dict[str, pd.DataFrame], notes: dict[str, Any]) -> None:
    """Lineup columns on Understat roster rows, shot assisters, and ``fact_match_event``."""
    roster, shots = t.get("fact_player_match_understat"), t.get("fact_shot")
    if roster is None or roster.empty or "roster_id" not in roster:
        return
    for df in (roster, shots):
        if df is not None and not df.empty and "player_uid" not in df:
            df["player_uid"] = pd.NA  # no FPL data to link against
    lineups, lineup_notes = derive_lineups(roster)
    notes.update({f"timeline/{k}": v for k, v in lineup_notes.items()})
    t["fact_player_match_understat"] = lineups
    if shots is not None and not shots.empty:
        names = lineups.drop_duplicates(["understat_match_id", "side", "player_name"]).set_index(
            ["understat_match_id", "side", "player_name"]
        )
        keys = pd.MultiIndex.from_frame(shots[["understat_match_id", "side", "assisted_by"]])
        found = names.reindex(keys)
        shots["assister_understat_id"] = found["understat_player_id"].astype("Int64").to_numpy()
        shots["assister_uid"] = found["player_uid"].to_numpy()
        named = shots["assisted_by"].notna()
        notes["timeline/assisters_unmatched"] = int(
            (named & shots["assister_understat_id"].isna()).sum()
        )
        notes["timeline/assisters_named"] = int(named.sum())
        t["fact_match_event"] = match_events(lineups, shots, t["us_match"])


def _align_understat_times(t: dict[str, pd.DataFrame], notes: dict[str, Any]) -> None:
    """Understat timestamps are UTC but can lag FPL's after a reschedule (±1–4 h in 18 %
    of 2016/17+ matches). For fixtures FPL knows, take event time from FPL's kickoff and
    keep Understat's publication lag, so every source agrees on when a match happened."""
    fpl = t.get("fpl_fixture", pd.DataFrame())
    us = t["us_match"]
    if fpl.empty:
        return
    kickoff = fpl.set_index("fixture_uid")["kickoff_at"]
    lag = us["observed_at"] - us["event_at"]
    fpl_time = us["fixture_uid"].map(kickoff)
    moved = fpl_time.notna() & (fpl_time != us["event_at"])
    notes["understat/event_time_aligned_to_fpl"] = int(moved.sum())
    us["event_at"] = fpl_time.fillna(us["event_at"])
    us["observed_at"] = us["event_at"] + lag
    by_mid = us.set_index("understat_match_id")[["event_at", "observed_at"]]
    for name in ("fact_shot", "fact_player_match_understat"):
        df = t.get(name, pd.DataFrame())
        if not df.empty:
            df["event_at"] = df["understat_match_id"].map(by_mid["event_at"])
            df["observed_at"] = df["understat_match_id"].map(by_mid["observed_at"])


def _align_imputed_kickoffs(t: dict[str, pd.DataFrame], notes: dict[str, Any]) -> None:
    """football-data rows before 2019/20 have no kickoff time (15:00 UK is imputed). When
    FPL or Understat knows the real kickoff, use it for the match and its odds, and
    recompute observation times (results: + lag; pre-match prices: collection rule;
    closing prices: kickoff)."""
    matches, odds, dim = t["fd_match"], t["snap_odds"], t["dim_fixture"]
    if matches.empty or dim.empty:
        return
    true_kickoff = dim.set_index("fixture_uid")["kickoff_at"]
    fix = matches["kickoff_time_imputed"].astype(bool) & matches["fixture_uid"].isin(
        true_kickoff.index
    )
    known = matches["fixture_uid"].map(true_kickoff)
    fix &= known.notna() & (known != matches["kickoff_at"])
    notes["football_data/imputed_kickoffs_aligned"] = int(fix.sum())
    if not fix.any():
        return
    lag = matches.loc[fix, "observed_at"] - matches.loc[fix, "event_at"]
    matches.loc[fix, "kickoff_at"] = known[fix]
    matches.loc[fix, "event_at"] = known[fix]
    matches.loc[fix, "observed_at"] = known[fix] + lag
    if odds.empty:
        return
    moved = odds["fixture_uid"].isin(set(matches.loc[fix, "fixture_uid"])) & (
        odds["source"] == fd.SOURCE
    )
    new_k = odds.loc[moved, "fixture_uid"].map(true_kickoff)
    odds.loc[moved, "kickoff_at"] = new_k
    odds.loc[moved, "observed_at"] = [
        k if closing else fd_prematch(k)
        for k, closing in zip(new_k, odds.loc[moved, "is_closing"], strict=True)
    ]


REPORT_TABLES = {"entity_review_queue", "entity_coverage"}


def build(
    lake: Lake, *, seasons: set[str] | None = None, write: bool = True, raise_on_fail: bool = True
) -> tuple[BuildReport, dict[str, pd.DataFrame]]:
    tables, notes = assemble(lake, seasons)
    report = BuildReport(notes=notes)
    for name, df in tables.items():
        tables[name] = validate(name, df)
        report.rows[name] = len(df)
    report.gates = run_gates(tables, raise_on_fail=raise_on_fail)
    if not write:
        return report, tables
    for name, df in sorted(tables.items()):
        if df.empty:
            continue
        if name in REPORT_TABLES:
            key = f"gold/reports/{name}.parquet"
            report.written.append(write_parquet(lake, key, df, list(df.columns[:3])))
        elif name in DIMENSIONS or "season" not in df.columns:
            key = f"silver/{name}/part-000.parquet"
            report.written.append(write_parquet(lake, key, df, SORT_KEYS.get(name, [])))
        else:
            for season, part in df.groupby("season"):
                key = f"silver/{name}/season={season}/part-000.parquet"
                report.written.append(write_parquet(lake, key, part, SORT_KEYS.get(name, [])))
    gate_rows = [
        {"check": g.name, "blocking": g.blocking, "passed": g.passed, "details": g.details}
        for g in report.gates
    ]
    lake.put_bytes(
        "gold/reports/silver_build.json",
        json.dumps(
            {"rows": report.rows, "notes": notes, "gates": gate_rows},
            indent=2,
            default=str,
            sort_keys=True,
        ).encode(),
        overwrite=True,
    )
    return report, tables


__all__ = ["BuildReport", "assemble", "build", "to_parquet_bytes"]
