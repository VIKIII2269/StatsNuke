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
from fplh.lake.silver.schemas import validate
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

    t["dim_fixture"] = build_dim_fixture(t["fpl_fixture"], t["fd_match"], us_match)
    all_teams = set()
    for name in ("fd_match", "fpl_fixture", "us_match"):
        df = t.get(name, pd.DataFrame())
        if not df.empty:
            all_teams |= set(df["home_team"]) | set(df["away_team"])
    t["dim_team"] = pd.DataFrame({"team_uid": sorted(all_teams)})

    roster = t.get("fact_player_match_understat", pd.DataFrame())
    fpl_players = t["fpl_player_season"]
    if not pm.empty:
        fpl_minutes = pm.groupby(["season", "team", "code"])["minutes"].sum().reset_index()
        if not roster.empty:
            us_minutes = roster.groupby(
                ["season", "team", "understat_player_id"], as_index=False
            ).agg(player_name=("player_name", "first"), minutes=("minutes", "sum"))
        else:
            us_minutes = pd.DataFrame(
                columns=["season", "team", "understat_player_id", "player_name", "minutes"]
            )
        result = link_players(fpl_minutes, us_minutes, fpl_players, load_overrides())
        t["player_link"] = result.links
        t["entity_review_queue"] = result.review
        t["entity_coverage"] = result.coverage
        t["dim_player"] = build_dim_player(fpl_players, result.links)
        notes["entity_link_methods"] = result.links["method"].value_counts().to_dict()
        if not roster.empty:
            code_by = result.links.set_index(["season", "team", "understat_player_id"])["code"]
            keys = pd.MultiIndex.from_frame(roster[["season", "team", "understat_player_id"]])
            codes = code_by.reindex(keys).to_numpy()
            roster["player_uid"] = pd.Series(codes, index=roster.index).map(
                lambda c: f"fpl:{int(c)}" if pd.notna(c) else pd.NA
            )
    return t


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
