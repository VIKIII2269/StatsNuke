"""Our own FPL captures → ``snap_fpl_player``, ``fpl_event`` and (live season)
``fact_player_match`` from ``element-summary`` pulls.

Snapshot facts are observed at fetch time. Post-gameweek rows are also stamped with
their pull's fetch time, which is conservative (never earlier than availability).
"""

from __future__ import annotations

import json

import pandas as pd

from fplh.entities.teams import TeamResolver
from fplh.lake.bronze import list_bronze, parse_key, read_bronze
from fplh.lake.silver.common import POSITION_BY_ELEMENT_TYPE, season_label
from fplh.lake.silver.vaastav import VaastavSeason, normalise_season
from fplh.lake.storage import Lake

SOURCE = "fpl"
SNAPSHOT_FIELDS = (
    "now_cost",
    "status",
    "chance_of_playing_next_round",
    "chance_of_playing_this_round",
    "news",
    "news_added",
    "selected_by_percent",
    "ep_next",
    "ep_this",
    "form",
    "transfers_in_event",
    "transfers_out_event",
)


def _season_of(bootstrap: dict[str, object]) -> int:
    events = bootstrap["events"]
    assert isinstance(events, list)
    first = min(pd.Timestamp(e["deadline_time"]) for e in events)
    return first.year if first.month >= 7 else first.year - 1


def snapshots(lake: Lake, teams: TeamResolver) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(snap_fpl_player, fpl_event) from every captured bootstrap-static."""
    players, events = [], []
    for key in list_bronze(lake, SOURCE, "bootstrap-static"):
        meta, payload = read_bronze(lake, key)
        if meta["http_status"] != 200:
            continue
        boot = json.loads(payload)
        obs = pd.Timestamp(meta["observed_at_us"])
        season = season_label(_season_of(boot))
        team_uid = {t["id"]: teams.uid(t["name"]) for t in boot["teams"]}
        for e in boot["elements"]:
            pos = POSITION_BY_ELEMENT_TYPE.get(e["element_type"])
            if pos is None:
                continue
            players.append(
                {
                    "season": season,
                    "element": e["id"],
                    "code": e["code"],
                    "position": pos,
                    "team": team_uid[e["team"]],
                    **{f: e.get(f) for f in SNAPSHOT_FIELDS},
                    "observed_at": obs,
                    "source": SOURCE,
                    "bronze_key": key,
                }
            )
        for ev in boot["events"]:
            events.append(
                {
                    "season": season,
                    "gameweek": ev["id"],
                    "deadline_at": pd.Timestamp(ev["deadline_time"]),
                    "finished": bool(ev.get("finished")),
                    "data_checked": bool(ev.get("data_checked")),
                    # the average and highest manager scores, once the gameweek has them
                    "average_entry_score": ev.get("average_entry_score"),
                    "highest_score": ev.get("highest_score"),
                    "observed_at": obs,
                    "source": SOURCE,
                }
            )
    snap = pd.DataFrame(players)
    if not snap.empty:
        for c in ("selected_by_percent", "ep_next", "ep_this", "form"):
            snap[c] = pd.to_numeric(snap[c], errors="coerce")
    return snap, pd.DataFrame(events)


def live_season(lake: Lake, teams: TeamResolver) -> VaastavSeason | None:
    """``fact_player_match`` for the season of the latest post-GW pull, if any."""
    boots = list_bronze(lake, SOURCE, "bootstrap-static")
    summaries = list_bronze(lake, SOURCE, "element-summary")
    if not boots or not summaries:
        return None
    latest_pull = parse_key(summaries[-1]).observed_at
    # The bootstrap captured closest before the latest pull describes its elements.
    boot_key = max((k for k in boots if parse_key(k).observed_at <= latest_pull), default=boots[0])
    boot = json.loads(read_bronze(lake, boot_key)[1])
    start_year = _season_of(boot)

    rows = []
    fetched = {}
    for key in summaries:
        pk = parse_key(key)
        if pk.observed_at.date() < latest_pull.date():
            continue  # only the latest pull (history rows repeat the whole season)
        data = json.loads(read_bronze(lake, key)[1])
        for h in data.get("history", []):
            rows.append(h)
            fetched[(h["element"], h["fixture"])] = pk.observed_at
    if not rows:
        return None
    gw = pd.DataFrame(rows)
    raw = pd.DataFrame(boot["elements"])
    season_teams = pd.DataFrame(boot["teams"])[["id", "name"]]
    out = normalise_season(
        start_year,
        gw,
        raw,
        pd.DataFrame(columns=["season", "team", "team_name"]),
        teams,
        season_teams=season_teams,
        bronze_key=summaries[-1],
    )
    out.player_match["observed_at"] = [
        pd.Timestamp(fetched[(e, f)])
        for e, f in zip(
            out.player_match["element"], out.player_match["fpl_fixture_id"], strict=True
        )
    ]
    out.player_match["source"] = SOURCE
    out.fixtures["source"] = SOURCE
    return out
