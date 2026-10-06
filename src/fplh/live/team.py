"""The live model team: model v2 playing the current FPL season (Phase 6a).

A fresh team starts at the first deadline it sees with £100m and unlimited transfers (as
the season replay does) and then plays by the rules: at each deadline
``advise`` forecasts the next five gameweeks with model v2 (the news-aware simulator over
the fused team rates at that deadline), solves the week with the same optimiser as the
replay (``optimize.step``) at the live prices, and stores the picks. After the gameweek is
final, ``score`` scores the stored picks with the official rules (auto-subs, captaincy,
chips) and records FPL's average and highest manager scores for comparison.

State lives in the lake (``state/model_team.json``), so a scheduled runner picks it up,
and each gameweek is decided once: a second ``advise`` for the same gameweek is a no-op.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any

import pandas as pd

from fplh.evaluate.phase3 import run, simulator
from fplh.features.information_set import SilverStore
from fplh.lake.storage import Lake
from fplh.optimize.milp import SquadRules, State
from fplh.optimize.step import commit, decide, players_frame
from fplh.rules.config import load_rules
from fplh.rules.team import score_gameweek

STATE_KEY = "state/model_team.json"
HORIZON = 5
N_SIMS = 1000
ADVISE_WITHIN = pd.Timedelta(hours=30)
LAST_GW = 38


@dataclass
class LiveTeam:
    season: str
    state: State
    weeks: dict[str, dict[str, Any]] = field(default_factory=dict)  # gameweek → record

    def to_json(self) -> bytes:
        doc = {"season": self.season, "state": asdict(self.state), "weeks": self.weeks}
        return json.dumps(doc, indent=1, sort_keys=True, default=str).encode()

    @classmethod
    def from_json(cls, raw: bytes) -> LiveTeam:
        doc = json.loads(raw)
        s = doc["state"]
        state = State(
            int(s["gameweek"]),
            {str(k): int(v) for k, v in s["squad"].items()},
            int(s["bank"]),
            int(s["free_transfers"]),
            {str(k): [int(g) for g in v] for k, v in s["chips_used"].items()},
        )
        return cls(str(doc["season"]), state, dict(doc["weeks"]))


def load(lake: Lake, season: str) -> LiveTeam | None:
    if not lake.exists(STATE_KEY):
        return None
    team = LiveTeam.from_json(lake.get_bytes(STATE_KEY))
    return team if team.season == season else None


def save(lake: Lake, team: LiveTeam) -> None:
    lake.put_bytes(STATE_KEY, team.to_json(), overwrite=True)


def events(store: SilverStore, season: str) -> pd.DataFrame:
    """The latest capture of each gameweek: deadline, finished, data_checked, scores."""
    ev = store.get("fpl_event")
    ev = ev[ev["season"] == season].sort_values("observed_at")
    out: pd.DataFrame = ev.drop_duplicates("gameweek", keep="last").set_index("gameweek")
    return out.sort_index()


def next_deadline(store: SilverStore, season: str, now: pd.Timestamp) -> tuple[int, pd.Timestamp]:
    ev = events(store, season)
    upcoming = ev[ev["deadline_at"] > now]
    if upcoming.empty:
        raise LookupError(f"no deadline after {now} in {season}")
    gw = int(upcoming.index[0])
    return gw, pd.Timestamp(str(upcoming.loc[gw, "deadline_at"]))


def market_state(store: SilverStore, season: str, at: pd.Timestamp) -> pd.DataFrame:
    """Per player at ``at``: position, team, price (£0.1m) from the latest bootstrap
    capture at or before it."""
    snap = store.get("snap_fpl_player")
    snap = snap[(snap["season"] == season) & (snap["observed_at"] <= at)]
    if snap.empty:
        raise LookupError(f"no FPL snapshot before {at}")
    last = snap[snap["observed_at"] == snap["observed_at"].max()]
    out = pd.DataFrame(
        {
            "position": last["position"].to_numpy(),
            "team": last["team"].to_numpy(),
            "price": pd.to_numeric(last["now_cost"]).to_numpy(),
            "status": last["status"].to_numpy() if "status" in last else "a",
        },
        index=pd.Index("fpl:" + last["code"].astype("int64").astype(str), name="player_uid"),
    )
    return out


def forecast(lake: Lake, store: SilverStore, deadline: pd.Timestamp) -> pd.DataFrame:
    """Model v2 at one deadline: per player-fixture expected points for five gameweeks."""
    sim = simulator(lake, store, [deadline], N_SIMS, HORIZON, minutes="news")
    out: pd.DataFrame = run(lake, store, sim, [deadline], HORIZON)
    return out


def expected_by_gameweek(
    pred: pd.DataFrame, store: SilverStore, horizon: list[int]
) -> pd.DataFrame:
    """index player_uid, columns E{g}: expected points summed over a gameweek's fixtures."""
    rounds = store.get("dim_fixture").set_index("fixture_uid")["round"]
    p = pred.assign(gw=pred["fixture_uid"].map(rounds)).dropna(subset=["gw", "expected_points"])
    p = p[p["gw"].astype(int).isin(horizon)]
    e = p.groupby(["player_uid", p["gw"].astype(int)])["expected_points"].sum().unstack()
    e = e.reindex(columns=horizon).fillna(0.0)
    e.columns = [f"E{g}" for g in horizon]
    return e


def advise(
    lake: Lake,
    store: SilverStore,
    season: str,
    now: pd.Timestamp,
    *,
    force: bool = False,
    dry_run: bool = False,
) -> dict[str, Any] | None:
    """Decide the next gameweek if its deadline is within ``ADVISE_WITHIN`` (or
    ``force``) and it has not been decided yet; returns that week's record."""
    gw, deadline = next_deadline(store, season, now)
    team = load(lake, season) or LiveTeam(season, State(gw, {}, 1000, 15, {}))
    if str(gw) in team.weeks and team.weeks[str(gw)].get("advised"):
        return None
    if not force and deadline - now > ADVISE_WITHIN:
        return None
    rules = SquadRules.from_rules(load_rules(season.replace("-", "/")))
    horizon = list(range(gw, min(gw + HORIZON, LAST_GW + 1)))
    pred = forecast(lake, store, deadline)
    expected = expected_by_gameweek(pred, store, horizon)
    market = market_state(store, season, now)
    players = players_frame(
        expected, market["position"], market["team"], market["price"], team.state
    )
    decision = decide(players, team.state, rules, horizon, market["price"])
    plan = decision.plan
    record: dict[str, Any] = {
        "advised": now.isoformat(),
        "deadline": deadline.isoformat(),
        "squad": plan.squad,
        "xi": plan.xi,
        "bench": plan.bench,
        "captain": plan.captain,
        "vice": plan.vice,
        "chip": plan.chip,
        "buys": plan.buys,
        "sells": plan.sells,
        "hits": plan.hits,
        "expected_points": round(float(plan.expected_points), 2),
        "bank": decision.bank,
        "free_transfers_before": team.state.free_transfers,
        "expected": {
            p: round(float(str(expected.at[p, expected.columns[0]])), 2)
            if p in expected.index
            else 0.0
            for p in plan.squad
        },
    }
    team.weeks[str(gw)] = record
    commit(team.state, decision, gw)
    team.state.gameweek = gw + 1
    if not dry_run:
        save(lake, team)
    return {"gameweek": gw, **record}


def score(lake: Lake, store: SilverStore, season: str) -> list[int]:
    """Score every decided gameweek that FPL has finalised and that is not scored yet."""
    team = load(lake, season)
    if team is None:
        return []
    ev = events(store, season)
    pm = store.get("fact_player_match")
    pm = pm[pm["season"] == season]
    rules = load_rules(season.replace("-", "/"))
    xi_min = {str(k): int(v) for k, v in rules.game.xi_min.items()}
    hit_cost = SquadRules.from_rules(rules).hit_cost
    done = []
    for key, week in sorted(team.weeks.items(), key=lambda kv: int(kv[0])):
        gw = int(key)
        if "points" in week or gw not in ev.index or not bool(ev.loc[gw, "data_checked"]):
            continue
        rows = pm[pm["round"] == gw]
        if rows.empty:
            continue
        pts = rows.groupby("player_uid")["total_points"].sum()
        mins = rows.groupby("player_uid")["minutes"].sum()
        picks = [*week["xi"], *week["bench"]]
        position = rows.drop_duplicates("player_uid").set_index("player_uid")["position"]
        pos = {p: str(position.get(p, "MID")) for p in picks}
        s = score_gameweek(
            picks,
            pos,
            {p: int(pts.get(p, 0)) for p in picks},
            {p: int(mins.get(p, 0)) for p in picks},
            week["captain"],
            week["vice"],
            week["chip"],
            xi_min,
        )
        week["gross"] = int(s.points)
        week["points"] = int(s.points) - hit_cost * int(week["hits"])
        week["substitutions"] = [list(x) for x in s.substitutions]
        week["average"] = _num(ev.loc[gw, "average_entry_score"])
        week["highest"] = _num(ev.loc[gw, "highest_score"])
        done.append(gw)
    if done:
        save(lake, team)
    return done


def _num(x: Any) -> int | None:
    return None if x is None or pd.isna(x) else int(float(str(x)))


def season_table(team: LiveTeam) -> pd.DataFrame:
    """One row per scored gameweek: points, average, highest, and running totals."""
    rows = [
        {"gw": int(k), **{c: w.get(c) for c in ("points", "average", "highest", "hits", "chip")}}
        for k, w in team.weeks.items()
        if "points" in w
    ]
    if not rows:
        return pd.DataFrame(columns=["gw", "points", "average", "highest", "hits", "chip"])
    t = pd.DataFrame(rows).sort_values("gw").reset_index(drop=True)
    t["total"] = t["points"].cumsum()
    t["average_total"] = t["average"].fillna(0).cumsum()
    return t
