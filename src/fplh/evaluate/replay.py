"""Season replay (ticket 4.2): every forecaster through the same optimiser (ARCHITECTURE.md
§11.3, the decision-level benchmark that isolates forecast quality).

For each strategy and season, from gameweek 2 (gameweek 1 has no deadline spine, so every
strategy starts with a free squad of £100m at the GW2 deadline):

1. expected points E[p, g] for the next ``horizon`` gameweeks, from the strategy's
   forecasts made at the deadline:
   * ``native``: the forecaster's own per-fixture forecasts at each horizon (the
     simulator run at horizon 5), summed over the player's fixtures in each gameweek;
   * ``repeat``: a horizon-1 forecaster's per-fixture forecast for next week, repeated
     for every fixture of the player's club in each later gameweek (doubles count twice,
     blanks zero);
2. ``optimize.milp.plan_week`` with the season's rules, at the deadline's prices;
3. transfers at those prices (sell prices by the FPL formula), then the gameweek scored
   with the official auto-substitution rules on the players' actual points (``rules.team``)
   less 4 points per hit;
4. bank, free transfers and chips carried forward; a free-hit squad reverts afterwards.

Prices and the schedule are public at the deadline; actual points are used only to score
the week after the decision.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import pandas as pd

from fplh.evaluate.bootstrap import compare
from fplh.features.information_set import SilverStore
from fplh.features.prices import price_table, sell_price
from fplh.features.spine import historical_deadlines
from fplh.lake.storage import Lake
from fplh.optimize.milp import SquadRules, State, plan_week
from fplh.rules.config import load_rules
from fplh.rules.team import score_gameweek

Mode = Literal["native", "repeat"]
LAST_GW = 38


@dataclass
class SeasonData:
    """What a replay needs for one season, built once and shared by every strategy."""

    season: str
    deadlines: dict[int, pd.Timestamp]  # gameweek → deadline
    gw_of_fixture: dict[str, int]
    fixtures_per_team: pd.DataFrame  # index team, columns gameweek: fixture count
    prices: pd.DataFrame  # index player_uid, columns gameweek
    team_at: pd.DataFrame  # player_uid, gw, team (the club at that gameweek's deadline)
    position: pd.Series  # player_uid → position
    points: pd.DataFrame  # index player_uid, columns gameweek: actual FPL points
    minutes: pd.DataFrame  # index player_uid, columns gameweek: minutes


def season_data(store: SilverStore, season: str) -> SeasonData:
    dim = store.get("dim_fixture")
    fx = dim[(dim["season"] == season) & dim["round"].notna()]
    gw_of = {str(u): int(r) for u, r in zip(fx["fixture_uid"], fx["round"], strict=True)}
    sides = pd.concat(
        [
            fx[["home_team", "round"]].set_axis(["team", "gw"], axis=1),
            fx[["away_team", "round"]].set_axis(["team", "gw"], axis=1),
        ]
    )
    counts = (
        sides.groupby(["team", "gw"])
        .size()
        .unstack(fill_value=0)
        .reindex(columns=range(1, LAST_GW + 1), fill_value=0)
    )
    deadlines = {
        int(r): pd.Timestamp(d)
        for r, d in historical_deadlines(dim, season).itertuples(index=False)
    }
    pm = store.get("fact_player_match")
    pm = pm[(pm["season"] == season) & pm["player_uid"].notna()]
    pts = pm.groupby(["player_uid", "round"])["total_points"].sum().unstack(fill_value=0)
    mins = pm.groupby(["player_uid", "round"])["minutes"].sum().unstack(fill_value=0)
    cols = range(1, LAST_GW + 1)
    # the club a player is at for each gameweek: his latest row up to it
    team = (
        pm.sort_values("kickoff_at")
        .drop_duplicates(["player_uid", "round"], keep="last")
        .pivot(index="player_uid", columns="round", values="team")
        .reindex(columns=cols)
        .ffill(axis=1)
        .bfill(axis=1)
    )
    team_at = (
        team.reset_index()
        .melt(id_vars="player_uid", var_name="gw", value_name="team")
        .dropna(subset=["team"])
    )
    team_at["gw"] = team_at["gw"].astype(int)
    position = (
        pm.sort_values("kickoff_at")
        .drop_duplicates("player_uid", keep="last")
        .set_index("player_uid")["position"]
    )
    return SeasonData(
        season,
        deadlines,
        gw_of,
        counts,
        price_table(store, season),
        team_at,
        position,
        pts.reindex(columns=cols, fill_value=0),
        mins.reindex(columns=cols, fill_value=0),
    )


def expected_points(
    pred: pd.DataFrame, data: SeasonData, gw: int, horizon: list[int], mode: Mode
) -> pd.DataFrame:
    """index player_uid, columns E{g} for g in horizon, from forecasts made at gw's deadline."""
    d = data.deadlines[gw]
    p = pred[pred["deadline_at"] == d]
    p = p.assign(gw=p["fixture_uid"].map(data.gw_of_fixture)).dropna(subset=["gw"])
    p["gw"] = p["gw"].astype(int)
    if mode == "native":
        e = p[p["gw"].isin(horizon)].groupby(["player_uid", "gw"])["expected_points"].sum()
        out = e.unstack().reindex(columns=horizon).fillna(0.0)
    else:
        nxt = p[p["gw"] == gw]
        per_fixture = nxt.groupby("player_uid")["expected_points"].mean()
        teams = data.team_at[data.team_at["gw"] == gw].set_index("player_uid")["team"]
        team = teams.reindex(per_fixture.index)
        counts = data.fixtures_per_team.reindex(team.to_numpy())[horizon].fillna(0).to_numpy()
        out = pd.DataFrame(
            per_fixture.to_numpy()[:, None] * counts, index=per_fixture.index, columns=horizon
        )
    out.columns = [f"E{g}" for g in horizon]
    return out.fillna(0.0)


@dataclass
class ReplayResult:
    strategy: str
    season: str
    log: pd.DataFrame  # one row per gameweek
    total: int = 0
    chips: dict[str, list[int]] = field(default_factory=dict)


def replay_season(
    strategy: str,
    pred: pd.DataFrame,
    mode: Mode,
    data: SeasonData,
    *,
    horizon: int = 5,
    delta: float = 0.9,
    beta: float = 0.1,
    chip_cost: dict[str, float] | None = None,
    time_limit: float = 30.0,
    until: int = LAST_GW,
) -> ReplayResult:
    season_rules = load_rules(data.season.replace("-", "/"))
    rules = SquadRules.from_rules(season_rules)
    xi_min = {str(k): int(v) for k, v in season_rules.game.xi_min.items()}
    state = State(2, {}, 1000, 15, {})
    rows = []
    for gw in range(2, until + 1):
        if gw not in data.deadlines:
            continue
        hz = list(range(gw, min(gw + horizon, LAST_GW + 1)))
        e = expected_points(pred, data, gw, hz, mode)
        price = data.prices[gw]
        teams = data.team_at[data.team_at["gw"] == gw].set_index("player_uid")["team"]
        ids = sorted(set(e.index) | set(state.squad))
        players = pd.DataFrame(index=pd.Index(ids, name="player_uid"))
        players["position"] = data.position.reindex(ids)
        players["team"] = teams.reindex(ids)
        players["price"] = price.reindex(ids)
        players = players.join(e).fillna({c: 0.0 for c in e.columns})
        for c in e.columns:
            players[c] = players[c].fillna(0.0)
        players = players.dropna(subset=["position", "team", "price"])
        players = players[players.index.isin(state.squad) | (players["price"] > 0)]
        state.gameweek = gw
        plan = plan_week(
            players,
            state,
            rules,
            hz,
            delta=delta,
            beta=beta,
            chip_cost=chip_cost,
            time_limit=time_limit,
        )
        # apply transfers at this deadline's prices
        bank = state.bank
        squad = dict(state.squad)
        for p in plan.sells:
            bank += sell_price(squad.pop(p), int(price[p]))
        for p in plan.buys:
            squad[p] = int(price[p])
            bank -= int(price[p])
        if bank < 0:
            raise RuntimeError(f"{strategy} {data.season} GW{gw}: negative bank {bank}")
        picks = plan.xi + plan.bench
        pos = {str(k): str(v) for k, v in data.position.reindex(picks).items()}
        score = score_gameweek(
            picks,
            pos,
            {str(k): int(v) for k, v in data.points[gw].reindex(picks).fillna(0).items()},
            {str(k): int(v) for k, v in data.minutes[gw].reindex(picks).fillna(0).items()},
            plan.captain,
            plan.vice,
            plan.chip,
            xi_min,
        )
        points = score.points - rules.hit_cost * plan.hits
        rows.append(
            {
                "strategy": strategy,
                "season": data.season,
                "gw": gw,
                "points": points,
                "gross": score.points,
                "hits": plan.hits,
                "transfers": len(plan.buys),
                "chip": plan.chip,
                "captain": plan.captain,
                "captain_points": int(data.points[gw].get(plan.captain, 0)),
                "expected": plan.expected_points,
                "bank": bank,
                "squad_value": int(sum(price.get(p, 0) for p in squad)),
                "squad": ",".join(sorted(plan.squad)),
                "xi": ",".join(plan.xi),
                "bench": ",".join(plan.bench),
                "vice": plan.vice,
            }
        )
        if plan.chip:
            state.chips_used.setdefault(plan.chip, []).append(gw)
        if plan.chip != "free_hit":  # a free-hit squad reverts; its transfers are not kept
            state.squad, state.bank = squad, bank
        state.free_transfers = plan.free_transfers_next
    log = pd.DataFrame(rows)
    return ReplayResult(
        strategy, data.season, log, int(log["points"].sum()) if len(log) else 0, state.chips_used
    )


def compare_strategies(logs: pd.DataFrame, a: str, b: str, n_boot: int = 2000) -> dict[str, float]:
    """Points per gameweek of strategy a minus b, gameweek-block bootstrap (positive favours a).
    ``compare`` reports loss differences, so points are negated."""
    la = logs[logs["strategy"] == a].set_index(["season", "gw"])["points"]
    lb = logs[logs["strategy"] == b].set_index(["season", "gw"])["points"]
    common = la.index.intersection(lb.index)
    block = pd.Series([f"{s}:{g}" for s, g in common])
    c = compare(
        pd.Series(-la.loc[common].to_numpy(float)),
        pd.Series(-lb.loc[common].to_numpy(float)),
        block,
        n_boot=n_boot,
    )
    return {
        "per_gw": -c.mean_diff,
        "ci_low": -c.ci_high,
        "ci_high": -c.ci_low,
        "dm_p": c.dm_pvalue,
        "total": float(la.loc[common].sum() - lb.loc[common].sum()),
        "gameweeks": float(len(common)),
    }


def season_totals(logs: pd.DataFrame) -> pd.DataFrame:
    out: pd.DataFrame = logs.pivot_table(
        index="strategy", columns="season", values="points", aggfunc="sum"
    )
    out["total"] = out.sum(axis=1)
    return out.sort_values("total", ascending=False)


def strategy_summary(logs: pd.DataFrame) -> pd.DataFrame:
    g = logs.groupby("strategy")
    out: pd.DataFrame = pd.DataFrame(
        {
            "points": g["points"].sum(),
            "hits": g["hits"].sum(),
            "transfers": g["transfers"].sum(),
            "captain_points_per_gw": g["captain_points"].mean(),
            "chips": g["chip"].apply(lambda c: ", ".join(sorted(c.dropna().astype(str)))),
            "mean_expected": g["expected"].mean(),
            "mean_gross": g["gross"].mean(),
        }
    )
    return out.sort_values("points", ascending=False)


# --- orchestration -------------------------------------------------------------------------

BASELINES = ("openfpl_replica (repeat)", "a0 (repeat)", "last5 (repeat)")
GATE_STRATEGY = "simulator (horizon 5)"
KEEP = ["player_uid", "fixture_uid", "deadline_at", "expected_points"]


def strategy_forecasts(lake: Lake, seasons: list[str]) -> dict[str, tuple[pd.DataFrame, Mode]]:
    """Walk-forward forecasts of every strategy (cached runs; the simulator's horizon-5 run
    takes about 2 h the first time)."""
    from fplh.evaluate.phase3 import run, simulator
    from fplh.models.baselines import A0Player, NaiveLast5
    from fplh.models.openfpl import OpenFPLReplica

    store = SilverStore(lake)
    dim = store.get("dim_fixture")
    by_season = {s: list(historical_deadlines(dim, s)["deadline_at"]) for s in seasons}
    ds = [d for v in by_season.values() for d in v]
    a0 = pd.concat(
        [run(lake, store, A0Player(load_rules(s)), v) for s, v in by_season.items()],
        ignore_index=True,
    )
    preds: dict[str, tuple[pd.DataFrame, Mode]] = {
        GATE_STRATEGY: (run(lake, store, simulator(lake, store, ds, 1000, 5), ds, 5), "native"),
        "simulator (repeat)": (run(lake, store, simulator(lake, store, ds), ds), "repeat"),
        "openfpl_replica (repeat)": (run(lake, store, OpenFPLReplica(), ds), "repeat"),
        "a0 (repeat)": (a0, "repeat"),
        "last5 (repeat)": (run(lake, store, NaiveLast5(), ds), "repeat"),
    }
    preds[V2_STRATEGY] = (v2_forecasts(lake, store, seasons, ds), "native")
    return {k: (v[KEEP].dropna(subset=["expected_points"]), m) for k, (v, m) in preds.items()}


V2_STRATEGY = "v2 (stacked, horizon 5)"
V2_TRAIN_FROM = ("2017-18", "2018-19", "2019-20", "2020-21", "2021-22")


def v2_forecasts(
    lake: Lake, store: SilverStore, seasons: list[str], deadlines: list[pd.Timestamp]
) -> pd.DataFrame:
    """Model v2 at horizon 5: the news-aware simulator's horizon-5 forecasts with the
    replica's next-week forecast repeated, stacked by trees trained walk-forward on the
    horizon-1 forecasts of every earlier season (``models.stack``)."""
    from fplh.evaluate.phase3 import run, simulator
    from fplh.evaluate.v2 import season_runs
    from fplh.models.stack import repeat_forecasts, stack_frame, walk_forward_stack

    h1 = season_runs(lake, store, [*V2_TRAIN_FROM, *seasons])
    sim5 = run(
        lake, store, simulator(lake, store, deadlines, 1000, 5, minutes="news"), deadlines, 5
    )
    pm = store.get("fact_player_match")
    crowd = store.get("fpl_round_transfers")
    prices = pm[["player_uid", "value", "observed_at"]]
    train = stack_frame(h1["sim_news"], h1["replica"], crowd, prices)
    targets = stack_frame(sim5, repeat_forecasts(h1["replica"], sim5), crowd, prices)
    y = pm[["player_uid", "fixture_uid", "total_points", "observed_at"]]
    out: pd.DataFrame = walk_forward_stack(train, y, deadlines, targets=targets)
    return out


def _job(args: tuple[str, str, str, pd.DataFrame, Mode, dict[str, float]]) -> pd.DataFrame:
    from threadpoolctl import threadpool_limits

    lake_uri, season, strategy, pred, mode, params = args
    with threadpool_limits(1):
        data = season_data(SilverStore(Lake(lake_uri)), season)
        return replay_season(strategy, pred, mode, data, **params).log  # type: ignore[arg-type]


@dataclass
class ReplayReport:
    totals: pd.DataFrame
    summary: pd.DataFrame
    gate: pd.DataFrame
    horizon_value: dict[str, float]
    sensitivity: pd.DataFrame
    passed: bool
    logs: pd.DataFrame
    v2_gate: pd.DataFrame = field(default_factory=pd.DataFrame)
    v2_passed: bool = False


def evaluate_replay(
    lake: Lake, lake_uri: str, seasons: list[str], *, jobs: int = 3, sensitivity: bool = False
) -> ReplayReport:
    from concurrent.futures import ProcessPoolExecutor

    from fplh.lake.parquet import to_parquet_bytes

    preds = strategy_forecasts(lake, seasons)
    work: list[tuple[str, str, str, pd.DataFrame, Mode, dict[str, float]]] = []
    for name, (pred, mode) in preds.items():
        for season in seasons:
            work.append((lake_uri, season, name, pred, mode, {}))
    if sensitivity:
        pred, mode = preds[GATE_STRATEGY]
        for beta, delta in ((0.05, 0.9), (0.2, 0.9), (0.1, 0.8), (0.1, 1.0)):
            label = f"{GATE_STRATEGY} β={beta} δ={delta}"
            for season in seasons:
                work.append((lake_uri, season, label, pred, mode, {"beta": beta, "delta": delta}))
    with ProcessPoolExecutor(max_workers=jobs) as pool:
        logs = pd.concat(list(pool.map(_job, work)), ignore_index=True)
    lake.put_bytes(
        "gold/replay/" + "_".join(seasons) + ".parquet",
        to_parquet_bytes(logs.drop(columns=["squad", "xi", "bench"]), ["strategy", "season", "gw"]),
        overwrite=True,
    )
    main = logs[~logs["strategy"].str.contains("β=")]
    totals = season_totals(main)
    total_of = {str(k): float(v) for k, v in totals["total"].items()}
    strongest = max(BASELINES, key=lambda b: total_of[b])
    gate = _gate(main, GATE_STRATEGY, BASELINES, seasons)
    strongest_row = gate[gate["baseline"] == strongest].iloc[0]
    passed = bool(strongest_row["ci_low"] > 0)
    v2_gate = (
        _gate(main, V2_STRATEGY, (*BASELINES, GATE_STRATEGY), seasons)
        if V2_STRATEGY in total_of
        else pd.DataFrame()
    )
    sens = (
        season_totals(logs[logs["strategy"].str.startswith(GATE_STRATEGY)])
        if sensitivity
        else (pd.DataFrame())
    )
    return ReplayReport(
        totals,
        strategy_summary(main),
        gate.assign(strongest=gate["baseline"] == strongest),
        compare_strategies(main, GATE_STRATEGY, "simulator (repeat)"),
        sens,
        passed,
        logs,
        v2_gate,
        bool(v2_gate[v2_gate["baseline"] == strongest]["ci_low"].iloc[0] > 0)
        if not v2_gate.empty
        else False,
    )


def _gate(
    main: pd.DataFrame, strategy: str, baselines: tuple[str, ...], seasons: list[str]
) -> pd.DataFrame:
    rows = []
    for b in baselines:
        c = compare_strategies(main, strategy, b)
        per_season = {
            s: float(
                main[(main["strategy"] == strategy) & (main["season"] == s)]["points"].sum()
                - main[(main["strategy"] == b) & (main["season"] == s)]["points"].sum()
            )
            for s in seasons
        }
        rows.append({"baseline": b, **c, **{f"diff {s}": v for s, v in per_season.items()}})
    return pd.DataFrame(rows)


def versus_average(log: pd.DataFrame, events: pd.DataFrame, season: str) -> pd.DataFrame:
    """A live replay's points per gameweek against FPL's average and highest manager scores
    (``fpl_event``, the last capture of each gameweek with a score)."""
    ev = events[(events["season"] == season) & events["average_entry_score"].notna()]
    ev = ev.sort_values("observed_at").drop_duplicates("gameweek", keep="last")
    ev = ev.set_index("gameweek")[["average_entry_score", "highest_score"]]
    joined = log.join(ev, on="gw", how="inner")
    out: pd.DataFrame = joined.assign(
        versus_average=joined["points"] - joined["average_entry_score"]
    )
    return out[
        ["strategy", "gw", "points", "average_entry_score", "highest_score", "versus_average"]
    ]
