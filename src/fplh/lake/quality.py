"""Silver quality gates (ARCHITECTURE.md §6.6).

Each check returns a :class:`CheckResult`. Blocking failures stop the build before any
silver table is written, so downstream jobs never see data that failed a gate.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field

import pandas as pd
import yaml

from fplh.settings import get_settings

Tables = Mapping[str, pd.DataFrame]
COVERAGE_MIN = 0.995


@dataclass
class CheckResult:
    name: str
    blocking: bool
    passed: bool
    details: str = ""
    failures: pd.DataFrame = field(default_factory=pd.DataFrame)

    def line(self) -> str:
        mark = "PASS" if self.passed else ("FAIL" if self.blocking else "WARN")
        return f"[{mark}] {self.name}: {self.details}"


class QualityGateError(RuntimeError):
    def __init__(self, results: list[CheckResult]) -> None:
        self.results = results
        failed = [r for r in results if r.blocking and not r.passed]
        super().__init__("quality gates failed:\n" + "\n".join(r.line() for r in failed))


def load_exceptions(check: str) -> list[dict[str, object]]:
    """Hand-verified upstream defects excused from ``check`` (configs/quality/exceptions.yaml)."""
    path = get_settings().configs_dir / "quality" / "exceptions.yaml"
    if not path.exists():
        return []
    entries = (yaml.safe_load(path.read_text()) or {}).get(check) or []
    for e in entries:
        if not e.get("reason"):
            raise ValueError(f"quality exception without a reason: {e}")
    return list(entries)


def _empty(*tables: pd.DataFrame) -> bool:
    return any(t is None or t.empty for t in tables)


def check_goal_conservation(t: Tables) -> CheckResult:
    """Team goals = Σ own players' goals + Σ opponent players' own goals."""
    pm, fx = t.get("fact_player_match"), t.get("fpl_fixture")
    if pm is None or fx is None or _empty(pm, fx):
        return CheckResult("goal_conservation", True, True, "no data")
    scored = pm.groupby(["fixture_uid", "team"])["goals_scored"].sum()
    own = pm.groupby(["fixture_uid", "opponent"])["own_goals"].sum()
    own.index = own.index.set_names(["fixture_uid", "team"])
    derived = scored.add(own, fill_value=0)
    played = fx.dropna(subset=["home_goals", "away_goals"])
    official = pd.concat(
        [
            played.set_index(["fixture_uid", "home_team"])["home_goals"].rename_axis(
                ["fixture_uid", "team"]
            ),
            played.set_index(["fixture_uid", "away_team"])["away_goals"].rename_axis(
                ["fixture_uid", "team"]
            ),
        ]
    ).astype("int64")
    both = pd.DataFrame(
        {"official": official, "derived": derived.reindex(official.index).fillna(0).astype("int64")}
    )
    bad = both[both["official"] != both["derived"]]
    return CheckResult(
        "goal_conservation",
        True,
        bad.empty,
        f"{len(bad)} of {len(both)} team-fixtures differ",
        bad.reset_index(),
    )


def check_player_plausibility(t: Tables) -> CheckResult:
    pm = t.get("fact_player_match")
    if pm is None or pm.empty:
        return CheckResult("player_plausibility", True, True, "no data")
    bad_minutes = pm[(pm["minutes"] < 0) | (pm["minutes"] > 130)]
    starters = pm.dropna(subset=["starts"]).groupby(["fixture_uid", "team"])["starts"].sum()
    too_many = starters[starters > 11]
    ok = bad_minutes.empty and too_many.empty
    return CheckResult(
        "player_plausibility",
        True,
        ok,
        f"{len(bad_minutes)} rows with minutes outside 0–130; "
        f"{len(too_many)} team-fixtures with > 11 starters",
        pd.concat([bad_minutes, too_many.reset_index()], ignore_index=True),
    )


def check_cross_source_scores(t: Tables) -> CheckResult:
    """Final scores agree across FPL, football-data and Understat for each fixture."""
    frames = []
    for name, src in (
        ("fpl_fixture", "fpl"),
        ("fd_match", "football_data"),
        ("us_match", "understat"),
    ):
        df = t.get(name)
        if df is None or df.empty:
            continue
        if name == "fd_match":
            df = df[df["division"] == "E0"]
        frames.append(df[["fixture_uid", "home_goals", "away_goals"]].dropna().assign(src=src))
    if len(frames) < 2:
        return CheckResult("cross_source_scores", True, True, "fewer than two sources")
    allf = pd.concat(frames)
    spread = allf.groupby("fixture_uid").agg(
        n=("src", "nunique"), h=("home_goals", "nunique"), a=("away_goals", "nunique")
    )
    bad = spread[(spread["n"] > 1) & ((spread["h"] > 1) | (spread["a"] > 1))]
    detail = allf[allf["fixture_uid"].isin(bad.index)].sort_values("fixture_uid")
    return CheckResult(
        "cross_source_scores",
        True,
        bad.empty,
        f"{len(bad)} of {int((spread['n'] > 1).sum())} multi-source fixtures disagree",
        detail,
    )


def check_understat_shots(t: Tables) -> CheckResult:
    """Shots per match side = Σ roster shots (own goals are not the side's shots)."""
    shots, roster = t.get("fact_shot"), t.get("fact_player_match_understat")
    if shots is None or roster is None or _empty(shots, roster):
        return CheckResult("understat_shot_conservation", True, True, "no data")
    counted = shots[shots["result"] != "OwnGoal"].groupby(["understat_match_id", "side"]).size()
    rostered = roster.groupby(["understat_match_id", "side"])["shots"].sum()
    both = pd.DataFrame({"shots": counted, "roster": rostered}).fillna(0)
    excused = {
        (int(str(e["match"])), str(e["side"]))
        for e in load_exceptions("understat_shot_conservation")
    }
    differs = both[both["shots"] != both["roster"]]
    is_excused = [(int(m), str(sd)) in excused for m, sd in differs.index]
    bad = differs[[not x for x in is_excused]]
    return CheckResult(
        "understat_shot_conservation",
        True,
        bad.empty,
        f"{len(bad)} of {len(both)} match sides differ "
        f"({sum(is_excused)} excused upstream defects)",
        bad.reset_index(),
    )


def check_kickoff_agreement(t: Tables, tolerance_hours: float = 36.0) -> CheckResult:
    dim = t.get("dim_fixture")
    frames = []
    for name in ("fpl_fixture", "fd_match", "us_match"):
        df = t.get(name)
        if df is not None and not df.empty:
            frames.append(df[["fixture_uid", "kickoff_at"]].assign(src=name))
    if dim is None or len(frames) < 2:
        return CheckResult("kickoff_agreement", False, True, "fewer than two sources")
    allf = pd.concat(frames)
    span = allf.groupby("fixture_uid")["kickoff_at"].agg(
        lambda s: (s.max() - s.min()).total_seconds() / 3600
    )
    bad = span[span > tolerance_hours]
    return CheckResult(
        "kickoff_agreement",
        False,
        bad.empty,
        f"{len(bad)} fixtures with kickoffs > {tolerance_hours:g} h apart",
        bad.reset_index(),
    )


def check_entity_coverage(t: Tables, minimum: float = COVERAGE_MIN) -> CheckResult:
    cov = t.get("entity_coverage")
    if cov is None or cov.empty:
        return CheckResult("entity_coverage", True, True, "no Understat data to link against")
    bad = cov[cov["share"] < minimum]
    worst = cov["share"].min()
    return CheckResult(
        "entity_coverage",
        True,
        bad.empty,
        f"min share {worst:.4f} (gate {minimum}); failing seasons: {bad['season'].tolist()}",
        bad,
    )


def check_odds_coverage(t: Tables, from_season: str = "2014-15") -> CheckResult:
    """Every EPL fixture (from ``from_season``) has a pre-match 1X2 price before kickoff."""
    dim, odds = t.get("dim_fixture"), t.get("snap_odds")
    if dim is None or odds is None or _empty(dim, odds):
        return CheckResult("odds_coverage", True, True, "no odds data")
    fx = dim[(dim["season"] >= from_season) & dim["in_football_data"]]
    pre = odds[
        (odds["market"] == "1x2")
        & (~odds["is_closing"])
        & (odds["observed_at"] < odds["kickoff_at"])
    ]
    missing = fx[~fx["fixture_uid"].isin(set(pre["fixture_uid"]))]
    return CheckResult(
        "odds_coverage",
        True,
        missing.empty,
        f"{len(missing)} of {len(fx)} fixtures without pre-match odds",
        missing,
    )


CHECKS: list[Callable[[Tables], CheckResult]] = [
    check_goal_conservation,
    check_player_plausibility,
    check_cross_source_scores,
    check_understat_shots,
    check_kickoff_agreement,
    check_entity_coverage,
    check_odds_coverage,
]


def run_gates(tables: Tables, *, raise_on_fail: bool = True) -> list[CheckResult]:
    results = [check(tables) for check in CHECKS]
    if raise_on_fail and any(r.blocking and not r.passed for r in results):
        raise QualityGateError(results)
    return results
