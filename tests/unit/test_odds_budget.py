from __future__ import annotations

import itertools
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pytest

from fplh.collectors.odds_budget import (
    FLOOR,
    PlannedCall,
    affordable,
    due,
    gameweek_deadlines,
    plan_calls,
    spare_due,
)

KICKOFFS = [
    datetime.fromisoformat(k.replace("Z", "+00:00"))
    for k in json.loads((Path(__file__).parent / "data" / "kickoffs_2025_26.json").read_text())
]
MONTHS = [
    (datetime(2025, m, 1, tzinfo=UTC), datetime(2025, m + 1, 1, tzinfo=UTC)) for m in (8, 9, 10, 11)
]


@pytest.mark.parametrize(("start", "end"), MONTHS)
def test_real_months_fit_budget_and_cover_every_kickoff(start: datetime, end: datetime) -> None:
    plan = plan_calls(KICKOFFS, period_start=start, period_end=end, budget=500)
    assert sum(c.credits for c in plan) <= 450  # 10 % reserve never planned
    slots = {k for k in KICKOFFS if start < k <= end}
    closing = {c.end for c in plan if c.kind == "closing"}
    assert slots <= closing, "every kickoff slot gets its closing call"
    market = [c for c in plan if c.is_market]
    assert all(a.end <= b.start for a, b in itertools.pairwise(market)), "no overlapping windows"


def test_priorities_under_a_tight_budget() -> None:
    start, end = MONTHS[1]
    slots = sorted({k for k in KICKOFFS if start < k <= end})
    plan = plan_calls(
        KICKOFFS, period_start=start, period_end=end, budget=2 * len(slots) + 4, reserve=0
    )
    kinds = [c.kind for c in plan]
    assert kinds.count("closing") == len(slots)
    assert kinds.count("spare") == 0
    assert sum(c.credits for c in plan) <= 2 * len(slots) + 4


def test_used_credits_reduce_the_plan() -> None:
    start, end = MONTHS[1]
    full = plan_calls(KICKOFFS, period_start=start, period_end=end, budget=500)
    later = plan_calls(KICKOFFS, period_start=start, period_end=end, budget=500, used=400)
    assert sum(c.credits for c in later) <= 50
    assert len(later) < len(full)


def test_deadlines_cluster_rounds() -> None:
    first_round = [k for k in KICKOFFS if k < datetime(2025, 8, 20, tzinfo=UTC)]
    assert gameweek_deadlines(first_round) == [min(first_round) - timedelta(minutes=90)]


def test_due_fires_each_window_once() -> None:
    start, end = MONTHS[1]
    plan = plan_calls(KICKOFFS, period_start=start, period_end=end, budget=500)
    call = next(c for c in plan if c.kind == "closing")
    now = call.start + timedelta(minutes=10)
    assert due(plan, now) == [call]
    assert due(plan, now, done=[call.id]) == []


# several fixtures share the Saturday 15:00 (UK) slot, one fixture elsewhere: ≈ 380 a season
EVENTS = [
    (f"e{i}-{j}", k)
    for i, k in enumerate(KICKOFFS)
    for j in range(4 if (k.weekday(), k.hour) == (5, 14) else 1)
]


def simulate_month(
    start: datetime, end: datetime, credits: int = 500, skip: float = 0.0, seed: int = 0
) -> tuple[int, list[PlannedCall]]:
    """The hourly collector (:47) over a month against a fake account charging full cost."""
    rng = np.random.default_rng(seed)
    remaining, fired, last = credits, [], None
    made: list[PlannedCall] = []
    now = start + timedelta(minutes=47)
    while now < end:
        if rng.random() >= skip:  # GitHub cron sometimes skips a run
            plan = plan_calls(
                KICKOFFS,
                events=EVENTS,
                period_start=now,
                period_end=end,
                available=max(remaining - FLOOR, 0),
            )
            calls = due(plan, now, fired)
            spare = remaining - FLOOR - sum(c.credits for c in plan)
            if not any(c.is_market for c in calls) and spare_due(
                now, plan=plan, spare_credits=spare, period_end=end, last_market_call=last
            ):
                calls.append(PlannedCall(now, now + timedelta(hours=1), "spare", 2))
            for c in calls:
                assert remaining >= FLOOR
                if not affordable(remaining, c.credits):
                    continue
                remaining -= c.credits
                fired.append(c.id)
                made.append(c)
                if c.is_market:
                    last = now
        now += timedelta(hours=1)
    return credits - remaining, made


@pytest.mark.parametrize(("start", "end"), MONTHS)
def test_a_month_uses_the_free_credits_without_exceeding_them(
    start: datetime, end: datetime
) -> None:
    spent, made = simulate_month(start, end)
    assert spent <= 500 - FLOOR  # never below the floor, so never over the quota
    assert spent >= 0.9 * (500 - FLOOR), f"only {spent} of 490 credits used"
    slots = {k for k in KICKOFFS if start + timedelta(hours=1) < k <= end}
    closing = {c.end for c in made if c.kind == "closing"}
    assert slots <= closing, "every kickoff slot captured after the lineups"
    props = {c.event_id for c in made if c.kind == "props"}
    events = {e for e, k in EVENTS if start + timedelta(hours=1) < k <= end}
    assert len(props & events) >= 0.95 * len(events)


@pytest.mark.parametrize(("start", "end"), MONTHS)
def test_deadline_calls_fire_once_per_round(start: datetime, end: datetime) -> None:
    _, made = simulate_month(start, end)
    rounds = [d for d in gameweek_deadlines(KICKOFFS) if start + timedelta(hours=4) < d <= end]
    assert len([c for c in made if c.kind == "deadline"]) == len(rounds)
    kinds = [(c.kind, c.event_id) for c in made if c.event_id]
    assert len(kinds) == len(set(kinds)), "each fixture's props fire once per kind"


def test_mid_round_kickoffs_are_not_new_deadlines() -> None:
    first = min(KICKOFFS)
    later = [k for k in KICKOFFS if first < k < first + timedelta(days=4)]
    plan = plan_calls(
        KICKOFFS, period_start=first + timedelta(minutes=5), period_end=later[-1], available=490
    )
    assert not [c for c in plan if c.kind == "deadline" and c.start < later[-1]]


@pytest.mark.parametrize("skip", [0.2, 0.5])
def test_missed_runs_never_overspend(skip: float) -> None:
    start, end = MONTHS[2]
    spent, _ = simulate_month(start, end, skip=skip, seed=1)
    assert spent <= 500 - FLOOR


def test_late_start_with_few_credits_left() -> None:
    start, end = MONTHS[1]
    spent, made = simulate_month(start + timedelta(days=20), end, credits=37)
    assert spent <= 37 - FLOOR
    kinds = [c.kind for c in made]
    assert kinds.count("closing") >= kinds.count("props")  # closing lines come first


def test_affordable_is_a_hard_floor() -> None:
    assert affordable(12, 2) and not affordable(11, 2)
    assert not affordable(10, 1)
    assert affordable(500, 2, floor=0)


def test_props_are_planned_per_fixture_after_priority_calls() -> None:
    start, end = MONTHS[1]
    plan = plan_calls(KICKOFFS, events=EVENTS, period_start=start, period_end=end, available=490)
    props = [c for c in plan if c.kind == "props"]
    events = [e for e, k in EVENTS if start < k <= end]
    assert {c.event_id for c in props} >= set(events)
    tight = plan_calls(KICKOFFS, events=EVENTS, period_start=start, period_end=end, available=60)
    assert sum(c.credits for c in tight) <= 60
    slots = {k for k in KICKOFFS if start < k <= end}
    kinds = [c.kind for c in tight]
    # match odds around every kickoff come before any props
    assert kinds.count("closing") == min(len(slots), 30)
    if 4 * len(slots) >= 60:
        assert kinds.count("props") == 0
