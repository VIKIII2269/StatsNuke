from __future__ import annotations

import itertools
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from fplh.collectors.odds_budget import due, gameweek_deadlines, plan_calls

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
    assert all(a.end <= b.start for a, b in itertools.pairwise(plan)), "no overlapping windows"


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
