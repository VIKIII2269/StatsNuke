"""Credit budget planner for The Odds API (spec gap 7).

The free tier gives 500 credits per month; one EPL odds call with ``h2h,totals`` in one
region costs 2. The planner is a pure function of the kickoff schedule, so it can be
unit-tested over a whole month. Priorities, strictly in order:

1. **closing**: one call in the hour before each distinct kickoff time (one call covers
   every fixture), which supplies the closing line;
2. **deadline**: one call in the hour ending two hours before each gameweek deadline;
3. **spare**: whatever budget remains, spread evenly over the rest of the month.

A reserve fraction is never planned so ad-hoc checks cannot exhaust the month.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Literal

Kind = Literal["closing", "deadline", "spare"]
PRIORITY: dict[Kind, int] = {"closing": 0, "deadline": 1, "spare": 2}
WINDOW = timedelta(hours=1)


@dataclass(frozen=True, order=True)
class PlannedCall:
    start: datetime
    end: datetime
    kind: Kind
    credits: int

    @property
    def id(self) -> str:
        return f"{self.kind}@{self.start.isoformat()}"

    def contains(self, t: datetime) -> bool:
        return self.start <= t < self.end


def gameweek_deadlines(
    kickoffs: Iterable[datetime],
    gap: timedelta = timedelta(hours=36),
    before: timedelta = timedelta(minutes=90),
) -> list[datetime]:
    """Approximate deadlines: kickoffs are clustered into rounds separated by ``gap``;
    each round's deadline is its first kickoff minus ``before`` (FPL's rule)."""
    out: list[datetime] = []
    prev: datetime | None = None
    for k in sorted(set(kickoffs)):
        if prev is None or k - prev > gap:
            out.append(k - before)
        prev = k
    return out


def _overlaps(a: PlannedCall, b: PlannedCall) -> bool:
    return a.start < b.end and b.start < a.end


def plan_calls(
    kickoffs: Iterable[datetime],
    *,
    period_start: datetime,
    period_end: datetime,
    budget: int,
    used: int = 0,
    cost: int = 2,
    reserve: float = 0.1,
    deadlines: Sequence[datetime] | None = None,
) -> list[PlannedCall]:
    """Calls for ``[period_start, period_end)`` given ``used`` credits so far."""
    available = int(budget * (1 - reserve)) - used
    ks = sorted({k for k in kickoffs if period_start < k <= period_end + WINDOW})
    ds = sorted(deadlines if deadlines is not None else gameweek_deadlines(ks))

    candidates: list[PlannedCall] = [PlannedCall(k - WINDOW, k, "closing", cost) for k in ks] + [
        PlannedCall(d - timedelta(hours=2) - WINDOW, d - timedelta(hours=2), "deadline", cost)
        for d in ds
    ]
    candidates = [c for c in candidates if c.end > period_start and c.start < period_end]

    chosen: list[PlannedCall] = []
    for c in sorted(candidates, key=lambda c: (PRIORITY[c.kind], c.start)):
        if available < c.credits:
            break
        if any(_overlaps(c, x) for x in chosen):
            continue  # a higher-priority call already covers this window
        chosen.append(c)
        available -= c.credits

    n_spare = available // cost
    if n_spare > 0:
        step = (period_end - period_start) / (n_spare + 1)
        for i in range(1, n_spare + 1):
            start = period_start + step * i
            spare = PlannedCall(start, start + WINDOW, "spare", cost)
            if not any(_overlaps(spare, x) for x in chosen):
                chosen.append(spare)
    return sorted(chosen)


def due(plan: Sequence[PlannedCall], now: datetime, done: Iterable[str] = ()) -> list[PlannedCall]:
    """Planned calls whose window contains ``now`` and that have not fired yet."""
    fired = set(done)
    return [c for c in plan if c.contains(now) and c.id not in fired]
