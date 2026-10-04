"""Credit budget planner for The Odds API (spec gap 7): use the free 500 credits a month as
fully as possible without ever running out.

The collector runs hourly (GitHub Actions, :47). Each run lists the upcoming events (free),
reads the *authoritative* credits remaining from the response headers, re-plans the rest
of the month and fires what is due now. Priorities, strictly in order (each reserves its
credits before anything lower):

1. **closing**: match odds (h2h + totals, 2 credits, one call covers every fixture) in the
   hour before each distinct kickoff time: after the lineups (≈ T−60) and close to the
   closing line;
2. **pre_lineup**: match odds in the hour before that (T−2 h to T−1 h), so the line move
   on team news is observed;
3. **deadline**: match odds two to three hours before each gameweek deadline;
4. **props**: the anytime-scorer market per fixture (1 credit each, event endpoint) in
   the closing hour;
5. **props_deadline**: the same per fixture at the deadline;
6. **spare**: whatever is left is spent on extra match-odds snapshots paced evenly to the
   end of the month (``spare_due``), so the month's credits are used, not wasted.

Kickoffs that already started drop out of the API's listing, so the collector keeps every
kickoff it has seen this month and passes them all: deadlines are derived from the full
schedule, not from what is still upcoming.

Never-exceed guarantee: plans only ever spend ``remaining − floor``, and the collector
re-checks the live remaining credits before *every* call (``affordable``), so a missed
run, a mid-month change or a planning error cannot overspend.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Literal

Kind = Literal["closing", "pre_lineup", "deadline", "props", "props_deadline", "spare"]
PRIORITY: dict[Kind, int] = {
    "closing": 0,
    "pre_lineup": 1,
    "deadline": 2,
    "props": 3,
    "props_deadline": 4,
    "spare": 5,
}
MARKET_KINDS = ("closing", "pre_lineup", "deadline", "spare")
WINDOW = timedelta(hours=1)
FLOOR = 10  # credits never planned or spent: a cushion against header lag and retries


@dataclass(frozen=True, order=True)
class PlannedCall:
    start: datetime
    end: datetime
    kind: Kind
    credits: int
    event_id: str | None = None

    @property
    def id(self) -> str:
        if self.event_id:  # props fire once per fixture and kind, whatever the window
            return f"{self.kind}#{self.event_id}"
        return f"{self.kind}@{self.start.isoformat()}"

    @property
    def is_market(self) -> bool:
        return self.kind in MARKET_KINDS

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
    budget: int = 500,
    used: int = 0,
    cost: int = 2,
    reserve: float = 0.1,
    deadlines: Sequence[datetime] | None = None,
    available: int | None = None,
    events: Sequence[tuple[str, datetime]] = (),
    prop_cost: int = 1,
) -> list[PlannedCall]:
    """Priority calls for ``[period_start, period_end)`` within ``available`` credits
    (default ``budget·(1 − reserve) − used``). Spare snapshots are not planned here: see
    ``spare_due``. Market-odds windows never overlap, so an hourly run fires at most one."""
    left = int(budget * (1 - reserve)) - used if available is None else available
    every = sorted(set(kickoffs))
    ks = [k for k in every if period_start < k <= period_end + 2 * WINDOW]
    # deadlines from every known kickoff, past ones included: once a round's first match has
    # kicked off, its later matches must not look like the start of a new round
    ds = sorted(deadlines if deadlines is not None else gameweek_deadlines(every))
    candidates: list[PlannedCall] = [PlannedCall(k - WINDOW, k, "closing", cost) for k in ks]
    candidates += [PlannedCall(k - 2 * WINDOW, k - WINDOW, "pre_lineup", cost) for k in ks]
    candidates += [
        PlannedCall(d - timedelta(hours=2) - WINDOW, d - timedelta(hours=2), "deadline", cost)
        for d in ds
    ]
    for eid, k in events:
        if period_start < k <= period_end + WINDOW:
            candidates.append(PlannedCall(k - WINDOW, k, "props", prop_cost, eid))
            d = max((x for x in ds if x <= k), default=None)
            if d is not None:
                w = d - timedelta(hours=2)
                candidates.append(PlannedCall(w - WINDOW, w, "props_deadline", prop_cost, eid))
    candidates = [c for c in candidates if c.end > period_start and c.start < period_end]

    chosen: list[PlannedCall] = []
    for c in sorted(candidates, key=lambda c: (PRIORITY[c.kind], c.start, c.event_id or "")):
        if c.credits > left:
            continue  # a cheaper, lower-priority call may still fit
        if c.is_market and any(x.is_market and _overlaps(c, x) for x in chosen):
            continue  # a higher-priority market call already covers this window
        if any(x.id == c.id for x in chosen):
            continue
        chosen.append(c)
        left -= c.credits
    return sorted(chosen)


def spare_due(
    now: datetime,
    *,
    plan: Sequence[PlannedCall],
    spare_credits: int,
    period_end: datetime,
    last_market_call: datetime | None,
    cost: int = 2,
) -> bool:
    """Whether to spend a spare match-odds snapshot now: credits left after every planned
    call, no planned market call this hour, and paced so the rest of the month gets its
    share (interval = time left ÷ spare calls left)."""
    if spare_credits < cost or any(c.is_market and c.contains(now) for c in plan):
        return False
    calls_left = spare_credits // cost
    interval = (period_end - now) / (calls_left + 1)
    return last_market_call is None or now - last_market_call >= interval


def affordable(remaining: int, cost: int, floor: int = FLOOR) -> bool:
    """The hard stop checked before every call: never let the account go below ``floor``."""
    return remaining - cost >= floor


def due(plan: Sequence[PlannedCall], now: datetime, done: Iterable[str] = ()) -> list[PlannedCall]:
    """Planned calls whose window contains ``now`` and that have not fired yet, in
    priority order."""
    fired = set(done)
    out = [c for c in plan if c.contains(now) and c.id not in fired]
    return sorted(out, key=lambda c: (PRIORITY[c.kind], c.event_id or ""))
