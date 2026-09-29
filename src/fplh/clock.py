"""Injectable UTC clock.

Live runs use :class:`SystemClock`; backtests use :class:`FixedClock` set to a historical
deadline. Everything that stamps ``observed_at`` or filters by deadline takes a clock, so a
backtest is a replay of live with a different clock (ARCHITECTURE.md P5).
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Protocol


class Clock(Protocol):
    def now(self) -> datetime: ...


class SystemClock:
    def now(self) -> datetime:
        return datetime.now(UTC)


class FixedClock:
    def __init__(self, at: datetime) -> None:
        if at.tzinfo is None:
            raise ValueError("FixedClock requires a timezone-aware datetime")
        self._at = at.astimezone(UTC)

    def now(self) -> datetime:
        return self._at

    def advance(self, seconds: float) -> None:
        self._at = self._at + timedelta(seconds=seconds)


def isoformat_z(ts: datetime) -> str:
    """UTC timestamp at second precision, e.g. ``2026-09-29T06:00:02Z``."""
    if ts.tzinfo is None:
        raise ValueError("timestamp must be timezone-aware")
    return ts.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
