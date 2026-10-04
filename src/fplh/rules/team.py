"""A manager's gameweek score from their picks and the players' actual points (FPL rules).

Picks are 15 players in order: 1–11 the starting XI, 12 the substitute goalkeeper, 13–15
the outfield bench in priority order. A player *played* if he had minutes in any of his
fixtures that gameweek (doubles are summed).

* **Automatic substitution:** each starter who did not play is replaced, in XI order, by
  the first bench player (in priority order) who played and keeps a valid formation (one
  goalkeeper, at least the ``xi_min`` per position); a goalkeeper only by the substitute
  goalkeeper.
* **Captaincy:** the captain's points count twice (three times with the triple captain);
  if the captain did not play, the vice-captain gets the multiplier instead (if he
  played).
* **Bench boost:** all 15 players score and no substitutions are made.
* Wildcard and free hit change transfers, not scoring.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

CHIPS = ("wildcard", "free_hit", "triple_captain", "bench_boost")


@dataclass(frozen=True)
class GameweekScore:
    points: int
    lineup: tuple[str, ...]  # the 11 (or 15 with bench boost) players who scored
    substitutions: tuple[tuple[str, str], ...] = field(default=())  # (out, in)
    multiplied: str | None = None  # the player whose points were multiplied


def _valid(positions: Sequence[str], xi_min: Mapping[str, int]) -> bool:
    count = {p: 0 for p in xi_min}
    for p in positions:
        count[p] = count.get(p, 0) + 1
    return count.get("GK", 0) == 1 and all(count.get(p, 0) >= n for p, n in xi_min.items())


def score_gameweek(
    picks: Sequence[str],
    position: Mapping[str, str],
    points: Mapping[str, int],
    minutes: Mapping[str, int],
    captain: str,
    vice: str,
    chip: str | None,
    xi_min: Mapping[str, int],
) -> GameweekScore:
    if len(picks) != 15 or len(set(picks)) != 15:
        raise ValueError("picks must be 15 distinct players")
    if chip is not None and chip not in CHIPS:
        raise ValueError(f"unknown chip {chip!r}")
    if captain not in picks[:11] or vice not in picks[:11] or captain == vice:
        raise ValueError("captain and vice must be two different starters")
    if not _valid([position[p] for p in picks[:11]], xi_min):
        raise ValueError("the starting XI is not a valid formation")

    def played(p: str) -> bool:
        return minutes.get(p, 0) > 0

    subs: list[tuple[str, str]] = []
    if chip == "bench_boost":
        lineup = list(picks)
    else:
        lineup = list(picks[:11])
        bench = list(picks[11:])
        used: set[str] = set()
        for i, starter in enumerate(list(lineup)):
            if played(starter):
                continue
            for b in bench:
                if b in used or not played(b):
                    continue
                if (position[starter] == "GK") != (position[b] == "GK"):
                    continue
                trial = lineup.copy()
                trial[i] = b
                if _valid([position[p] for p in trial], xi_min):
                    lineup = trial
                    used.add(b)
                    subs.append((starter, b))
                    break
    multiplier = 3 if chip == "triple_captain" else 2
    chosen = captain if played(captain) else (vice if played(vice) else None)
    total = sum(points.get(p, 0) for p in lineup)
    if chosen is not None and chosen in lineup:
        total += (multiplier - 1) * points.get(chosen, 0)
    return GameweekScore(int(total), tuple(lineup), tuple(subs), chosen)
