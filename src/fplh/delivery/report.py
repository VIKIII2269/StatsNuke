"""Markdown report for the live issue thread: the model team's week and season, and the
paper betting forward test. Plain markdown so a workflow can post it with ``gh``."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pandas as pd

from fplh.live.team import LiveTeam, season_table


def _name(names: Mapping[str, str], uid: str) -> str:
    return str(names.get(uid, uid))


def team_week(gw: int, week: Mapping[str, Any], names: Mapping[str, str]) -> str:
    exp = week.get("expected", {})

    def who(p: str) -> str:
        tag = " (C)" if p == week["captain"] else " (V)" if p == week["vice"] else ""
        return f"{_name(names, p)}{tag} {exp.get(p, 0):.1f}"

    lines = [
        f"### Model team, GW{gw} (deadline {str(week['deadline'])[:16].replace('T', ' ')} UTC)",
        f"- **Expected points:** {week['expected_points']:.1f}"
        + (f" · **chip:** {week['chip'].replace('_', ' ')}" if week.get("chip") else ""),
        "- **XI:** " + ", ".join(who(p) for p in week["xi"]),
        "- **Bench:** " + ", ".join(who(p) for p in week["bench"]),
    ]
    if week.get("buys"):
        out = ", ".join(_name(names, p) for p in week["sells"]) or "—"
        inn = ", ".join(_name(names, p) for p in week["buys"])
        hits = f" ({week['hits']} hit{'s' if week['hits'] != 1 else ''})" if week["hits"] else ""
        lines.append(f"- **Transfers:** out {out} → in {inn}{hits}")
    lines.append(f"- **Bank:** £{week['bank'] / 10:.1f}m")
    return "\n".join(lines)


def team_results(team: LiveTeam) -> str:
    t = season_table(team)
    if t.empty:
        return "_No gameweek scored yet._"
    last = t.iloc[-1]
    head = (
        f"### Results\n- **GW{int(last['gw'])}:** {int(last['points'])} points "
        f"(average {_fmt(last['average'])}, highest {_fmt(last['highest'])})\n"
        f"- **Season:** {int(last['total'])} points against an average manager's "
        f"{int(last['average_total'])} ({int(last['total'] - last['average_total']):+d})"
    )
    rows = ["| GW | Points | Average | Highest | Hits | Chip |", "|---|---|---|---|---|---|"]
    for _, r in t.iterrows():
        rows.append(
            f"| {int(r['gw'])} | {int(r['points'])} | {_fmt(r['average'])} | "
            f"{_fmt(r['highest'])} | {int(r['hits'])} | {r['chip'] or ''} |"
        )
    return head + "\n\n" + "\n".join(rows)


def bets_section(new: pd.DataFrame, summary: Mapping[str, float]) -> str:
    lines = ["### Paper bets (consensus value, paper only)"]
    if new.empty:
        lines.append("- No new paper bets since the last report.")
    else:
        lines.append("| Match | Market | Pick | Book | Price | Edge |")
        lines.append("|---|---|---|---|---|---|")
        for _, b in new.sort_values("ev", ascending=False).iterrows():
            lines.append(
                f"| {b['home_team']} v {b['away_team']} | {b['market']} | {b['outcome']} | "
                f"{b['bookmaker']} | {b['price']:.2f} | {b['ev']:+.1%} |"
            )
    if summary.get("bets"):
        lines.append(
            f"- **Season so far:** {int(summary['bets'])} settled bets, mean CLV "
            f"{summary['mean_clv']:+.2%} [{summary['clv_low']:+.2%}, {summary['clv_high']:+.2%}], "
            f"hit rate {summary['hit_rate']:.0%}, paper ROI {summary['roi']:+.1%} "
            f"(1 unit per bet)"
        )
    return "\n".join(lines)


def _fmt(x: Any) -> str:
    return "—" if x is None or pd.isna(x) else str(int(x))


def render(sections: list[str]) -> str:
    return "\n\n".join(s for s in sections if s) + "\n"
