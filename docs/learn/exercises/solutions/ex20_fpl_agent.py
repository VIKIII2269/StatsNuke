"""Reference solutions for exercise 20."""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Any

from fplh.models.goal_benchmarks import g1_grid

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "agent"))
import fpl_agent


def clean_sheet_odds(home_rate: float, away_rate: float) -> dict[str, float]:
    g = g1_grid(home_rate, away_rate, fpl_agent.RHO)
    return {
        "p_home_clean_sheet": round(float(g[:, 0].sum()), 4),
        "p_away_clean_sheet": round(float(g[0, :].sum()), 4),
    }


CLEAN_SHEET_SPEC: dict[str, Any] = {
    "description": "Clean-sheet probabilities for both sides of a match, given each team's "
    "expected goals (e.g. from invert_market). Uses the repo's Dixon–Coles grid. Use this for "
    "goalkeeper/defender clean-sheet questions.",
    "input_schema": {
        "type": "object",
        "properties": {
            "home_rate": {"type": "number", "description": "home expected goals, 0.3–3.5"},
            "away_rate": {"type": "number", "description": "away expected goals, 0.3–3.5"},
        },
        "required": ["home_rate", "away_rate"],
        "additionalProperties": False,
    },
}


def register(name: str, spec: dict[str, Any], fn: Any) -> None:
    fpl_agent.TOOLS[name] = (spec, fn)


def extract_number(text: str) -> float | None:
    m = re.search(r"(-?\d+(?:\.\d+)?)\s*(%)?", text)
    if not m:
        return None
    value = float(m.group(1))
    return round(value / 100, 10) if m.group(2) else value


def pass_rates(transcripts: list[dict[str, Any]]) -> dict[str, float]:
    ans, traj, both = [], [], []
    for t in transcripts:
        x = extract_number(t["answer"])
        a = x is not None and abs(x - t["expected"]) <= t["tolerance"] + 1e-12
        j = set(t["required_tools"]) <= set(t["tools_called"])
        ans.append(a)
        traj.append(j)
        both.append(a and j)
    n = len(transcripts)
    return {"answer": sum(ans) / n, "trajectory": sum(traj) / n, "both": sum(both) / n}
