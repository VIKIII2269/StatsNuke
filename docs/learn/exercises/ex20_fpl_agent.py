"""Exercise 20: extend the FPL agent with a tool, then grade recorded agent runs.

Run: uv run python docs/learn/exercises/ex20_fpl_agent.py   (offline, no API key needed)
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from _check import close, run, task

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "agent"))
import fpl_agent

# ------------------------------------------------------------------ demo
print("tools:", sorted(fpl_agent.TOOLS))
print(fpl_agent.call_tool("match_markets", {"home_rate": 1.7, "away_rate": 0.9})[0][:120], "…")

TRANSCRIPTS: list[dict[str, Any]] = [
    {  # right answer, right tools
        "expected": 0.4066,
        "tolerance": 0.005,
        "required_tools": {"match_markets"},
        "tools_called": ["match_markets"],
        "answer": "Arsenal keep a clean sheet 40.7% of the time.",
    },
    {  # right number, but computed in its head (no tool) → trajectory fails
        "expected": 0.5422,
        "tolerance": 0.005,
        "required_tools": {"devig_odds"},
        "tools_called": [],
        "answer": "Roughly 0.542 for the home win.",
    },
    {  # used the tool, then misreported the number
        "expected": 0.2088,
        "tolerance": 0.005,
        "required_tools": {"devig_odds"},
        "tools_called": ["devig_odds"],
        "answer": "The away side is about 27.8% to win.",
    },
    {  # fine
        "expected": 12,
        "tolerance": 0,
        "required_tools": {"score_player"},
        "tools_called": ["score_player", "score_player"],
        "answer": "That's 12 points.",
    },
]


# ------------------------------------------------------------------ your tasks
def clean_sheet_odds(home_rate: float, away_rate: float) -> dict[str, float]:
    """{'p_home_clean_sheet': P(away scores 0), 'p_away_clean_sheet': P(home scores 0)} from
    fplh's Dixon–Coles grid g1_grid(home_rate, away_rate, fpl_agent.RHO), rounded to 4 d.p."""
    raise NotImplementedError


# Fill in a tool definition: a "description" (a prompt for the model: what, units, when to use)
# and an "input_schema" (JSON Schema object: both rates required numbers, no extra fields).
CLEAN_SHEET_SPEC: dict[str, Any] = {}


def register(name: str, spec: dict[str, Any], fn: Any) -> None:
    """Add the tool to fpl_agent.TOOLS so tool_specs() and call_tool() see it."""
    raise NotImplementedError


def extract_number(text: str) -> float | None:
    """First number in the text; percentages become fractions ('40.7%' → 0.407). None if none."""
    raise NotImplementedError


def pass_rates(transcripts: list[dict[str, Any]]) -> dict[str, float]:
    """{'answer': share whose extracted number is within tolerance of expected,
    'trajectory': share whose required_tools ⊆ tools_called,
    'both': share passing both}."""
    raise NotImplementedError


@task("clean_sheet_odds agrees with the agent's match_markets tool")
def _() -> None:
    got = clean_sheet_odds(1.7, 0.9)
    ref = fpl_agent.match_markets(1.7, 0.9)
    close(
        [got["p_home_clean_sheet"], got["p_away_clean_sheet"]],
        [ref["p_home_clean_sheet"], ref["p_away_clean_sheet"]],
    )


@task("the tool spec is a well-formed, strict-shaped schema with a real description")
def _() -> None:
    s = CLEAN_SHEET_SPEC
    assert len(s.get("description", "")) > 40, "write a description the model can act on"
    schema = s["input_schema"]
    assert schema["type"] == "object"
    assert set(schema["required"]) == {"home_rate", "away_rate"}
    assert schema.get("additionalProperties") is False
    assert all(schema["properties"][k]["type"] == "number" for k in ("home_rate", "away_rate"))


@task("registered tool works through call_tool, and bad input is an error result")
def _() -> None:
    register("clean_sheet_odds", CLEAN_SHEET_SPEC, clean_sheet_odds)
    assert "clean_sheet_odds" in [t["name"] for t in fpl_agent.tool_specs()]
    content, err = fpl_agent.call_tool("clean_sheet_odds", {"home_rate": 1.7, "away_rate": 0.9})
    assert not err, content
    _, err = fpl_agent.call_tool("clean_sheet_odds", {"home_rate": 1.7})
    assert err


@task("extract_number")
def _() -> None:
    assert extract_number("Arsenal keep a clean sheet 40.7% of the time.") == 0.407
    assert extract_number("Roughly 0.542 for the home win.") == 0.542
    assert extract_number("That's 12 points.") == 12.0
    assert extract_number("no idea") is None


@task("pass_rates on the recorded transcripts")
def _() -> None:
    r = pass_rates(TRANSCRIPTS)
    close([r["answer"], r["trajectory"], r["both"]], [0.75, 0.75, 0.5])


run(globals())
