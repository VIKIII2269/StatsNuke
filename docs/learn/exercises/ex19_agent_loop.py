"""Exercise 19: write an agent loop, offline, against a scripted fake model.

Run: uv run python docs/learn/exercises/ex19_agent_loop.py

The fake model mimics the Messages API response shape: an object with `stop_reason` and a
list of content blocks (`text` or `tool_use` with id, name, input). Your loop must:
append the assistant content, run tools, return ALL results in ONE user message, turn
problems into `is_error` results, handle refusal / max_tokens, and cap iterations.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from _check import close, run, task

from fplh.models.market import devig


@dataclass
class Block:
    type: str  # "text" | "tool_use"
    text: str = ""
    id: str = ""
    name: str = ""
    input: dict[str, Any] = field(default_factory=dict)


@dataclass
class Response:
    stop_reason: str  # "end_turn" | "tool_use" | "refusal" | "max_tokens"
    content: list[Block]


class ScriptedModel:
    """Returns pre-written responses in order; records the messages it was sent."""

    def __init__(self, script: list[Response]) -> None:
        self.script = list(script)
        self.seen: list[list[dict[str, Any]]] = []

    def create(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> Response:
        self.seen.append(json.loads(json.dumps(messages, default=lambda b: b.__dict__)))
        if not self.script:
            return Response(
                "tool_use", [Block("tool_use", id="x", name="devig_odds", input={"prices": [2, 2]})]
            )
        return self.script.pop(0)


DEVIG_SCHEMA = {
    "type": "object",
    "properties": {
        "prices": {"type": "array", "items": {"type": "number"}},
        "method": {"type": "string", "enum": ["multiplicative", "power", "shin"]},
    },
    "required": ["prices"],
    "additionalProperties": False,
}


def devig_tool(prices: list[float], method: str = "multiplicative") -> dict[str, Any]:
    return {"probabilities": [round(float(p), 4) for p in devig(prices, method)]}


TOOLS: dict[str, tuple[dict[str, Any], Callable[..., Any]]] = {
    "devig_odds": (DEVIG_SCHEMA, devig_tool)
}
TYPES = {
    "number": (int, float),
    "integer": (int,),
    "string": (str,),
    "array": (list,),
    "object": (dict,),
}


# ------------------------------------------------------------------ your tasks
def validate(schema: dict[str, Any], inp: dict[str, Any]) -> str | None:
    """None if valid, else an error message. Check: required keys present; no unknown keys when
    additionalProperties is False; each present key has the declared JSON type (TYPES);
    enum membership when an enum is declared."""
    raise NotImplementedError


def run_agent(
    model: ScriptedModel,
    tools: dict[str, tuple[dict[str, Any], Callable[..., Any]]],
    question: str,
    max_steps: int = 6,
) -> tuple[str, list[tuple[str, dict[str, Any], bool]]]:
    """Return (final_text, trace). trace holds (tool name, input, is_error) per tool call.
    - messages start as [{"role": "user", "content": question}]
    - each step: r = model.create(messages, tool_specs), then append
      {"role": "assistant", "content": r.content}
    - end_turn → return the joined text of r's text blocks
    - refusal → return "[refused]"; max_tokens → "[truncated]"
    - tool_use → for EVERY tool_use block build {"type": "tool_result", "tool_use_id": b.id,
      "content": json.dumps(result)} — or with "is_error": True and an error string for an unknown
      tool, invalid input (validate) or an exception — then append ONE user message with all of them
    - after max_steps model calls without finishing → return "[step limit]"
    tool_specs is a list of {"name", "input_schema"} built from `tools`."""
    raise NotImplementedError


def grade_trajectory(trace: list[tuple[str, dict[str, Any], bool]], required: set[str]) -> bool:
    """True if every required tool was called at least once WITHOUT error."""
    raise NotImplementedError


def tool_use(id_: str, name: str, **inp: Any) -> Block:
    return Block("tool_use", id=id_, name=name, input=inp)


@task("validate catches missing, unknown, mistyped and out-of-enum inputs")
def _() -> None:
    assert validate(DEVIG_SCHEMA, {"prices": [2.0, 3.5, 4.0]}) is None
    assert validate(DEVIG_SCHEMA, {}) is not None
    assert validate(DEVIG_SCHEMA, {"prices": [2.0], "stake": 10}) is not None
    assert validate(DEVIG_SCHEMA, {"prices": "2.0,3.5"}) is not None
    assert validate(DEVIG_SCHEMA, {"prices": [2.0, 2.0], "method": "magic"}) is not None


@task("one tool call, then an answer; the result goes back with the right id")
def _() -> None:
    m = ScriptedModel(
        [
            Response(
                "tool_use",
                [
                    Block("text", text="Let me de-vig."),
                    tool_use("t1", "devig_odds", prices=[2.0, 3.5, 4.0]),
                ],
            ),
            Response("end_turn", [Block("text", text="Home is 48.3%.")]),
        ]
    )
    text, trace = run_agent(m, TOOLS, "De-vig 2.0/3.5/4.0")
    assert text == "Home is 48.3%."
    assert trace == [("devig_odds", {"prices": [2.0, 3.5, 4.0]}, False)]
    last = m.seen[1][-1]
    assert last["role"] == "user"
    assert last["content"][0]["tool_use_id"] == "t1"
    close(json.loads(last["content"][0]["content"])["probabilities"][0], 0.4828, tol=1e-4)


@task("two tool calls in one turn → ONE user message with both results")
def _() -> None:
    m = ScriptedModel(
        [
            Response(
                "tool_use",
                [
                    tool_use("a", "devig_odds", prices=[2.0, 2.0]),
                    tool_use("b", "devig_odds", prices=[1.5, 3.0]),
                ],
            ),
            Response("end_turn", [Block("text", text="done")]),
        ]
    )
    run_agent(m, TOOLS, "two markets")
    results = m.seen[1][-1]["content"]
    assert [r["tool_use_id"] for r in results] == ["a", "b"]
    assert sum(1 for msg in m.seen[1] if msg["role"] == "user") == 2


@task("unknown tools and invalid inputs become is_error results, and the loop continues")
def _() -> None:
    m = ScriptedModel(
        [
            Response("tool_use", [tool_use("u", "place_bet", stake=100)]),
            Response("tool_use", [tool_use("v", "devig_odds")]),
            Response("end_turn", [Block("text", text="I can't place bets.")]),
        ]
    )
    text, trace = run_agent(m, TOOLS, "bet £100 on the home side")
    assert text == "I can't place bets."
    assert [t[2] for t in trace] == [True, True]
    assert m.seen[1][-1]["content"][0]["is_error"] is True


@task("refusal, truncation and the step limit are handled")
def _() -> None:
    assert run_agent(ScriptedModel([Response("refusal", [])]), TOOLS, "q")[0] == "[refused]"
    assert (
        run_agent(ScriptedModel([Response("max_tokens", [Block("text", text="…")])]), TOOLS, "q")[0]
        == "[truncated]"
    )
    looping = ScriptedModel([])  # always asks for another tool call
    text, _trace = run_agent(looping, TOOLS, "q", max_steps=4)
    assert text == "[step limit]"
    assert len(looping.seen) == 4


@task("grade_trajectory")
def _() -> None:
    ok = [("devig_odds", {"prices": [2, 2]}, True), ("devig_odds", {"prices": [2, 2]}, False)]
    assert grade_trajectory(ok, {"devig_odds"})
    assert not grade_trajectory(ok[:1], {"devig_odds"})
    assert not grade_trajectory(ok, {"devig_odds", "simulate_match"})


run(globals())
