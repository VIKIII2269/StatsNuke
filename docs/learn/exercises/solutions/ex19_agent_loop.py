"""Reference solutions for exercise 19."""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

TYPES = {
    "number": (int, float),
    "integer": (int,),
    "string": (str,),
    "array": (list,),
    "object": (dict,),
}


def validate(schema: dict[str, Any], inp: dict[str, Any]) -> str | None:
    props = schema.get("properties", {})
    missing = [k for k in schema.get("required", []) if k not in inp]
    if missing:
        return f"missing required field(s): {missing}"
    if schema.get("additionalProperties") is False:
        extra = sorted(set(inp) - set(props))
        if extra:
            return f"unknown field(s): {extra}"
    for k, v in inp.items():
        spec = props.get(k, {})
        kind = spec.get("type")
        if (kind and not isinstance(v, TYPES[kind])) or (kind == "number" and isinstance(v, bool)):
            return f"{k} must be of type {kind}"
        if "enum" in spec and v not in spec["enum"]:
            return f"{k} must be one of {spec['enum']}"
    return None


def run_agent(
    model: Any,
    tools: dict[str, tuple[dict[str, Any], Callable[..., Any]]],
    question: str,
    max_steps: int = 6,
) -> tuple[str, list[tuple[str, dict[str, Any], bool]]]:
    specs = [{"name": n, "input_schema": s} for n, (s, _) in tools.items()]
    messages: list[dict[str, Any]] = [{"role": "user", "content": question}]
    trace: list[tuple[str, dict[str, Any], bool]] = []
    for _ in range(max_steps):
        r = model.create(messages, specs)
        messages.append({"role": "assistant", "content": r.content})
        if r.stop_reason == "end_turn":
            return "".join(b.text for b in r.content if b.type == "text"), trace
        if r.stop_reason == "refusal":
            return "[refused]", trace
        if r.stop_reason == "max_tokens":
            return "[truncated]", trace
        results = []
        for b in r.content:
            if b.type != "tool_use":
                continue
            error: str | None = None
            if b.name not in tools:
                error = f"unknown tool {b.name!r}; available: {sorted(tools)}"
            else:
                schema, fn = tools[b.name]
                error = validate(schema, b.input)
                if error is None:
                    try:
                        out = json.dumps(fn(**b.input))
                    except Exception as exc:  # report to the model, don't crash
                        error = f"{type(exc).__name__}: {exc}"
            trace.append((b.name, b.input, error is not None))
            if error is None:
                results.append({"type": "tool_result", "tool_use_id": b.id, "content": out})
            else:
                results.append(
                    {"type": "tool_result", "tool_use_id": b.id, "content": error, "is_error": True}
                )
        messages.append({"role": "user", "content": results})
    return "[step limit]", trace


def grade_trajectory(trace: list[tuple[str, dict[str, Any], bool]], required: set[str]) -> bool:
    ok = {name for name, _, err in trace if not err}
    return required <= ok
