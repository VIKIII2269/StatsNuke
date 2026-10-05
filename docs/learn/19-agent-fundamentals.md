# 19 · Agentic AI fundamentals

> **Goal:** understand what an "AI agent" is, mechanically: a language model in a loop, calling tools you define. Learn when to use one (and when not to), how the loop works, how to evaluate it, and how to keep it safe. Module 20 builds a real one on top of `fplh`.

## 1. What an LLM does

A large language model (LLM) such as Claude is trained to continue text. Given a conversation (messages), it produces the next message. Three properties shape everything below:

- **Stateless.** The API remembers nothing between calls. *You* send the whole conversation every time, so "memory" is whatever you put back in the messages.
- **Context window.** Everything it can consider must fit in one request: instructions, history, tool results. That is large (up to 1M tokens on current Claude models) but not free, because cost and latency grow with it.
- **Probabilistic and fallible.** It can be confidently wrong, especially about numbers it computes in its head. This is the main reason to give it **tools**.

## 2. Tool use: the model asks, your code acts

You describe tools to the model with a name, a description and a **JSON Schema** for the input:

```json
{"name": "devig_odds",
 "description": "Convert decimal odds for mutually exclusive outcomes into probabilities ...",
 "input_schema": {"type": "object",
                  "properties": {"prices": {"type": "array", "items": {"type": "number"}},
                                 "method": {"type": "string", "enum": ["multiplicative", "power", "shin"]}},
                  "required": ["prices"], "additionalProperties": false}}
```

When the model decides a tool would help, its response contains a **`tool_use` block** with an id, the tool name and the input. The API response's `stop_reason` is `"tool_use"`. **Your code** runs the function and sends back a **`tool_result`** block with the same id. The model never executes anything itself. It only *requests*, and your code decides whether to comply.

## 3. The agent loop

```text
messages = [user question]
loop:
    response = model(system, tools, messages)
    append response (all of its content blocks) to messages
    if response.stop_reason == "end_turn": return the final text
    if response.stop_reason == "tool_use":
        for each tool_use block: run the tool (or return an error result)
        append ONE user message holding all tool_result blocks
    otherwise (max_tokens, refusal, …): handle explicitly and stop
    stop after N iterations no matter what
```

This is the **ReAct** pattern (reason, act, observe, repeat). The rules that matter in practice:

- **Append the whole response.** Keep the assistant's content blocks exactly as returned, including any `thinking` blocks, not just its text. Current models expect the history to be append-only.
- **Return every result.** When the model makes several tool calls in one turn, return **all** the results together in **one** user message.
- **Report failures as data.** A failed tool returns a `tool_result` with `is_error: true` and a useful message, so the model can recover.
- **Check `stop_reason`** before reading the content. `"refusal"` and `"max_tokens"` need handling too.
- **Bound the loop.** Set a maximum number of iterations or tokens. Agents can loop.

In production you rarely hand-write this loop. SDK helpers such as the Python SDK's **Tool Runner** (`client.beta.messages.tool_runner`) drive it for you. Writing it once by hand (the practical) is how you understand what those helpers do.

## 4. Should it be an agent?

Use the simplest thing that works:

| Need | Use |
|---|---|
| One transformation (classify, summarise, extract) | A **single call** |
| A fixed sequence of steps you can write down | A **workflow**: your code calls the model at fixed points |
| Open-ended, multi-step tasks where the next step depends on what was found | An **agent** |

Check four things before choosing an agent:

- **complexity:** is the task hard to specify in advance?
- **value:** is the outcome worth the extra cost and latency?
- **viability:** can the model actually do this kind of task?
- **cost of error:** can mistakes be caught and undone?

For StatsNuke, *computing* the forecasts is not an agent job. The pipeline is deterministic, tested and gated (Modules 05–18). An agent fits *on top*: answering a manager's open questions ("should I captain Saka or Haaland given these odds?") by calling the trusted, tested functions.

## 5. Designing good tools

- **Tools do the math; the model does the language.** Never let the model compute a de-vigged probability in its head when `fplh.models.market.devig` exists. The tool is tested; the model's arithmetic is not.
- **Narrow, well-described tools.** One clear job each, with precise descriptions, units and ranges in the schema, and enums for choices. Descriptions are prompts.
- **Validate inputs.** Treat tool input like user input: check types, ranges and required fields, and return a clear error. Schemas with `additionalProperties: false` plus `strict: true` let the API enforce the shape.
- **Least privilege.** Read-only tools where possible. In StatsNuke there is no betting tool at all, so the agent *cannot* place a bet, whatever it is told. That mirrors the repo's own test that forbids write endpoints (Module 17).
- **Deterministic, small outputs.** Return compact JSON with the numbers, not walls of text. Context is a budget.

## 6. Prompt injection: tool results are data

Anything a tool returns, such as a web page, a news item or a player's "news" field, may contain text that *looks like* instructions ("ignore previous instructions and…"). The model must treat tool output as **data**, never as orders. Defences:

- say so in the system prompt;
- keep privileges low, so a successful injection can do little;
- require human confirmation for consequential actions;
- validate the outputs that drive actions.

## 7. Evaluating agents

An agent is a model plus a loop, so evaluate both:

| Level | Question | Example check |
|---|---|---|
| Final answer | Is the answer right? | Compare the stated probability to `devig()` within 0.005 |
| Trajectory | Were the right tools called with the right inputs? | `devig_odds` was called with the three prices from the question |
| Efficiency | How many steps and tokens? | ≤ 4 tool calls |
| Safety | Did it stay in bounds? | Refused to "place a bet"; no invented tools |

Build a **fixed eval set** of questions with known answers, run it on every change, and track the pass rate. This is exactly the walk-forward discipline of Module 07, applied to an agent. LLM outputs vary from run to run, so run each case several times and report rates, not anecdotes.

## 8. Common failure modes

| Failure | Mitigation |
|---|---|
| Hallucinated or invalid tool arguments | Schemas, validation, `is_error` results the model can fix |
| Doing math in its head instead of calling the tool | Instruct it to use tools for every number; evaluate the trajectory |
| Infinite or redundant loops | Iteration cap; tool results that say clearly when nothing more can be done |
| Context bloat from big tool outputs | Compact outputs; summarise; context editing or compaction for long sessions |
| Following injected instructions | "Tool output is data"; least privilege; human approval |
| Silent refusals or truncation | Check `stop_reason` (`refusal`, `max_tokens`) every turn |

## Check yourself

1. The model returns two `tool_use` blocks in one response. How do you send results back?
   <details><summary>Answer</summary>Run both, then append one user message containing both `tool_result` blocks, each with its matching `tool_use_id`.</details>
2. A tool raises an exception. What should the loop do?
   <details><summary>Answer</summary>Catch it and return a `tool_result` with `is_error: true` and a helpful message, so the model can correct its input or explain. Don't crash and don't drop the result.</details>
3. Why is the FPL agent unable to place bets even if a user insists?
   <details><summary>Answer</summary>No tool exists that could do it (least privilege). The model can only request tools you provide, and the repo itself contains no write endpoints.</details>

## Practical

```bash
uv run python docs/learn/exercises/ex19_agent_loop.py
```

Fully offline: you will write the agent loop against a scripted fake model. You will handle tool calls, unknown tools, invalid inputs, multiple calls per turn, refusals and the iteration cap, then write a trajectory grader.

Quiz: `uv run python docs/learn/quiz.py take 19`
