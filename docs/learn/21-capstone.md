# 21 · Capstone: final exam and build projects

> **Goal:** show you can use what you've learned *together*. The capstone has three parts: a timed mixed exam drawn from every module, an oral-style "explain the system" check, and three build projects that extend StatsNuke the way its own roadmap (Phase 5) would.

## 1. The final exam

```bash
uv run python docs/learn/quiz.py exam                 # 2 questions per module (44), 60 minutes
uv run python docs/learn/quiz.py exam --per-module 3 --minutes 90
```

It draws questions at random from every module's bank, plus this module's integrative ones. When the time runs out, unanswered questions score 0. The score is recorded as module `exam`, and `quiz.py report` shows your attempts and best result. **Pass mark: 80 %.** Revise with `quiz.py review` first; it prioritises everything you have missed.

## 2. Explain the system (self-check, or ask Claude to examine you)

Answer each question out loud or in writing in about 3 minutes, then compare with the referenced module. Ask Claude to "examine me on the capstone oral" and it will grade your answers and record them with `quiz.py record`.

1. Walk a single forecast from raw bytes to a captain choice. Name every layer, and the guarantee each one provides (Modules 05 → 16).
2. Why does every fact carry `observed_at`, and what three mechanisms prove no model sees the future? (05)
3. Derive the Gamma–Poisson posterior mean and explain the method-of-moments prior strength. (04)
4. Explain how M1 updates after a match: the Laplace step in η-space, then conditioning the full state. Why does one match move *other* teams' ratings? (10)
5. Why does the simulator allocate team goals to players instead of simulating each player's goals? (14, 15)
6. Phase 3 passed and Phase 4 failed. Explain both results statistically, including power. (07, 15, 17)
7. Which single component would you improve next, and why? Use the error attribution numbers. (13, 15)
8. Design an agent tool and an eval for it. Where could the agent leak future data, and how do you stop it? (19, 20)

## 3. Build projects

Each project has a clear acceptance test, in the repo's own style. Work on a branch, keep tests green (`uv run pytest`, `uv run ruff check .`, `uv run mypy`), and write down results honestly, including failures.

### Project 1: a CRPS comparison for the points pmf

The simulator outputs a whole points pmf (−4…25), but its gate compares only MSE of the mean.

- Write `crps_comparison(pmf_a, pmf_b, y, block)` that computes per-row discrete CRPS for two forecasters (reuse `fplh.evaluate.metrics.crps_discrete`, after shifting the support to start at 0) and compares them with `fplh.evaluate.bootstrap.compare`.
- **Acceptance:**
  - a unit test where a sharper, correct pmf beats a flat one with a CI excluding 0;
  - a test where two identical pmfs give a CI containing 0.
- *Stretch:* add a randomised-PIT histogram check.

### Project 2: a rank-aware objective in the MILP

ARCHITECTURE §10.2 proposes valuing players relative to the field: v_{p,t} = E_{p,t}·(1 − EO_{p,t}), where EO is effective ownership. That rewards differentials when chasing rank.

- Add an optional `ownership` input to `optimise`, and an `objective="rank"` mode that replaces E by v in the XI and captain terms.
- **Acceptance:**
  - a toy test (like `tests/unit/test_milp.py`) where two players have equal E but different EO, and the rank objective picks the low-EO one while EV mode is indifferent;
  - brute-force agreement on a tiny instance.
- *Stretch:* replay a season with a synthetic EO and report the change in points (expect a cost in EV).

### Project 3: a new agent tool with an eval set

- Add `simulate_match` (or `compare_captains`) to `docs/learn/agent/fpl_agent.py`, with a schema, validation and a docstring that says which rates it expects (nominal vs mean goals, Module 12).
- Write `docs/learn/agent/evals.yaml` with at least 8 cases: answer cases, trajectory cases, and two safety cases (a bet request and an injected instruction in a tool result).
- Write a runner that, with credentials available, runs each case 3 times and prints pass rates per category (reuse `extract_number` and `pass_rates` from the Module 20 practical).
- **Acceptance:** the offline tests pass (tool output matches a direct `fplh` call; the validator rejects bad inputs), and with credentials, a report of pass rates.

## 4. What "done" looks like

Run `uv run python docs/learn/quiz.py report`. Done means:

- every module mastered (best quiz ≥ 80 % and the practical passing);
- exam ≥ 80 %;
- at least one project merged with tests;
- a course score near 100.

At that point you can rebuild StatsNuke, and more importantly, you can tell when a model is genuinely better and when it only looks that way.

Quiz (integrative questions): `uv run python docs/learn/quiz.py take 21`
