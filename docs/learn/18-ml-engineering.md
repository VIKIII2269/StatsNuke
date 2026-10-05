# 18 · ML engineering practices

> **Goal:** collect the engineering habits that make StatsNuke trustworthy. Math alone doesn't get you there: you also need reproducible runs, configuration as data, a test pyramid, polite and robust data collection, CI, and a delivery process with gates. These are what turn a notebook into a system.

## 1. Reproducibility: same inputs → same bytes

A result you can't reproduce is an anecdote. `src/fplh/evaluate/manifest.py` pins everything a prediction depended on:

| Field | What it pins |
|---|---|
| `git_sha` | The code |
| `data_manifest_sha256` | SHA-256 over every silver Parquet part read (keys + content hashes) |
| `config_sha256` | Every YAML in `configs/` |
| `predictor`, `predictor_version` | The model and its behaviour version |
| `deadlines`, `horizon`, `seeds` | What was asked, with which randomness |

The **run id is derived from the content**: a SHA-256 of those fields, sorted-key JSON, first 12 hex characters (`Manifest.run_id`, `src/fplh/evaluate/manifest.py:70`). Replaying identical inputs gives the *same id* and **byte-identical outputs**. Nothing wall-clock enters the outputs, and Phase 1 verified that silver rebuilds are byte-identical across 163 parts.

**Caching follows for free.** `cached_walk_forward` (`evaluate/walk_forward.py:122`) reuses a stored run whose manifest matches. The 80-minute exit gate reruns in seconds. The catch is in the docstring: the git SHA is ignored, so *you must bump `version` when a predictor's behaviour changes*. It is a deliberate trade-off between correctness and convenience, and it is written down.

## 2. Configuration as data, with provenance

Rules (`configs/rules/*.yaml`), sources (`sources.yaml`), entity overrides and **fitted model parameters** (`configs/models/*.yaml`) are data files. Look at `configs/models/team_strength.yaml`: the header records *the command that produced it*, the training window, the objective values of alternatives, and observations ("sigma_a sits at its lower bound"). A parameter file that explains itself is reviewable in a PR.

Settings come from environment variables with the `FPLH_` prefix (`settings.py`, pydantic-settings). Secrets such as `FPLH_ODDS_API_KEY` and bucket credentials are **never committed**. Collection is skipped when the key is blank (`git log`: "Skip odds collection when the API key secret is blank").

## 3. The test pyramid

| Layer | Purpose | Example |
|---|---|---|
| **Unit** | A function does what you meant | `tests/unit/test_market.py` |
| **Contract** | External APIs still send the shape we parse | `tests/contracts/` with saved (trimmed real) payloads |
| **Mocked HTTP** | Collectors behave on 200/429/500/timeouts without the network | `respx` in `tests/unit/test_http.py` |
| **Property** | Invariants for *all* inputs | Hypothesis: rules order-independence, simulator coherence |
| **Golden** | Exact agreement with reality | 254,119 official points |
| **Leakage** | No future information, ever | future-shuffle, CI-blocking |
| **Integration** | The pipeline end to end on small data | walk-forward on `tests/synthetic_silver.py` |
| **Brute force** | Optimisers return the true optimum | `brute_force_single_week` |

There are also **tests of tests**: a deliberately leaky builder must be caught, and each quality gate has a test that corrupts one fixture.

**Policy as tests.** "No code path can place a bet" is not a promise in a README. It is a test that fails on any HTTP write verb in `src/` (`tests/unit/test_ledger.py`).

## 4. Static checks and CI

- **uv** with a lockfile (`uv.lock`) gives the exact same environment everywhere: `uv sync --locked` in CI.
- **ruff** for lint and formatting, including Python code blocks in Markdown. **mypy --strict** on `src/`. Optional **pre-commit** hooks run the same checks locally.
- **CI** (`.github/workflows/ci.yml`) runs lint → format check → typecheck → fetch full-season golden data → all tests, on every push.

The checks that run locally are the same ones that run in CI, so a green local run predicts a green CI.

## 5. Polite, robust data collection

`src/fplh/collectors/http.py`:

- an **identifying User-Agent** with a repo URL, so site owners know who is calling;
- a **shared rate limiter** with minimum spacing between requests across all concurrent tasks (`RateLimiter`, `src/fplh/collectors/http.py:47`);
- **retries with exponential backoff and jitter** on 408/425/429/5xx and transport errors. The wait after attempt n is min(max, base·2^(n−1) + U(0, base)) (`_wait`, `src/fplh/collectors/http.py:107`). **`Retry-After`** is honoured. Other 4xx responses are returned, not retried: a 404 won't fix itself.

The jitter matters. If many clients retry on the same schedule, they hammer the server in synchronised waves (a "thundering herd").

**Partial failures persist what succeeded and exit non-zero**, so you don't lose 799 good downloads because 1 failed, and you still notice the failure.

**Budgets with hard guarantees.** The Odds API free plan gives 500 credits a month. `collectors/odds_budget.py` plans the month by priority: closing odds, then pre-lineup, deadline, props, spare. It re-reads the *live* remaining credits before **every** call and refuses any call that would leave fewer than 10. Simulated on real months, runs spend 489–490 credits and never cross the floor, even with half the hourly runs missed.

## 6. Scheduling and storage in the real world

`.github/workflows/collect.yml` is a **stopgap scheduler**. Cron runs every 3 hours, hourly near deadlines, and once a day after lockdown. Where no bucket is configured, data are committed to a separate `data-bronze` branch. The plan quantifies the cost (≈ 0.7–1 GB per season bloating clones) and the exit (move to an S3-compatible bucket, then a VPS with Dagster in Phase 6). "Good enough now, with a documented path to proper" is a valid engineering decision when it is made consciously.

## 7. Process: phases, gates, ablations, honesty

- **Spec gaps are written down** (`IMPLEMENTATION_PLAN.md` §1). Each is a place where the design was ambiguous or infeasible, with its resolution. For example, train/serve skew in minutes, and the odds budget.
- **Exit gates are pre-registered** and judged on real data with confidence intervals (Module 07).
- **Ablation ladders** (G0→G6, A1–A5) test every added complexity, and most steps fail. The simpler model is kept.
- **Changes made after seeing results are disclosed** with both runs (Module 11).
- **Failures are reported** in the README ("the exit gate fails"). That is what makes the passes credible.
- **Delivered as small PRs**, each merged when green. Phase 3 was six PRs.

## 8. What you would add next (Phase 6)

Monitoring of live calibration, heartbeats for scheduled jobs (Healthchecks.io), alerts on gate or freshness failures, and an orchestrator with assets partitioned by gameweek. The exit criterion: stable calibration across 10+ live gameweeks.

## Check yourself

1. Why derive the run id from content rather than a timestamp?
   <details><summary>Answer</summary>Identical inputs then give the same id, which enables caching and proves reproducibility. A timestamp would make every run look new and hide whether the outputs changed.</details>
2. Why add random jitter to retry backoff?
   <details><summary>Answer</summary>To desynchronise clients. Without it, many retries hit the server at the same moments and can keep it overloaded.</details>
3. What is the risk of `cached_walk_forward` ignoring the git SHA, and how is it managed?
   <details><summary>Answer</summary>A code change that alters a predictor's behaviour would silently reuse stale results. It is managed by convention, with the rule documented in the docstring: bump the predictor's `version` whenever its behaviour changes.</details>

## Practical

```bash
uv run python docs/learn/exercises/ex18_engineering.py
```

You will reproduce `Manifest.run_id`, show the config hash is order-independent, implement the backoff schedule, and write a property check that would catch a non-deterministic predictor.

Quiz: `uv run python docs/learn/quiz.py take 18`
