# StatsNuke

Calibrated probabilistic forecasts for English Premier League matches and players, turned into
**Fantasy Premier League decisions** and a **paper-only market comparison ledger**.

- **Design:** [`ARCHITECTURE.md`](ARCHITECTURE.md), the market-anchored hierarchical generative model, data
  platform, simulator, rules engine, optimizer and evaluation protocol.
- **Plan and status:** [`docs/IMPLEMENTATION_PLAN.md`](docs/IMPLEMENTATION_PLAN.md), with the phased tickets, spec gaps and
  going-live checklist.

> **Paper only.** No code path places bets or moves money. Real-money online gaming is prohibited for the
> operator's jurisdiction (ARCHITECTURE.md §2.2); betting prices are used purely as a forecasting benchmark.
> FPL is free to play.

## Status

Phase 0 (collect and score) is built:

- FPL snapshot and post-gameweek collectors writing an append-only **bronze** lake (local directory or any
  fsspec URL, e.g. an R2/B2 bucket);
- versioned per-season **scoring rules** with a vectorised rules engine and official bonus tie rule;
- **golden tests** reproducing official FPL points exactly: all 29,747 player-fixtures of 2025/26 and every
  finalised 2026/27 row available so far;
- a GitHub Actions **stopgap scheduler**. It goes live once merged to `main`; see the
  [going-live checklist](docs/IMPLEMENTATION_PLAN.md#04-going-live-actions-for-the-repo-owner).

## Quickstart

Requires [uv](https://docs.astral.sh/uv/) (it installs Python 3.12 if needed).

```bash
uv sync                                     # install (add --extra s3 for bucket storage)
uv run pytest                               # unit, contract, property, offline golden tests

# golden tests on full seasons
uv run fplh golden fetch --season 2025/26 --season 2026/27
uv run fplh golden check --season 2025/26

# collect (writes to ./lake unless FPLH_LAKE_URI is set)
uv run fplh collect fpl-snapshot --with-fixtures
uv run fplh collect fpl-snapshot --only-within-hours 24   # store only near a deadline
uv run fplh collect fpl-post-gw                           # latest finalised gameweek, once

# score any events CSV with a season's rules
uv run fplh rules score --season 2026/27 --csv events.csv --out points.csv
```

Settings are environment variables with the `FPLH_` prefix (see `src/fplh/settings.py`):

| Variable | Default | Meaning |
|---|---|---|
| `FPLH_LAKE_URI` | `lake` | Lake root: a local path or an fsspec URL such as `s3://statsnuke-lake` |
| `FPLH_USER_AGENT` | `StatsNuke-fplh/0.1 (+repo URL)` | Sent with every request |
| `FPLH_HTTP_TIMEOUT_S` | `30` | Per-request timeout |

Bucket credentials come from the standard `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY` and
`AWS_ENDPOINT_URL` variables. Never commit them.

## Layout

```text
configs/            rules/fpl_<season>.yaml (scoring rules as data), sources.yaml (endpoints, rate limits, lags)
src/fplh/           collectors/, lake/, rules/, cli.py; later phases add entities/, features/, models/, sim/, optimize/, evaluate/
tests/              unit/, contracts/ (saved payloads), golden/ (official points), property/ (hypothesis)
scripts/            golden sample builder; data-branch persistence for the stopgap scheduler
.github/workflows/  ci.yml (lint, types, tests), collect.yml (scheduled collection)
```

## Development

```bash
uv run ruff check . && uv run ruff format --check .
uv run mypy
uv run pre-commit install    # optional: run the same checks on commit
```
