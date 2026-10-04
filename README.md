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

**Phase 0 (collect and score)** is merged: FPL snapshot and post-gameweek collectors, an append-only bronze lake, versioned scoring rules with a vectorised engine, and golden tests.

**Phase 1 (lake, entities, harness)** is built:

- **backfills** for vaastav, football-data, Understat and The Odds API, the last within a monthly credit budget;
- **silver tables** with point-in-time observation times, pandera contracts and quality gates;
- **entity resolution** for teams, fixtures and players across sources;
- a **point-in-time feature spine** with CI-blocking leakage tests;
- **metrics**, gameweek-block bootstrap and Diebold–Mariano tests;
- a **reproducible walk-forward runner** and the A0 baseline.

Official FPL points are reproduced exactly for all 254,119 player-fixtures from 2016/17 to 2026/27.

Every Phase 1 gate passes on the full real data: 100 % player coverage across sources and exact
official points; see [the plan, §2.2](docs/IMPLEMENTATION_PLAN.md#22-results-on-real-data).

**Phase 2 (team level)** is built and evaluated walk-forward on 2022/23–2024/25:

- **de-vig** (multiplicative, power, Shin), with the default chosen by closing-price calibration;
- the **G0–G3 scoreline models** with market inversion;
- the **M1 dynamic team-strength filter** (goals + xG, Championship prior for promoted teams), with a NumPyro NUTS reference;
- **M3 fusion** of market and model rates.

xG improves M1 significantly. M1 alone reaches RPS 0.197 against the market's 0.194 at the deadline. The fused forecast ties the market on RPS (the exit gate, met as non-inferiority) and is best on scoreline log loss; see [the plan, §3.2](docs/IMPLEMENTATION_PLAN.md#32-results-on-real-data).

**Phase 3 (match and player level)** is built and passes its exit gate walk-forward on 2022/23–2024/25. It comprises:

- an **event timeline** from Understat rosters;
- the **G4–G6 in-match goal process** with an emulator;
- **component models:**
  - minutes (M4: calibrated XGBoost);
  - attack (M5/M6: shrunk Gamma–Poisson);
  - defensive actions (M7);
  - saves (M8);
  - cards (M9);
  - bonus (M10);
- a **vectorised player simulator** scoring with the official rules.

The simulator's MSE is 3.633, against 3.661 for an OpenFPL re-implementation, 4.015 for A0 and 4.354 for the last-5 average. Every gameweek-block CI excludes 0. P(60+) and P(haul) are calibrated to within 0.01. Minutes is the largest single contribution. See [the plan, §4.6](docs/IMPLEMENTATION_PLAN.md#46-the-exit-gate-the-player-simulator-walk-forward).

**Phase 4 (decisions)** is built: a multi-gameweek MILP for transfers, captaincy and chips, a season replay of every forecaster through the same optimiser, and a paper-only market ledger.
- **Replay totals over 2022/23–2024/25:** the simulator scores 6,956, against 6,699 for the OpenFPL replica, 6,220 for A0 and 6,000 for last-5. The simulator comes first in every season.
- **The exit gate fails:** the margin over the replica (+2.3 points per gameweek) has a CI that includes 0.
- **The ledger** shows no betting edge (CLV −1.2 %).

See [the plan, §5](docs/IMPLEMENTATION_PLAN.md#5-phase-4-decisions).

## Quickstart

Requires [uv](https://docs.astral.sh/uv/) (it installs Python 3.12 if needed).

```bash
uv sync                                     # install (add --extra s3 for bucket storage)
uv run pytest                               # unit, contract, property, offline golden tests

# golden tests on full seasons
uv run fplh golden fetch --season 2025/26 --season 2026/27
uv run fplh golden check --season 2025/26

# historical backfills → silver → evaluation
uv run fplh backfill vaastav            # also: football-data, understat (add --current for this season only)
uv run fplh silver build                # normalise, resolve entities, run quality gates, write
uv run fplh golden check-silver         # official points reproduced for every season
uv run fplh evaluate leakage --season 2024-25
uv run fplh evaluate a0 --season 2024-25

# Phase 2: team level (uv sync --extra bayes for the NUTS reference)
uv run fplh models fit-devig --before 2022-07-01     # writes configs/models/market.yaml
uv run fplh models fit-m1 --before 2022-07-01         # writes configs/models/team_strength.yaml (~70 min)
uv run fplh evaluate phase2 --season 2022-23 --season 2023-24 --season 2024-25

# Phase 3: match and player level
uv run fplh evaluate phase3-benchmarks --season 2022-23 --season 2023-24 --season 2024-25
uv run fplh evaluate g-ladder --season 2022-23 --season 2023-24 --season 2024-25   # writes goal_process.yaml
uv run fplh evaluate components --season 2022-23 --season 2023-24 --season 2024-25
uv run fplh evaluate phase3 --season 2022-23 --season 2023-24 --season 2024-25     # the exit gate (~80 min)
uv run fplh evaluate attribution --season 2022-23 --season 2023-24 --season 2024-25 --every 2
uv run fplh evaluate ep-next --season 2026-27                                     # live gameweeks with captures

# Phase 4: decisions
uv run fplh evaluate replay --season 2022-23 --season 2023-24 --season 2024-25    # the exit gate (~30 min cached)
uv run fplh evaluate ledger --season 2022-23 --season 2023-24 --season 2024-25    # paper only

# collect (writes to ./lake unless FPLH_LAKE_URI is set)
uv run fplh collect fpl-snapshot --with-fixtures
uv run fplh collect fpl-snapshot --only-within-hours 24   # store only near a deadline
uv run fplh collect fpl-post-gw                           # latest finalised gameweek, once
uv run fplh collect odds --due                            # needs FPLH_ODDS_API_KEY; ≤ 490 of 500 credits a month

# score any events CSV with a season's rules
uv run fplh rules score --season 2026/27 --csv events.csv --out points.csv
```

Settings are environment variables with the `FPLH_` prefix (see `src/fplh/settings.py`):

| Variable | Default | Meaning |
|---|---|---|
| `FPLH_LAKE_URI` | `lake` | Lake root: a local path or an fsspec URL such as `s3://statsnuke-lake` |
| `FPLH_USER_AGENT` | `StatsNuke-fplh/0.1 (+repo URL)` | Sent with every request |
| `FPLH_HTTP_TIMEOUT_S` | `30` | Per-request timeout |
| `FPLH_ODDS_API_KEY` | unset | The Odds API key (secret); odds collection is skipped without it |

Bucket credentials come from the standard `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY` and
`AWS_ENDPOINT_URL` variables. Never commit them.

## Layout

```text
configs/            rules/fpl_<season>.yaml (scoring rules as data), sources.yaml (endpoints, rate limits, lags)
src/fplh/           collectors/, lake/ (bronze, silver/, quality), entities/, features/, evaluate/, models/, rules/, cli.py
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
