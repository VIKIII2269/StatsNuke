# 00 · A tour of StatsNuke

> **Goal of this module:** a mental map of the whole system. By the end you should be able to say what each folder does, why the phases exist, and what "better than the baseline" means here.

## 1. The problem in one paragraph

[Fantasy Premier League](https://fantasy.premierleague.com) (FPL) is a free game:

- You pick **15 real players** within a **£100m** budget: 2 goalkeepers, 5 defenders, 5 midfielders and 3 forwards, with at most 3 from any one club.
- Every gameweek you choose a **starting XI** and a **captain**, whose points are doubled.
- Your players score points from what they really do on the pitch: minutes played, goals, assists, clean sheets, saves, bonus.
- You may make **transfers**. Each one beyond your free transfers costs 4 points (a "hit").
- Four **chips** give a one-off boost: wildcard, free hit, bench boost and triple captain.

To play well you need two things:

1. a good **forecast** of each player's points, over several gameweeks ahead;
2. a good **decision**: the best squad, XI, captain, transfers and chip timing, given that forecast.

StatsNuke does both, and checks itself against honest baselines at every step.

> **Paper only.** The system also compares its match probabilities with bookmakers' prices, because betting markets are excellent forecasters and make a tough benchmark. **No code can place a bet** (a test enforces it). Money never moves.

## 2. The pipeline

```text
 COLLECT            STORE                 MODEL                       SIMULATE              DECIDE           CHECK
 ───────            ─────                 ─────                       ────────              ──────           ─────
 FPL API      ┐                     ┌─ M1 team strength ┐
 vaastav      │     bronze (raw,    ├─ M2 market (odds) ├─ M3 fusion ─┐
 Understat    ├──>  immutable) ──>  │                   ┘             │
 football-data│     silver (clean,  ├─ G4–G6 in-match goal process ───┼─> player simulator ─> MILP optimiser ─> season replay
 Odds API     ┘     point-in-time)  ├─ M4 minutes (XGBoost)           │   (Monte Carlo,         (squad, XI,       paper ledger
                                    └─ M5–M10 attack, defence,        │    official rules)      captain, chips)   metrics, CIs
                                       saves, cards, bonus ───────────┘
```

Read it left to right:

1. **Collect** (`src/fplh/collectors/`). Polite HTTP clients download raw data from five sources.
2. **Store** (`src/fplh/lake/`). Raw bytes go to **bronze**, which is never edited. Cleaned, validated tables go to **silver**, where every fact carries the time it became *knowable* (`observed_at`).
3. **Model** (`src/fplh/models/`). Each model predicts one real quantity: team scoring rates, minutes, shots, saves and so on.
4. **Simulate** (`src/fplh/sim/`). The models feed a Monte Carlo match simulator. It plays each fixture thousands of times, minute by minute, and scores every simulated match with the *official* FPL rules.
5. **Decide** (`src/fplh/optimize/`). A mixed-integer linear program (MILP) picks the squad, XI, captain, transfers and chips that maximise expected points over the next few gameweeks.
6. **Check** (`src/fplh/evaluate/`, `src/fplh/delivery/`). Walk-forward evaluation, a full season replay, and a paper betting ledger measure whether any of this beats simpler methods.

## 3. Why it is built in phases with exit gates

The project is built in phases. **A phase is done when its exit gate passes, not when a date arrives** (`docs/IMPLEMENTATION_PLAN.md`):

| Phase | Theme | Exit gate | Result |
|---|---|---|---|
| 0 | Collect and score | The rules engine reproduces official points exactly | 0 mismatches on 254,119 player-fixtures |
| 1 | Lake, entities, harness | Coverage, no leakage, A0 baseline logged | Met |
| 2 | Team level | The fused forecast is at least as good as the market | Met as *non-inferiority* (a tie) |
| 3 | Player level | The simulator beats the OpenFPL replica and the naive floors | Met: MSE 3.633 vs 3.661 |
| 4 | Decisions | The season replay beats the strongest baseline, CI excluding 0 | **Failed**: +2.3 pts/GW, CI [−1.3, +6.1] |

Two lessons are hidden in that table:

- **Gates are honest.** Phase 4 failed, and the repo says so plainly. The simulator *did* come first in every season, but the confidence interval included zero, so the gain is not proven. You will learn why in Modules 07 and 17.
- **Each phase builds a floor for the next.** You cannot evaluate a player model without leak-free data (Phase 1). You cannot simulate players without team scoring rates (Phase 2).

## 4. The baselines (the bars to beat)

A model is only "good" relative to something. StatsNuke always compares against:

| Name | What it is | Why it matters |
|---|---|---|
| **Last 5** | A player's average points over their last 5 matches | The naive floor; anything worse is broken |
| **A0** | Season per-90 rates × naive minutes, with market clean-sheet odds | A sensible, cheap baseline |
| **OpenFPL replica** | XGBoost on 211 rolling features, re-implemented here | A strong published ML approach; the real bar |
| **The market** | Bookmaker prices with the margin removed | The best public match forecaster there is |

## 5. The repository layout

```text
src/fplh/
  collectors/   HTTP clients and source adapters (FPL, vaastav, Understat, football-data, Odds API)
  lake/         bronze (raw) and silver (clean) storage, data-quality gates
  entities/     one identity per team, fixture and player across sources
  features/     the information set 𝓘(D), the spine, leakage checks, feature builders
  rules/        FPL scoring rules as data (YAML) and a vectorised scoring engine
  models/       M1–M10, the G-ladder, baselines, fusion, shrinkage
  sim/          team-only and player-level Monte Carlo simulators, the emulator
  optimize/     the multi-gameweek MILP
  evaluate/     metrics, bootstrap, walk-forward runner, manifests, replay, phase gates
  delivery/     the paper ledger
  cli.py        `fplh …` commands that run everything
configs/        rules per season, model parameters, sources, entity overrides
tests/          unit, contract, property (hypothesis), golden, leakage tests
```

Spend ten minutes opening a few files. Notice that **every module starts with a docstring** explaining what it does and why, often quoting a section of `ARCHITECTURE.md`. That habit is part of the engineering you are learning.

## 6. Vocabulary you will meet everywhere

| Term | Meaning |
|---|---|
| **Deadline D** | The moment a gameweek's teams lock: 90 minutes before its first kickoff. Every forecast is made "as of" a deadline. |
| **Information set 𝓘(D)** | Every fact with `observed_at ≤ D`, meaning everything a forecaster could have known then. |
| **Leakage** | Using information from after D. It produces fake, too-good results. |
| **Walk-forward** | Evaluate by stepping through deadlines in time order, training only on the past at each one. |
| **Horizon h** | How many gameweeks ahead a forecast is for (h = 1 is the next gameweek). |
| **xG** | Expected goals: the probability that a shot becomes a goal, summed over shots. |
| **Calibrated** | When you say 30 %, it happens about 30 % of the time. |
| **Exit gate** | A pre-registered test a phase must pass before the next one starts. |

## 7. How results are reported (a preview)

You will see lines like this throughout the plan:

> Simulator − OpenFPL replica: MSE difference **−0.029**, 95 % CI **[−0.048, −0.011]**, DM p = 0.003

They mean:

- *Difference −0.029:* the simulator's mean squared error is lower (lower is better).
- *95 % CI:* resampling whole gameweeks, the difference is between −0.048 and −0.011 in 95 % of resamples. The whole interval is below 0, so the improvement is unlikely to be luck.
- *DM p = 0.003:* a Diebold–Mariano test says a difference this large would happen by chance about 0.3 % of the time if the models were equally good.

Module 07 derives all of this.

## Check yourself

1. Why does StatsNuke store an `observed_at` time on every fact?
   <details><summary>Answer</summary>So that a forecast at deadline D uses only facts with observed_at ≤ D. Without it you cannot replay history honestly, and walk-forward results would leak the future.</details>
2. The Phase 4 replay scored 6,956 points against the replica's 6,699. Why is the gate marked *failed*?
   <details><summary>Answer</summary>The gate asks for a confidence interval that excludes 0. The per-gameweek margin of +2.34 has 95 % CI [−1.26, +6.05], which includes 0. With noisy gameweek scores, three seasons cannot rule out that the gap is luck.</details>
3. Name the four baselines from weakest to strongest.
   <details><summary>Answer</summary>Last 5 → A0 → OpenFPL replica, and at match level, the market.</details>

## Practical

None for this module. Take the quiz:

```bash
uv run python docs/learn/quiz.py take 00
```
