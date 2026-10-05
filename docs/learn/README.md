# Learn StatsNuke: the math, the models, the engineering and agents

This course teaches you everything you need to build this repository from scratch. It assumes you can write basic Python and nothing else. Each module covers four things:

1. the **problem** the piece of StatsNuke solves;
2. the **math**, built up from definitions with worked numbers;
3. the **real code**, with links to the lines that implement it;
4. **why** it is built this way, and what the real-data results said.

Every module ends with a **graded quiz**, and most also have a **coding practical**. Your scores are recorded, so you can see what you have mastered and what to revise.

## How to study

```bash
uv sync                                                  # once: installs fplh and its dependencies
uv run python docs/learn/quiz.py list                    # modules and your mastery
# for each module, in order:
#   1. read docs/learn/NN-*.md and answer its "Check yourself" questions as you go
#   2. solve the practical:  uv run python docs/learn/exercises/exNN_*.py
#   3. take the quiz:        uv run python docs/learn/quiz.py take NN
uv run python docs/learn/quiz.py practicals              # records which practicals pass
uv run python docs/learn/quiz.py report                  # scores, weakest topics, what next
uv run python docs/learn/quiz.py review                  # spaced repetition of what you missed
uv run python docs/learn/quiz.py exam                    # the final exam (module 21)
```

**Mastery.** A module counts as mastered when your best quiz score is at least **80 %** and its practical passes. Questions are weighted by difficulty (1–3 points). Only attempts that cover at least 60 % of a module's questions count toward mastery; shorter quizzes still feed review and the weak-topic statistics.

**Course score.** The course score out of 100 is 70 % the average of your best quiz scores and 30 % the share of practicals that pass.

**Score history.** Scores go to `progress/scores.csv`. Per-question history, which drives spaced repetition, goes to `progress/questions.json`. Commit both files to keep your record.

**Learning with Claude.** You can also study here: ask Claude to "quiz me on module 7". Claude asks questions from the same bank, grades your answers (free-text ones included), and records them with `quiz.py record`, so chat quizzes count too.

**Practicals.** Each practical prints `PASS`, `TODO` or `FAIL` per task, and passes only when every task passes. The reference solutions are in `exercises/solutions/`. Try the tasks before you look.

## Roadmap

| # | Module | You will learn | Code it unlocks |
|---|---|---|---|
| **A** | **Foundations** | | |
| 00 | [Tour of the system](00-tour.md) | What StatsNuke does end to end; FPL rules; phases and exit gates | the whole repo |
| 01 | [Python, NumPy and pandas the repo's way](01-python-numpy-pandas.md) | Vectorisation, broadcasting, dataclasses, Protocols, typing | `sim/`, `evaluate/walk_forward.py` |
| 02 | [Probability from zero](02-probability.md) | Random variables, expectation, variance; the distributions used here (Bernoulli → NegBin); logit and sigmoid | everything |
| 03 | [Likelihood and optimisation](03-likelihood-and-optimisation.md) | Likelihood, log-likelihood, MLE, derivatives and gradients, Newton, L-BFGS-B, ridge, overfitting | every `fit` |
| 04 | [Bayes and shrinkage](04-bayes-and-shrinkage.md) | Prior → posterior, conjugacy, Gamma–Poisson, method of moments, time decay | `models/shrinkage.py` |
| **B** | **Data and evaluation** | | |
| 05 | [The data platform](05-data-platform.md) | Bronze/silver, bitemporal facts, information sets, leakage tests, entity resolution, contracts | `lake/`, `features/`, `entities/` |
| 06 | [Rules as data](06-rules-engine.md) | YAML rules, the vectorised engine, bonus ties, golden tests, property tests | `rules/` |
| 07 | [Evaluating forecasts](07-evaluation.md) | Log loss, Brier, RPS, ECE, PIT, CRPS; walk-forward; block bootstrap; Diebold–Mariano | `evaluate/metrics.py`, `bootstrap.py` |
| **C** | **Team level** | | |
| 08 | [Betting markets and de-vig](08-markets-devig.md) | Odds, overround, multiplicative/power/Shin, favourite–longshot bias, market inversion | `models/market.py` |
| 09 | [Scoreline models G0–G3](09-scorelines.md) | Poisson grids, Dixon–Coles, bivariate Poisson, COM-Poisson, markets from a grid | `models/goal_benchmarks.py` |
| 10 | [Dynamic team strength (M1)](10-team-strength.md) | State-space models, random walks, the Kalman idea, the Laplace update, xG likelihood, MCMC/NUTS | `models/team_strength.py` |
| 11 | [Fusion (M3)](11-fusion.md) | Log-linear pooling, the sigmoid weight, cross-validated ridge | `models/fusion.py` |
| **D** | **Match and player level** | | |
| 12 | [The in-match goal process (G4–G6)](12-in-match-process.md) | Hazards, piecewise-exponential Poisson GLMs, offsets, frailty, dispersion, emulators | `models/goal_process.py`, `sim/team.py`, `sim/emulator.py` |
| 13 | [Minutes with XGBoost (M4)](13-minutes-xgboost.md) | Trees, gradient boosting, monotone constraints, isotonic calibration, staged models | `models/minutes.py` |
| 14 | [Component models (M5–M10)](14-components.md) | Attack, defence (NegBin thresholds), saves, cards, bonus | `models/attack.py` … `bonus.py` |
| 15 | [Monte Carlo simulation](15-monte-carlo.md) | Simulation, Monte Carlo error, common random numbers, systematic sampling, coherence | `sim/simulator.py` |
| **E** | **Decisions and engineering** | | |
| 16 | [Optimisation with MILP](16-milp.md) | LP → integer programming, binary tricks, the FPL squad model | `optimize/milp.py` |
| 17 | [Season replay and the paper ledger](17-replay-and-ledger.md) | Replay as the real test, EV, Kelly, closing-line value | `evaluate/replay.py`, `delivery/ledger.py` |
| 18 | [ML engineering practices](18-ml-engineering.md) | Reproducibility, manifests, caching, CI, the test pyramid, exit gates | `evaluate/manifest.py`, `.github/` |
| **F** | **Agentic AI** | | |
| 19 | [Agent fundamentals](19-agent-fundamentals.md) | LLMs, tool use, the agent loop, context and memory, evals, guardrails | `exercises/ex19` |
| 20 | [Build an FPL agent](20-fpl-agent.md) | Tools over `fplh`, the Claude tool-use loop, evaluating an agent, safety | `agent/fpl_agent.py` |
| 21 | [Capstone](21-capstone.md) | The final exam and three build projects | everything |

Study Part A first. B and C can be read in either order, and D needs C. Module 16 needs only Parts A and B. Part F needs Module 00 and any one of C, D or E.

```text
00 → 01 → 02 → 03 → 04 ─┬─> 05 → 06 → 07 ─┬─> 08 → 09 → 10 → 11 → 12 → 13 → 14 → 15 ─┐
                        │                   └─> 16 → 17 ─────────────────────────────── ├─> 18 → 21
                        └──────────────────────────────────────────> 19 → 20 ──────────┘
```

## Files

```text
docs/learn/
  NN-*.md                 lessons
  quiz.py                 quiz runner and score tracker
  quizzes/NN_*.yaml       graded question banks (answers and explanations inside)
  exercises/exNN_*.py     practicals; solutions/ holds the reference answers
  agent/fpl_agent.py      the reference FPL agent (module 20)
  progress/               your scores (commit these)
  tests/                  tests for the quiz engine and question banks
```
