# 05 · The data platform: lakes, time and leakage

> **Goal:** understand why most of the effort in a forecasting system goes into data. Learn how StatsNuke stores raw data so it can always be rebuilt, how every fact gets a "when could we have known this" time, and how the code *proves* no model ever sees the future.

## 1. The problem: honest history is hard

Suppose you want to know how well a model would have predicted gameweek 10 of 2023/24. You must give it **exactly** what a person could have known 90 minutes before that gameweek's first kickoff. That rules out:

- the final scores of gameweek 10 (obviously);
- an end-of-season player registration table (it says which club a player *ended* at);
- a vaastav column like `xP` that was computed after the matches (`configs/sources.yaml` warns about it);
- closing odds, which are only known at kickoff;
- a stats site's numbers for gameweek 9, if the site publishes 24 hours after the match and the deadline is 20 hours after it.

Breaking any of these rules is **leakage**. It makes a backtest look better than reality, sometimes dramatically. Once you trust a leaky backtest, every later decision is built on sand.

## 2. Storage layers: bronze, silver, gold

| Layer | What | Rule | Code |
|---|---|---|---|
| **Bronze** | Every API response or file download, byte-for-byte, gzip-compressed, plus a `.meta.json` sidecar (URL, params, HTTP status, `observed_at`, SHA-256) | **Create-only.** Nothing can overwrite or delete it. | `lake/bronze.py` |
| **Silver** | Typed Parquet tables: normalised, validated and entity-resolved | **Rebuildable** from bronze at any time; rebuilds are byte-identical | `lake/silver/` |
| **Gold** | Predictions, cached emulators, evaluation outputs | Rebuildable from silver plus model code | `evaluate/`, `sim/emulator.py` |

Why keep raw bytes forever?

1. **Bugs in parsing are inevitable.** With bronze intact, you fix the parser and rebuild silver. Without it, the data are gone.
2. **Some data exist only "now".** FPL's live player status, news and prices are *current state*. If you don't snapshot them every few hours, that history is lost forever. That is why Phase 0, collection, came first.
3. **Auditability.** The SHA-256 is verified on read, so a corrupted object is detected, not silently used.

Bronze keys look like `bronze/source=fpl/endpoint=bootstrap-static/dt=2026-09-29/obs=2026-09-29T06:00:02Z.json.gz`. The `key=value` folder names are **Hive-style partitions**: tools can filter by source or date without opening files.

## 3. Bitemporal facts: event time vs observation time

Every fact f carries two times (ARCHITECTURE §6.2):

- **e_f, event time:** when it happened, such as the kickoff.
- **o_f, observation time (`observed_at`):** when the system *could first have known* it.

For data the collector polls live, o_f is the fetch time. For **backfilled history**, where you download five years of CSVs today, the real fetch time is useless, so the repo uses a conservative **publication lag** per source (`configs/sources.yaml`):

$$o_f = e_f + \ell_s,\qquad \ell_{\text{FPL}} = 33\text{ h},\quad \ell_{\text{Understat}} = 24\text{ h},\quad \ell_{\text{football-data}} = 48\text{ h}$$

The lags are deliberately *conservative*. If you assume data arrived later than it really did, you only make the forecast slightly worse. If you assume it arrived earlier, you leak.

## 4. The information set 𝓘(D)

$$\mathcal I(D) = \{f : o_f \le D\}$$

`InformationSet` (`src/fplh/features/information_set.py:76`) is the **only** door to data for every model and feature builder:

```python
visible = df[df["observed_at"] <= self._deadline].copy()  # line 127
```

Three design choices make it hard to leak by accident:

1. **Builders receive an `InformationSet` and nothing else.** There is no database connection, no path and no raw frame.
2. **Dimension tables expose only knowable columns.** `dim_fixture` serves the fixture id, teams, kickoff and round, but **not** the final score (`DIMENSION_COLUMNS`, line 38). Tables with no observation time raise `LeakageError` (line 124).
3. **It records the latest `observed_at` it served** (`max_observed_at`), so every run can *prove* max o_f ≤ D. The walk-forward runner checks this after every deadline (`evaluate/walk_forward.py:82`).

`restrict(D_g)` (line 90) returns the information set at an *earlier* deadline. Models use it to build training rows "as they would have looked" at each past deadline. Asking for a *later* deadline raises.

## 5. Proving it: the future-shuffle test

Filtering is necessary but not sufficient, because code can always find a back door. `src/fplh/features/leakage.py` adds a *behavioural* test:

1. Compute all features at deadline D.
2. **Scramble every row observed after D**: shuffle numbers, add 1, flip booleans (`perturb_future`, line 19).
3. Compute the features again.
4. They must be **identical** (`check_leakage`, line 60).

If any feature changed, it depended on the future. The test `test_a_leaky_builder_is_caught` (`tests/leakage/test_leakage.py`) writes a deliberately leaky builder, one that reaches into `info._store`, and checks that the shuffle test catches it. That is a **test of the test**. These tests block CI.

## 6. The spine and as-of joins

Every forecast row is keyed by **(player_uid, fixture_uid, deadline_at, horizon)**, the *spine* (`features/spine.py:13`). Feature families attach to it by an **as-of join**: for each spine row, take the latest fact with `observed_at ≤ deadline_at` (`asof_join`, `spine.py:84`, which uses DuckDB's `ASOF LEFT JOIN`).

```text
spine row: player 7, deadline 2024-10-05 11:00
snapshots for player 7:  10-03 09:00 price 7.5 | 10-05 08:00 price 7.6 | 10-05 14:00 price 7.7
as-of result → price 7.6   (the 14:00 snapshot is after the deadline)
```

The deadline itself is computed honestly: the first kickoff of the round minus 90 minutes (`historical_deadlines`, `spine.py:16`).

## 7. Entity resolution: one player, many names

The same human appears as "Martin Ødegaard" (FPL), "Martin Odegaard" (Understat) and maybe "M. Ødegaard" elsewhere. Every source must map to **one `player_uid`** (`src/fplh/entities/players.py`):

1. **Candidates:** only players at the same team in the same season.
2. **Name score:** the best rapidfuzz `token_set_ratio` over name variants, after `normalise_name` (strip accents, punctuation, case; transliterate Ø, æ, ł explicitly, because Unicode decomposition leaves them intact).
3. **Appearance overlap:** the Jaccard index of the fixtures each source says the player played, over fixtures *both* cover. This replaced a summed-minutes check that wrongly vetoed correct links, because the sources count substitute minutes differently.
4. **One-to-one greedy assignment,** strongest evidence first.
5. **Decision:** auto-link when the score is ≥ 92 with overlap ≥ 0.8, or the score is 80–92 with overlap ≥ 0.95 over ≥ 3 shared fixtures. Other candidates scoring ≥ 80 go to a review queue.
6. **Overrides** in `configs/entities/player_overrides.yaml` win, and each one records its evidence (Hegazi/Hegazy, Jonny/Jonathan Castro Otto, …).

The **coverage gate** demands that ≥ 99.5 % of FPL minutes belong to linked players. The result was **100 %** in every season from 2016/17 to 2026/27.

## 8. Data contracts and quality gates

Before any silver table is written, `src/fplh/lake/quality.py` runs checks. **Blocking** failures stop the build, so downstream code never sees bad data:

| Gate | Idea | Real result |
|---|---|---|
| Schema contracts (pandera) | Types, ranges, non-null keys | Pass |
| Goal conservation | Team goals = own players' goals + opponents' own goals | 7,620 / 7,620 team-fixtures |
| Cross-source scores | FPL, football-data and Understat agree on the score | 0 disagreements in 4,610 fixtures |
| Shot conservation | Understat shots add up per side | 1 upstream defect, excused *with evidence* in `configs/quality/exceptions.yaml` |
| Coverage | ≥ 99.5 % of minutes linked | 100 % |

Note the exceptions file. Real data have real defects, so an exception must carry a written `reason`, or loading it raises (`load_exceptions`, `quality.py:44`). Every exception is explained, never silently swallowed.

Each gate has a test that **corrupts one fixture** and checks the gate catches it.

## 9. Lessons from real data (from the plan)

- vaastav 2025/26 had 10 exact duplicate rows. They are dropped and counted; *conflicting* duplicates raise.
- Before 2020/21, vaastav rows have no team, and `players_raw.team` is the *end-of-season* club. Using it would leak transfers, so teams come from the fixture.
- Understat timestamps differ from FPL's after reschedules (1–4 h in 18 % of matches), so linked facts take FPL's kickoff.
- football-data files before 2019/20 have no kickoff time. Imputing 15:00 made midweek prices look post-kickoff, so real kickoffs come from FPL or Understat.

Each of these is a silent bug in a less careful pipeline.

## Check yourself

1. Understat publishes about 24 h after a match. A match kicked off Saturday 15:00 UTC, and the next deadline is Sunday 11:00 UTC. Is the match's xG in 𝓘(D)?
   <details><summary>Answer</summary>No. o_f = Saturday 15:00 + 24 h = Sunday 15:00, which is after the 11:00 deadline. The model must not use it, even though the match itself happened before the deadline.</details>
2. Why does `dim_fixture` hide `home_goals` even though it is a dimension table?
   <details><summary>Answer</summary>Dimension tables have no observation time, so a final score stored there would be visible at every deadline, including those before the match. Outcomes must come from time-indexed fact tables.</details>
3. What does the future-shuffle test catch that filtering alone does not?
   <details><summary>Answer</summary>Back doors. Code that bypasses the filter (for example by reaching into the store directly) or reads leaky columns still produces features that change when future rows are scrambled. The behavioural test catches any dependence on the future, however it arises.</details>

## Practical

```bash
uv run python docs/learn/exercises/ex05_data.py
```

You will compute observation times from lags, write a leak-free feature builder that passes the repo's `check_leakage` on synthetic silver (and watch a leaky one fail), do an as-of join by hand, and implement the entity-link decision rule.

Quiz: `uv run python docs/learn/quiz.py take 05`
