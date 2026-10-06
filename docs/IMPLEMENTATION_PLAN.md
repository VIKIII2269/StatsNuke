# Implementation plan

Companion to [`ARCHITECTURE.md`](../ARCHITECTURE.md), the design spec. This document covers:

- what is built;
- the gaps in the spec found while planning, and how each will be resolved;
- the phased ticket list, with acceptance criteria for every phase.

Phases are gated by their **exit criteria, not by the calendar**. The spec's §13 timeline (8 weeks) is optimistic for one developer. The order is fixed, the dates are not.

| Phase | Theme | Status |
|---|---|---|
| 0 | Collect and score | **Built** (collector goes live once merged to `main`; see §0.4) |
| 1 | Lake, entities, walk-forward harness | **Done**: every gate passes on the full real data (§2.2) |
| 2 | Team level (M1–M3, G0–G3) | **Done**: exit gate met as non-inferiority, fused ties the market (§3.2) |
| 3 | Match and player level (G4–G6, M4–M10, simulator) | **Done**: the simulator beats the OpenFPL re-implementation and both floors walk-forward (§4.6) |
| 4 | Decisions (MILP, season replay, paper ledger) | Planned |
| 5 | Extensions, each gated by an ablation | Planned |
| 6 | Operations (VPS, Dagster, Telegram, monitoring) | Planned |

---

## 0. Phase 0: Collect and score (built)

Why first: FPL snapshots (prices, status, `chance_of_playing_*`, news, ownership, `ep_next`) exist only as current state (§5.3.2). The 2026/27 season is already well under way, and every week without the collector permanently loses training features and the `ep_next` benchmark.

### 0.1 What exists

| Area | Files | Notes |
|---|---|---|
| Packaging and tooling | `pyproject.toml`, `uv.lock`, `.pre-commit-config.yaml` | Python 3.12, uv, ruff, mypy (strict), pytest, hypothesis, respx |
| Settings | `src/fplh/settings.py`, `configs/sources.yaml`, `src/fplh/sources.py` | Env prefix `FPLH_`; `FPLH_LAKE_URI` defaults to `./lake`; publication lags ℓ_s per source |
| Clock | `src/fplh/clock.py` | `SystemClock` live, `FixedClock` for backtests (P5) |
| Storage | `src/fplh/lake/storage.py` | fsspec: local path now, `s3://` (R2/B2) later with `pip install .[s3]`; create-only writes, atomic on local disk |
| Bronze | `src/fplh/lake/bronze.py` | Spec §6.1 path layout; gzip payload (byte-exact) + `.meta.json` sidecar with status, URL, params, `observed_at`, SHA-256; hash verified on read |
| HTTP | `src/fplh/collectors/http.py` | Identifying UA, shared rate limiter, retries on 408/425/429/5xx and transport errors with exponential backoff and jitter, honours `Retry-After`; 4xx returned, not retried |
| Collectors | `src/fplh/collectors/{base,fpl}.py` | `Collector` protocol (adapter seam, §5.3.6); FPL snapshot, fixtures, and post-GW (`event/{gw}/live` + every `element-summary`, bounded concurrency; partial failures persist what succeeded and exit non-zero) |
| Contracts | `src/fplh/collectors/fpl_schema.py`, `tests/contracts/` | Required fields typed; extra fields tolerated |
| Rules | `configs/rules/fpl_2025_26.yaml`, `fpl_2026_27.yaml`, `src/fplh/rules/{config,engine,bonus}.py` | Strict schema that rejects unknown keys and any `VERIFY` placeholder; vectorised engine returns a per-component breakdown; official bonus tie rule |
| Golden | `src/fplh/rules/golden.py`, `tests/golden/`, `scripts/make_golden_sample.py` | vaastav `merged_gw.csv` and our own `element-summary` pulls normalise to one frame |
| CLI | `src/fplh/cli.py` | `fplh collect {fpl-snapshot,fpl-fixtures,fpl-post-gw}`, `fplh rules score`, `fplh golden {fetch,check}` |
| Scheduling | `.github/workflows/collect.yml`, `scripts/{persist_to_branch,restore_state_from_branch}.sh` | Stopgap cron; see §0.4 |
| CI | `.github/workflows/ci.yml` | ruff, mypy, full test suite including full-season golden |

### 0.2 Golden results (the Phase 0 exit gate)

| Season | Rows | `total_points` | Clean-sheet flag | Defensive count | Bonus from BPS |
|---|---|---|---|---|---|
| 2025/26 (full season) | 29,747 | 0 mismatches | 0 | 0 | 0 |
| 2026/27 (GW1; vaastav only has GW1 so far) | 610 | 0 mismatches | 0 | 0 | 0 |

Findings:
- **GK goal points are not identified by the data.** No goalkeeper scored in 2025/26 or 2026/27 GW1. The configs use 10, from the official rules page. The golden gate will catch it if that is wrong the first time a GK scores.
- **vaastav 2025/26 contains 10 exact duplicate element × fixture rows** (Ben Gannon-Doak, Junior Kroupi). They don't affect points, but a naive bonus re-ranking over them gives 2 wrong rows. The loader drops *exact* duplicates, counts them, and raises on *conflicting* duplicates. Phase 1 silver normalisers need the same rule.
- **Our own clean-sheet and defensive-count derivations agree with FPL's official fields on every row.** Clean sheets are computed as ≥ 60 minutes and 0 conceded on the pitch. Defensive contributions are CBI + tackles for DEF, plus recoveries for MID/FWD. So the simulator can derive both from events, and neither needs to be read from FPL.

### 0.3 Known limitations

- **The build sandbox could not reach the FPL API** (the proxy returns 403). The collectors are verified with mocked HTTP (respx), not against live FPL. The first `workflow_dispatch` run is the live check.
- **`bootstrap-static.json`, `fixtures.json` and `event-live.json` contract samples are hand-built** from the documented shape. `element-summary.json` uses real values. Replace the hand-built ones with trimmed real payloads after the first live run (see `tests/contracts/samples/README.md`).

### 0.4 Going live (actions for the repo owner)

1. **Merge to `main`.** GitHub runs scheduled workflows only from the default branch.
2. Run **Actions → Collect → Run workflow** once with `snapshot`, and confirm a green run and a new commit on the `data-bronze` branch. If FPL returns 403 to GitHub-hosted runners, the collector must move to the VPS right away (Phase 6 task, pulled forward).
3. **Strongly recommended:** create an R2 (or B2) bucket and add the repository secrets `FPLH_LAKE_URI` (e.g. `s3://statsnuke-lake`), `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY` and `AWS_ENDPOINT_URL`. The workflow then writes to the bucket and stops using the data branch. Revised estimate for the branch fallback: one gzipped bootstrap every 3 h, plus hourly snapshots on deadline days, plus about 800 element summaries per gameweek comes to **≈ 0.7–1 GB per season**. That bloats every full clone of the repo, which is acceptable for weeks, not seasons.
4. **Actions minutes:** about 24 short hourly runs, 8 snapshot runs and 1 post-GW run per day. That's free on a public repo, and roughly 1,000 of the 2,000 free minutes per month on a private one.

Cadence implemented (§12.4):

| Job | Cron (UTC) | Behaviour |
|---|---|---|
| Snapshot | `17 */3 * * *` | `bootstrap-static` + `fixtures` |
| Deadline hourly | `47 * * * *` | Fetches `bootstrap-static`; stores it only if a deadline is ≤ 24 h away |
| Post-GW | `43 9 * * *` | Latest `data_checked` gameweek, once (tracked in `state/fpl_post_gw.json`) |

---

## 1. Spec gaps and their resolutions

These were found while planning. Each one changes how a later phase is built.

1. **Train/serve feature skew in M4 (minutes), §7.6.** `status`, `chance_of_playing`, `news` and ownership history exist only from our own captures (2026/27 onward), yet §11.1 trains on 2014/15–2024/25.
   - *Resolution:* M4 = **base model** trained on history *without* snapshot features, plus a **news overlay**: a logit offset trained only on captured seasons, shrunk to 0 while data is thin.
   - The walk-forward harness marks snapshot features *missing* before capture began and never imputes them.
   - Ablation A4 compares base-only against base plus overlay on live 2026/27.
   - *Phase 3:* the base model is built (§4.4). `chance_of_playing` is a feature that stays missing before captures, and the captures run from 30 September 2026 (§4.6). The overlay is fitted once a captured season accrues.
2. **The `ep_next` benchmark (G2) can be measured only live**, from our first captured deadline. The 2022/23–2025/26 tuning and holdout periods benchmark against OpenFPL and the naive floors instead. This changes the §13 Phase 3 exit criterion to "beats OpenFPL and naive floors walk-forward; beats `ep_next` on live gameweeks as they accrue". *Phase 3:* the first part is met (§4.6); `fplh evaluate ep-next` runs the second as captured gameweeks get results.
3. **Pinnacle closing odds after the July 2025 API closure.** First Phase 1 task for football-data: check whether `PSCH/PSCD/PSCA` are still populated for 2025/26 and 2026/27. If they aren't, the match benchmark becomes de-vigged **market-average closing** (`AvgCH/AvgCD/AvgCA`), and G1 is restated against it.
4. **The shot-rate ↔ goal-rate link is under-specified (§7.3 vs §7.5).** The emulator takes fused **goal** rates as its inputs and maps them to base shot rates, `λ̄^S_k = λ̄_k / (team mean xG per shot × league finishing)`. The inversion and the intensity model then share one parameterisation. Frailty variance and state effects are fixed at their fitted values inside the emulator (open question 6). *Phase 3:* the emulator inverts fused mean goals to G6 nominal rates by 2-D Newton (§4.3), so inversion and simulation share one parameterisation.
5. **Emulator cost.** A grid of 60² points × 10⁵ sims × ~96 steps needs a **team-only simulator path** with no player allocation, lineups or bookings. Scoreline markets depend only on team intensities, red cards and frailty. Budget: 1–2 h on Colab CPU, cached by `goal_process` model version. *Phase 3:* a 20² grid × 10⁵ team-only simulations (§4.3) takes minutes and is cached in gold by parameter hash.
6. **Publication lags for backfilled history** are now configured in `configs/sources.yaml`:
   - FPL and vaastav history: 33 h (lockdown);
   - Understat: 24 h;
   - football-data results: 48 h.

   Closing odds are known only at kickoff and are never a pre-deadline feature.
7. **The Odds API budget can't cover the §12.4 cadence.** 500 credits/month doesn't allow "every 30 min in the final 6 h". The budget planner's priority order is:
   - (1) one closing snapshot per kickoff slot, about 24 per month × 2 credits;
   - (2) one snapshot at deadline − 2 h per gameweek;
   - (3) opportunistic snapshots from what is left. That totals ≈ 150–250 credits per month.
8. **Free-transfer behaviour in chip weeks** belongs in the rules config (`game.free_transfers_preserved_on`, already added), not hard-coded in the MILP.
9. **Walk-forward compute.** A full NUTS refit every 4 gameweeks × 38 × 3 tuning seasons is about 30 refits per configuration. Run the ablation ladders with SVI/Laplace, and run NUTS only on finalists and for the holdout. *Phase 3:* no player-level model needs NUTS. Walk-forward runs are cached by predictor version, data and configuration, so the full exit gate with two ablations runs once in about 80 minutes on 4 cores, and reruns are free.
10. **vaastav is now three updates per season.** 2026/27 golden and training data must come from our own post-GW pulls. That makes the Phase 0 collector a hard dependency for Phase 1 as well.

---

## 2. Phase 1: Lake, entities, walk-forward harness

Goal: every historical fact in silver with correct `o_f`, one identity per player, and a harness that can replay any deadline.

### 2.1 Tickets

| # | Ticket | Status | Where |
|---|---|---|---|
| 1.1 | football-data E0/E1 backfill + odds-column report | **Built and run** (68 files, 1993/94–2026/27) | `collectors/football_data.py`, `lake/silver/football_data.py`, `fplh backfill football-data`, `fplh report odds-columns` |
| 1.2 | vaastav backfill | **Built and run** (2016/17–2026/27) | `collectors/vaastav.py`, `lake/silver/vaastav.py`, `fplh backfill vaastav` |
| 1.3 | Understat league + match backfill | **Built and run** (13 seasons, 4,610 matches) | `collectors/understat.py`, `lake/silver/understat.py`, `fplh backfill understat` |
| 1.4 | The Odds API + credit budget planner | Built, mock-tested; live check needs `FPLH_ODDS_API_KEY` | `collectors/odds.py`, `collectors/odds_budget.py`, `fplh collect odds --due` |
| 1.5 | FBref events | **Deferred to Phase 3** (user decision): lineups and minutes come from FPL, per-player subs and cards from Understat rosters | — |
| 1.6 | Silver normalisers + pandera contracts | **Built**; rebuilds are byte-identical | `lake/silver/`, `fplh silver build` |
| 1.7 | Entities (teams, fixtures, players) | **Built**; player coverage 100 % in every season | `entities/`, `configs/entities/` |
| 1.8 | Quality gates | **Built**; each gate has a test that corrupts one fixture | `lake/quality.py` |
| 1.9 | Information set + spine | **Built** | `features/information_set.py`, `features/spine.py`, `features/builders.py` |
| 1.10 | Leakage tests | **Built**, CI-blocking; clean on real 2024/25 silver | `tests/leakage/`, `fplh evaluate leakage` |
| 1.11 | Metrics | **Built** | `evaluate/metrics.py` |
| 1.12 | Block bootstrap + Diebold–Mariano | **Built**; size ≈ 5 % | `evaluate/bootstrap.py` |
| 1.13 | Walk-forward + manifests + tracking | **Built**; replays give the same run id and identical bytes | `evaluate/walk_forward.py`, `evaluate/manifest.py`, `evaluate/tracking.py` |
| 1.14 | A0 leaderboard | **Logged** for 2022/23–2024/25: player and match levels, with market prices | `models/baselines.py`, `models/market.py`, `evaluate/a0.py`, `fplh evaluate a0` |

### 2.2 Results on real data

Full backfills: vaastav 2016/17–2026/27, football-data E0 + E1 1993/94–2026/27 (68 files) and Understat 2014/15–2026/27 (4,610 matches). `fplh silver build` passes **every blocking gate**:

| Gate | Result |
|---|---|
| Goal conservation | 7,620 / 7,620 FPL team-fixtures |
| Cross-source scores (FPL, football-data, Understat) | 4,610 multi-source fixtures, 0 disagreements |
| Understat shot conservation | 9,220 match sides; 1 upstream defect excused with its evidence in `configs/quality/exceptions.yaml` |
| Kickoff agreement | 0 fixtures more than 36 h apart |
| Player coverage (gate ≥ 99.5 %) | **100 % of FPL minutes in every season 2016/17–2026/27**: 5,764 team-season links (13 from 5 hand-checked overrides) |
| Pre-match odds coverage (2014/15+) | 2,710 / 2,710 fixtures |

Also:
- golden `total_points` reproduced on all 254,119 player-fixtures, 0 mismatches;
- leakage checks clean on real 2024/25;
- rebuilds byte-identical across 163 silver parts.

**Odds availability (spec gap 3), `fplh report odds-columns`:**
- Pinnacle pre-match and closing 1X2 are complete for 2012/13–2024/25, cover 55 % of 2025/26 and are absent in 2026/27, confirming the July 2025 API closure in the data.
- Market-average closing (1X2 and O/U 2.5) is complete from 2019/20.
- **Benchmark decision:** de-vigged Pinnacle closing through 2024/25, market-average closing afterwards. The tuning seasons have both.

**A0 baseline** (walk-forward, horizon 1, logged to the leaderboard):

| Season | Player MAE (played) | Player ρ within position (played) | Match RPS, A0 pre-match prices | Match RPS, de-vigged closing |
|---|---|---|---|---|
| 2022/23 | 2.044 | 0.326 | 0.2043 | 0.2028 |
| 2023/24 | 2.076 | 0.332 | 0.1790 | 0.1757 |
| 2024/25 | 1.987 | 0.328 | 0.1983 | 0.1971 |

- Market clean-sheet rates raise player ρ from about 0.30 (league-average rates) to about 0.33.
- The match rows cover fixtures whose pre-match price was observable by the deadline (342–369 per season); prices collected after a round's deadline are correctly unusable.

### 2.4 Findings from real data

- **Open question 1 (historical defensive actions):** vaastav `merged_gw.csv` carries FPL's own `clearances_blocks_interceptions`, `recoveries` and `tackles` for **2016/17–2018/19** as well as 2025/26+. M7 therefore has four native seasons, not one, and 2025/26 can be a holdout for M7 after all. The only gap is 2019/20–2024/25. Before use, check the older counts against 2025/26 per-position distributions for definition drift.
- **vaastav quirks**, each handled and counted in the build notes:
  - before 2020/21 there is no `team`/`position` per row. Teams come from the fixture, never from `players_raw.team`, which is the end-of-season club;
  - 2019/20 has 59 unplayed placeholder rows for postponed fixture 275 (GW29 → GW39);
  - 2024/25 has 322 assistant-manager rows (position `AM`), which are dropped;
  - 2021/22 labels 101 rows `GKP`;
  - 2025/26 team IDs are not alphabetical, and 2026/27 `teams.csv` uses full club names (hence aliases).
- **GK goal points:** 6 up to 2024/25, identified by data (Alisson, 2020/21). 10 from 2025/26, per the official rules page, still unidentified by data.
- **Diebold–Mariano at season scale:** the textbook normal DM rejected 8.6 % of A-vs-A comparisons at 38 gameweeks. The implemented version (HLN correction, t critical values, NW truncation at h − 1) rejects ≈ 5 %.
- **Name matching:** Unicode decomposition leaves `Ø`, `æ`, `ł` and similar letters intact, so they are transliterated explicitly (Ødegaard, Højbjerg, Fabiański).

- **Player linking needed appearance overlap, not minutes:**
  - FPL and Understat count substitute minutes differently (about 1 min per appearance), and summed-minutes checks vetoed correct links (2016/17 coverage 98.56 %).
  - Links now require overlap in the fixtures each source says the player played, counted only over fixtures *both* sources cover, because during a live season their windows differ (FPL history to GW1 vs Understat to GW5 in 2026/27).
  - Understat names are HTML-unescaped (`N&#039;Diaye`).
  - The 5 remaining spelling and transliteration cases (Hegazi/Hegazy, Zambo Anguissa/Franck Zambo, Yéremy/Yeremi Pino, Fer/Fernando López, Jonathan Castro Otto/Jonny) are overrides with their evidence.
- **Understat's live JSON keys are `dates`/`teams`/`players` and `rosters`/`shots`**, not the `…Data` names a third-party scraper used. Contract tests now use trimmed real payloads.
- **Kickoff times:**
  - Understat timestamps are UTC but differ from FPL's after reschedules (1–4 h in 18 % of 2016/17+ matches), so linked facts take FPL's kickoff.
  - football-data files before 2019/20 have no kickoff time. The 15:00 imputation made midweek and holiday prices look post-kickoff, so the real kickoff now comes from FPL or Understat, and the pre-match observation rule is capped at kickoff − 1 h, which is always after the round's deadline.

Exit: coverage gate passes; leakage tests green; A0 logged. **Met.**

## 3. Phase 2: Team level

Goal: team scoring rates from a dynamic model (M1), the market (M2) and their fusion (M3), with the scoreline-model ladder G0–G3, evaluated walk-forward on the tuning seasons 2022/23–2024/25.

### 3.1 Tickets

| # | Ticket | Status | Where |
|---|---|---|---|
| 2.1 | De-vig: multiplicative, power, Shin; default by closing-price calibration | **Built and fitted**: power (see §3.2) | `models/market.py`, `fplh models fit-devig`, `configs/models/market.yaml` |
| 2.2 | G0 Poisson, G1 Dixon–Coles, G2 diagonal-inflated bivariate Poisson, G3 COM-Poisson + market inversion | **Built**; grids sum to 1, inversion reproduces synthetic prices to 1e-6 | `models/goal_benchmarks.py` |
| 2.3 | M1: mean-reverting attack/defence ratings, Poisson goals + ω-weighted Gamma xG, summer regression, E1 promoted-team prior, horizon variance | **Built and fitted** on 2014/15–2021/22. The squad market-value shift is **deferred**: no squad-value source yet | `models/team_strength.py`, `fplh models fit-m1`, `configs/models/team_strength.yaml` |
| 2.3 | NumPyro NUTS reference model | **Built** (optional `[bayes]` extra); recovers simulated ratings | `models/team_strength_nuts.py` |
| 2.4 | Laplace/EKF filter for every deadline | **Built**; filtered ratings agree with the NUTS refit on the same data (test) | `models/team_strength.py` |
| 2.5 | M3 fusion: logistic time-to-kickoff and liquidity weight, ridge, scoreline NLL | **Built**; trained on deadline-time snapshots of the 5 seasons before evaluation, ridge by leave-one-season-out CV | `models/fusion.py` |
| 2.6 | Ablations A1–A3 and the G0–G3 ladder | **Logged** with gameweek-block bootstrap CIs and DM tests | `evaluate/team_level.py`, `evaluate/phase2.py`, `fplh evaluate phase2` |

### 3.2 Results on real data

**De-vig (`fplh models fit-devig --before 2022-07-01`).** Mean 1X2 log loss of closing prices, EPL matches before the tuning seasons:

| Book | Matches | Multiplicative | Power | Shin |
|---|---|---|---|---|
| Pinnacle | 3,800 | 0.953245 | **0.953187** | 0.953239 |
| Market average | 1,140 | **0.968636** | 0.969353 | 0.969162 |

Power wins on Pinnacle, the sharper book, and is the default. The methods differ by < 1e-4 nats on Pinnacle's thin margins, so the choice barely matters for the forecasts.

**M1 fit (`fplh models fit-m1 --before 2022-07-01 --restarts 2`).** One-step-ahead predictive log-likelihood of goals per match, 2014/15–2021/22:

| Parameters | Log-likelihood / match |
|---|---|
| Defaults | −2.90762 |
| Fitted, goals only (ω = 0) | −2.90591 |
| Fitted, without the E1 promoted-team prior | −2.90289 |
| **Fitted** | **−2.89881** |

**Walk-forward evaluation (`fplh evaluate phase2 --season 2022-23 --season 2023-24 --season 2024-25`).** Horizon 1 (the next gameweek), forecasts made at each FPL deadline from only what was observable then. 1,057 fixtures had a pre-match price observable at the deadline; every model is scored on the same fixtures.

| Model | RPS | 1X2 log loss | Scoreline log loss |
|---|---|---|---|
| Market only (deadline prices, power de-vig, G1 inversion) | **0.19399** | **0.95200** | 2.98453 |
| M3 fused (market + M1) | 0.19419 | 0.95251 | **2.98280** |
| M1, default hyper-parameters (A3) | 0.19685 | 0.96302 | 2.99026 |
| M1 + G1 | 0.19711 | 0.96161 | 2.99407 |
| M1 + G3 | 0.19715 | 0.96224 | 2.99284 |
| M1 + G2 | 0.19716 | 0.96222 | 2.99359 |
| M1 + G0 | 0.19717 | 0.96239 | 2.99325 |
| M1, goals only (A2) | 0.20038 | 0.97144 | 3.01006 |

For reference, the A0 de-vigged *closing* prices score about 0.192 RPS on these seasons (§2.2); they include team news and late money that no deadline-time forecast can see.

Comparisons (a − b; negative favours a; 95 % gameweek-block bootstrap CI; DM p-value):

| a | b | RPS | Scoreline log loss | 1X2 log loss |
|---|---|---|---|---|
| M1 + G1 | M1 + G0 | −0.00009 [−0.00023, 0.00004], p 0.23 | +0.0003 [−0.0018, 0.0024], p 0.82 | −0.0011 [−0.0027, 0.0005], p 0.20 |
| M1 + G2 | M1 + G0 | −0.00002 [−0.00005, 0.00002], p 0.40 | +0.0004 [−0.0003, 0.0010], p 0.29 | −0.0002 [−0.0006, 0.0001], p 0.19 |
| M1 + G3 | M1 + G0 | −0.00002 [−0.00006, 0.00003], p 0.47 | −0.0005 [−0.0012, 0.0002], p 0.15 | −0.0002 [−0.0004, 0.0001], p 0.17 |
| **M1 (goals + xG)** | **M1 goals only (A2)** | **−0.0041 [−0.0071, −0.0018], p 0.003** | **−0.0208 [−0.0360, −0.0095], p 0.003** | **−0.0117 [−0.0215, −0.0047], p 0.008** |
| M1 tuned | M1 default (A3) | +0.0013 [−0.0014, 0.0048], p 0.43 | +0.0061 [−0.0051, 0.0188], p 0.33 | +0.0019 [−0.0060, 0.0122], p 0.68 |
| M3 fused | market only (A1, exit gate) | +0.00012 [−0.00046, 0.00071], p 0.70 | −0.0022 [−0.0049, 0.0004], p 0.10 | +0.0002 [−0.0018, 0.0021], p 0.83 |
| M3 fused | M1 + G1 | −0.0020 [−0.0046, 0.0008], p 0.14 | −0.0072 [−0.0202, 0.0077], p 0.29 | −0.0065 [−0.0150, 0.0028], p 0.15 |

Fusion as fitted on 2017/18–2021/22 (ridge 1.0 chosen by leave-one-season-out NLL: 2.87886, 2.87872, **2.87846**, 2.87865 for ridge 0.01, 0.1, 1, 10): market weight w = σ(1.58 − 0.0002 log(1 + τ) − 0.00005 liq) ≈ 0.83 at every deadline, home bias −0.016, away bias +0.001.

### 3.3 Findings from real data

- **xG is worth having (A2).** Goals + xG beats goals only on every metric, with the CIs excluding zero (RPS −0.0041, DM p = 0.003). At the optimum the Gamma xG likelihood enters with weight ω = 0.12 (Gamma shape κ = 16), next to the full-weight Poisson goals likelihood.
- **Tuning barely matters at deadlines (A3).** The fitted hyper-parameters improve the one-step-ahead likelihood (−2.8988 vs −2.9076), but walk-forward at deadlines they are indistinguishable from the defaults (RPS +0.0013, CI [−0.0014, 0.0048]).
- **G1–G3 add nothing over G0.** Dixon–Coles ρ = −0.045, bivariate-Poisson λ₃ → 0 and COM-Poisson ν = 1.018 are all close to the Poisson case; every CI includes zero. G0 stays the default; G1 is used for market inversion and fusion because its ρ is the only non-trivial dependence parameter.
- **Fitting M1 needs restarts.** With the two E1 slopes, M1 tunes 14 hyper-parameters. One 400-iteration Nelder–Mead run stopped at −2.90002, worse than a point it had not found (−2.89975). Two warm-started chains with restarts agree to 1e-4. At the optimum `sigma_a` sits at its lower bound (attack ratings barely drift within a season) and `season_regress` at 1 (no summer regression toward the mean).
- **The filter must run single-threaded.** It makes thousands of tiny linear-algebra calls, and BLAS threads only contend: one pass over 3,040 matches took 76 s with 4 threads and 2.4 s with one.
- **The market's goal bias is season noise.** The home-goal log bias of de-vigged pre-match prices ranges from −0.13 (2024/25) to +0.04 (2022/23, 2023/24), with −0.10 in the behind-closed-doors season 2020/21. The first fusion fit trained on three seasons with a fixed ridge of 0.01: it put all its weight on the market (α₀ = 11.6) and learned a home bias of −0.06, making it slightly worse than market-only (RPS +0.0008, CI [−0.0003, 0.0020]). Training on five seasons with the ridge chosen by leave-one-season-out CV within them (no evaluated season involved) shrinks the bias to −0.016 and gives M1 about 17 % weight. This change was made after seeing the first result; both runs are reported here.
- **M1 adds little to deadline prices (A1).** Fused vs market-only is a tie on RPS and 1X2 log loss. Fused is better on the scoreline grid (−0.0022), but its CI still includes zero (p = 0.10). The market already prices most of what goals and xG know; Phase 3 adds what it may not (minutes, line-ups, player-level xG).
- **Market prices are inverted under the grid's own goal model**, so a market-only forecast reproduces the market's 1X2 exactly, and the best-price composite (`market_max`) is never treated as a book.

### 3.4 Open items

- Squad market-value shift for the summer prior (needs a squad-value source).
- Re-run `fplh models fit-m1` and `fplh evaluate phase2` whenever silver changes materially; both are reproducible from the CLI (about 70 min each on 4 cores).

Exit: fused ≥ market-only on RPS (paired bootstrap), and G0–G3 results logged. **Met as non-inferiority:** fused − market RPS is +0.00012 with 95 % CI [−0.00046, +0.00071] (DM p = 0.70), so fused is not worse, and it is the best model on scoreline log loss, but it is **not** significantly better than the market on any metric. The G0–G3 ladder and A1–A3 are logged to the leaderboard.

## 4. Phase 3: Match and player level

Delivered as six PRs, each merged when green:
1. event timeline and BPS rules;
2. benchmarks (an OpenFPL re-implementation and naive floors);
3. G4–G6 and the emulator;
4. M4–M6 and the simulator core;
5. M7–M10;
6. player walk-forward, attribution and the exit gate.

Decisions:
- OpenFPL is re-implemented: XGBoost on its rolling features, trained walk-forward on our data. Its pickles are not loaded.
- Event timing comes from Understat rosters, because FBref is blocked.
- The 2025/26 holdout stays untouched. The exception is M7, which uses within-2025/26 walk-forward (open question 1c).
- No scikit-learn and no LightGBM: xgboost and `scipy.optimize.isotonic_regression` cover M4.

| # | Ticket | Acceptance criteria |
|---|---|---|
| 3.0 | Event timeline (`lake/silver/timeline.py`, `fact_match_event`), lineup, goal-timeline and substitution gates, `InformationSet.restrict`, typed BPS tables, bonus eligibility mask | **Built** (§4.1) |
| 3.0b | Benchmarks: OpenFPL re-implementation (`features/openfpl.py`, `models/openfpl.py`), last-5 floor, player scoring (`evaluate/player_level.py`), `fplh evaluate phase3-benchmarks` | **Built** (§4.2) |
| 3.1 | `models/goal_process.py` (G4–G6): piecewise-exponential Poisson GLM, hierarchical state effects, red-card hazard, frailty, stoppage-time model | **Built** (§4.3): dispersion reported; G6 drives the simulator, G0 stays the pre-match default |
| 3.2 | `sim/emulator.py`: team-only grid, bicubic spline per market, 2-D Newton inversion | **Built** (§4.3): max error 0.005; cached by parameter hash |
| 3.3 | `models/minutes.py` (M4): staged XGBoost with monotone constraints, isotonic calibration, substitution-era shift, news overlay (gap 1) | **Built** (§4.4): ECE 0.002 / 0.004; A4 beats naive |
| 3.4 | `models/attack.py` (M5/M6): conjugate Gamma-Poisson with decay, shrunk shot quality and finishing, penalties and own goals | **Built** (§4.4): A5 logged |
| 3.5 | `models/defence.py` (M7): NegBin with definition-drift rescaling and an opponent factor | **Built** (§4.5): threshold Brier beats the position mean within 2018/19 and 2025/26 |
| 3.6 | `models/gk.py`, `models/cards.py`, `models/bonus.py` (M8–M10; effective BPS weights, official table as reference) | **Built** (§4.5): each beats its baseline |
| 3.7 | `sim/simulator.py`: vectorised (sims × players) minute stepper, common random numbers, epistemic batches, DGW summation; calls `rules.engine.score_arrays` + `rules.bonus.assign_bonus_array` | **Built** (§4.4–4.6): property tests pass; walk-forward predictor `models/player_sim.py` |
| 3.8 | `evaluate/attribution.py` (§11.5) | **Built** (§4.6): parts sum to the error on every row (checked in code and tests) |

Exit: §8.5 validation passes; the simulator beats OpenFPL and the naive floors walk-forward; the `ep_next` comparison runs on live gameweeks (gap 2). **Met** (§4.6); `ep_next` runs as captures and results accrue.

### 4.1 Event timeline and BPS rules on real data

**Timeline.** `fact_match_event` holds 12,464 goals, 451 own goals, 29,694 substitutions, 580 red cards and 2 unreplaced exits for 4,610 matches (2014/15–2026/27).
- **Substitution minutes** come from Understat roster pairs. Of 25,787 pairs whose substitute plays to the end, 0 disagree by more than 1 minute with "90 − the substitute's minutes".
- **Red cards:** all 580 are placed, 117 of them at or after 90 minutes (`red_late`).
- **Goal minutes** come from `fact_shot`. An own-goal row carries the scorer's side; the goal counts for the opponent. With that convention, goal events reproduce every final score.
- **Assisters:** all 86,894 named assisters match a rostered player.
- **Gates:** 0 sides with more than 11 on the pitch; 0 with fewer than 11 starters.
- **Not identifiable:** yellow-card minutes and first-half stoppage time (its shots are recorded at minutes 45–50). Second-half stoppage is visible: shot minutes run to 105.

**Substitution limits** (`configs/rules/football.yaml`, verified by a warning gate with 0 violations):

| Period | Permanent substitutions |
|---|---|
| To June 2020 | 3 |
| Project Restart (June 2020, rest of 2019/20) | 5 |
| 2020/21–2021/22 | 3 |
| From 2022/23 | 5 |

Up to 2 concussion substitutes have been allowed since February 2021. They account for every side above its base limit.

**BPS.** Every season YAML now carries a typed base table for the events the simulator generates. `fplh rules check-bps` estimates effective weights by least squares (R² 0.83–0.86):
- The 60+ minutes award applies from exactly 60 minutes.
- The penalty-save weight fell from 15 (estimates 14.2–16.0 through 2023/24) to about 8 (8.9 in 2024/25, 7.0 in 2025/26), so 8 is recorded from 2024/25.
- Goals, clean sheets and penalty misses carry the BPS of correlated actions the table scores separately (shots on target, big chances). M10 therefore reconstructs BPS with the effective weights and keeps the rest in its residual.

### 4.2 The bars the simulator must clear

`fplh evaluate phase3-benchmarks --season 2022-23 --season 2023-24 --season 2024-25` runs every benchmark walk-forward at horizon 1. All models are scored on the same 80,973 player-fixtures, of which 33,208 had minutes > 0, over 110 gameweek blocks. Runtime is 10 minutes.

**OpenFPL replica.** OpenFPL's feature set rebuilt on silver: trailing 1/3/5/10/38-match means of FPL, Understat, team and opponent statistics, 211 features. It trains one XGBoost model per position, refit every 4 deadlines. Each training row is featurised at its own round deadline by `features/trailing.py`, which computes the as-of features exactly. A test checks that the batch features equal the per-deadline features. Tree counts per position (GK 135, DEF 166, MID 114, FWD 80) come from early stopping on 2021/22.

| Model | MSE | MAE | Spearman within position | Spearman, played | Top-10 precision |
|---|---|---|---|---|---|
| **OpenFPL replica** | **3.661** | **0.996** | 0.696 | **0.371** | **0.429** |
| A0 (season per-90 × naive minutes, market CS) | 4.015 | 1.001 | **0.714** | 0.331 | 0.410 |
| Naive last-5 average | 4.354 | 1.050 | 0.681 | 0.283 | 0.384 |

Paired differences in MSE (95 % gameweek-block CI, DM p):

| Comparison | Difference | 95 % CI | DM p |
|---|---|---|---|
| Replica − last 5 | −0.707 | [−0.770, −0.646] | < 1e-40 |
| Replica − A0 | −0.374 | [−0.455, −0.308] | < 1e-16 |
| Last 5 − A0 | +0.333 | [0.279, 0.385] | |

The replica clears both floors, so the simulator's exit bar is MSE 3.66 on these rows. A0 still ranks all rows best (Spearman 0.714): its minutes model separates benched from starting players. The replica ranks players who played better (0.371 vs 0.331). The simulator needs both.

### 4.3 In-match goal process (G4–G6) and the emulator

`fplh evaluate g-ladder --season 2022-23 --season 2023-24 --season 2024-25` fits G4–G6 on 2,660 matches from 2015/16 to 2021/22, builds their emulators and scores them at the tuning-season deadlines. It writes `configs/models/goal_process.yaml`. The first run takes about an hour; reruns take about 7 minutes because emulators and walk-forward runs are cached by their inputs.

**The model** (`models/goal_process.py`): a per-minute goal intensity built from
- each match's pre-match expected goals (M1 as the offset);
- an 18-bin time profile plus a stoppage bin;
- game-state effects by goal difference and time bucket;
- red-card effects;
- a Gamma match frailty, which integrates out exactly, so the likelihood and gradient are closed-form (the spec's log-normal has no closed form);
- a red-card hazard;
- a second-half stoppage survival curve.

G6 takes its game-state effects from non-penalty shots (about 10× the events) plus a shrunk per-state conversion offset. Fits take 0.4–2 s. A test recovers known state effects from 6,000 simulated matches.

Fitted values:
- **Red cards:** your own red multiplies your scoring rate by e^−0.54 ≈ 0.58; the opponent's red by e^0.60 ≈ 1.82. The red hazard rises through the match.
- **Frailty:** variance fits to ≈ 0. Goals are not overdispersed; Pearson φ̂ is 0.77–1.11 per season and below 1 in 5 of 7. So the frailty effectively drops out, as the spec allows.
- **Game state:** effects are small. A team trailing by one after minute 75 scores about 10 % more.
- **Time profile:** the rate rises through the match, with a bump at minutes 45–49 where first-half stoppage time is recorded.

**Ladder** (M1 deadline rates, 1,140 fixtures; differences a − b with 95 % gameweek-block CIs):

| Model | Scoreline log loss | 1X2 log loss | RPS |
|---|---|---|---|
| G0 Poisson | 3.0007 | 0.9600 | 0.19712 |
| G1 Dixon–Coles | 3.0014 | 0.9594 | 0.19708 |
| G4 | 3.0009 | 0.9598 | 0.19713 |
| G5 | 3.0017 | 0.9597 | 0.19714 |
| G6 | 3.0010 | 0.9596 | 0.19712 |

| Comparison | Scoreline log-loss difference (95 % CI) |
|---|---|
| G4 − G0 | +0.00002 [−0.0024, 0.0027] |
| G5 − G4 | +0.0007 [−0.0009, 0.0024] |
| G6 − G5 | −0.0007 [−0.0018, 0.0004] |

None of these is significant, as with G1–G3 in Phase 2: pre-match scorelines are Poisson to within what three seasons can resolve. **G0 stays the pre-match default.** The simulator needs in-match dynamics, so it uses the highest level not significantly worse than its predecessor: **G6**.

**§8.5 team checks:**

| Check | Result | Target |
|---|---|---|
| Emulator vs a 2·10⁵-draw direct simulation, 60 random mean-goal pairs × 4 probabilities | mean abs 0.0013, max 0.0050 (the reference's own SE is 0.0011); mean goals within 0.012 | max ≤ 0.005 |
| Market reproduction: two-rate inversion of de-vigged 1X2 + O/U 2.5 (1,057 fixtures) | mean abs 0.0048 (G1: 0.0029) | within 0.002 of G1 |
| Draw rate, observed − predicted | +0.010, CI [−0.013, 0.032] | CI covers 0 |
| Scoreline cells up to 4–4 inside their block CIs | 25 / 25 | ≥ 90 % |
| In-play next-goal ECE from the actual state at 15–75′ (5,700 forecasts): home / away / none | 0.015 / **0.024** / 0.014 | ≤ 0.02 |

- **The away miss is real, not noise:** a perfectly calibrated forecaster scores ≤ 0.018 at the 95th percentile on these forecasts.
- **Pattern:** at 15′ the split is exact. From 30′, "away scores next" is under-predicted by 2–4 points.
- **Not a missing model term:** the training-period residuals by side × time bucket and side × goal difference are all within |z| < 1.4.
- **So it is a shift in the tuning seasons:** away teams score relatively more as the match goes on. Refitting on recent seasons (or a side × time term once the data supports it) is left for the player-level PRs to revisit.

**Emulator.**
- A 20 × 20 log-spaced grid of nominal rates × 10⁵ simulations with common random numbers, plus a smoothing bicubic spline per output. That's about 1/20 of the spec's 60² × 10⁵ (gap 5).
- Mean goals are inverted to nominal rates by 2-D Newton. Wrapped as a `GoalModel`, it plugs into market inversion and fusion unchanged.
- Emulators are cached in gold by parameter hash.

### 4.4 Minutes (M4), attack (M5/M6) and the simulator core

`fplh evaluate components --season 2022-23 --season 2023-24 --season 2024-25` scores M4 walk-forward at every deadline (refit every 4 deadlines; about 10 minutes) and M5/M6 at every deadline.

**M4 minutes** (`features/minutes.py`, `models/minutes.py`). Three stages, each an XGBoost classifier (native API, monotone in recent starts and chance of playing), isotonically calibrated on the last 20 % of its training rows:
- π^S = P(start);
- π^60 = P(60+ | start);
- π^B = P(appearance | not started).

Features, each training row featurised at its own round deadline:
- trailing starts, appearances, 60+ and minutes over 1/3/5/10 matches, from Understat lineups;
- 60+ share when starting;
- depth at position among the side's registered players;
- rest days and days since the last appearance;
- price;
- chance of playing, which is missing until snapshot captures exist, so the news overlay of gap 1 is a no-op for now.

Findings:
- **Substitution eras.** The features carry no era, and trees cannot extrapolate the 2022/23 move to five substitutes. π^B therefore gets a logit shift per substitution limit, measured on that era's own training rows. Before 2022/23 those are only the 2019/20 restart. Measured on the era's own rows, the shift brought 2022/23's mean π^B from 0.191 to 0.169 (observed 0.163); the version calibrated against the latest window had given 0.191.
- **No per-team normalisation.** The deadline squad list also holds departed and long-absent players, so scaling π^S to sum to 11 over it biased every probability.
- **Bug found by the walk-forward.** A day count divided raw timestamps by ns per day, while fixture tables are in µs and the spine in ns. Walk-forward P(start) fell to 0.14 against 0.30 observed. A test now checks that spine and training features agree across units.
- Horizon drift and the small-sample prior blend are not needed at horizon 1: the trees see each player's match count.

| Stage | n | Brier | ECE | Mean predicted | Observed |
|---|---|---|---|---|---|
| Start | 80,973 | 0.0805 | **0.0015** | 0.301 | 0.301 |
| 60+ given start | 24,357 | 0.0615 | **0.0039** | 0.929 | 0.933 |
| Appearance given not started | 56,616 | 0.0866 | 0.0029 | 0.157 | 0.156 |
| 60+ | 80,973 | 0.0845 | 0.0036 | 0.280 | 0.282 |
| Appearance | 80,973 | 0.0924 | 0.0037 | 0.410 | 0.410 |

**A4** (Brier against the naive share of the last three matches, 95 % gameweek-block CI):

| Target | Model | Naive | Difference (95 % CI) |
|---|---|---|---|
| Start | 0.0805 | 0.1044 | −0.0241 [−0.0258, −0.0225] |
| 60+ | 0.0845 | 0.1041 | −0.0198 [−0.0213, −0.0183] |
| Appearance | 0.0924 | 0.1108 | −0.0185 [−0.0202, −0.0171] |

**M5/M6 attack** (`models/attack.py`): decayed (half-life 365 days) Understat player-matches and shots give each player:
- a Gamma–Poisson non-penalty shot rate, shrunk to the mean of the player's position × role group, with prior strength from the method of moments;
- shot quality (xG per shot) shrunk with 20 pseudo-shots;
- finishing (goals per xG) shrunk with 30 pseudo-xG towards 1;
- an assist weight (xA per 90, Gamma–Poisson);
- a penalty-taker weight (decayed attempts).

League constants: penalties are 7.4 % of goals, own goals 3.3 %; penalty conversion is 0.79; 0.021 missed penalties per side-match; 88 % of goals carry an FPL assist.

**A5** (non-penalty goals per appearance, Poisson log loss with minutes as exposure, 95 % gameweek-block CI):

| Players | n | Shrunk | Raw decayed per-90 | Difference (95 % CI) |
|---|---|---|---|---|
| All | 34,296 | 0.2692 | 0.3488 | −0.081 [−0.092, −0.070] |
| Under 10 decayed 90s of history | 8,551 | 0.2197 | 0.4494 | −0.231 [−0.270, −0.191] |

**Simulator core** (`sim/simulator.py`). The team process (G6) supplies goal and red-card minutes, so team totals are exactly those of the goal process.

How players are placed:
- **Starting XI:** 1 GK + 10 outfield, drawn by systematic sampling, so inclusion probabilities equal π^S exactly.
- **Exits:** at the empirical exit minutes from lineups, with π^60 deciding 60+.
- **Entrants:** ∝ π^B, within the era's substitution limit.
- **Reds:** to on-pitch players.

How goals are allocated:
- **Own goals** are credited to opponents.
- **Penalties** go to the taker on the pitch.
- **Open-play goals** go ∝ M5/M6 rates.
- **Assists** are given at the league share.
- **Missed penalties** are added as their own events.

Points come from `score_arrays`.

Property tests (Hypothesis) check:
- player goals plus opponents' own goals equal team goals;
- exactly 11 starters with one GK;
- substitutes within the limit;
- minutes in range;
- identical replays;
- exact inclusion probabilities;
- simulated starts and 60+ matching their targets;
- the team-goal distribution matching the team-only simulation (χ²).

10 fixtures × 5,000 simulations take 2.2 s (target ≤ 15 s). Cards, saves, defensive contributions and bonus come with M7–M10 in PR 5.

### 4.5 Defence (M7), saves (M8), cards (M9) and bonus (M10)

`fplh evaluate components --season 2022-23 --season 2023-24 --season 2024-25 --part cards --part saves --part bonus --part defence` refits every model at each deadline. It scores each one given the player's actual minutes; bonus is scored given the match's actual events. So each model is judged on what it adds to the simulator. Runtime is about 15 minutes. All CIs are 95 % gameweek-block bootstraps.

**M9 cards** (`models/cards.py`): a decayed Gamma–Poisson yellow rate per 90, shrunk to position × role.
- In FPL data a yellow never comes with a red, and there is never more than one yellow. So the simulator draws at most one yellow, removes it on a red, and sends off the player ∝ yellow rate.
- No game-state or referee term: yellow minutes are not identifiable, and the referee is known only after the match.

| Target | n | Model | Position rate | Difference (95 % CI) | ECE |
|---|---|---|---|---|---|
| Yellow (Brier) | 34,156 | 0.1121 | 0.1127 | −0.0006 [−0.0009, −0.0003] | 0.021 |

**M8 saves** (`models/gk.py`): saves ~ NegBin(m · g_k · (a + b·μ_opp)), where μ_opp is the opponent's M1 pre-match rate (point in time).
- Given μ, the match's realised goals add nothing (coefficient 0.04), so saves are drawn independently of simulated goals. Fitted: a ≈ 1.17, b ≈ 1.24, NegBin size ≈ 14.
- The keeper multiplier g_k is shrunk heavily; its spread is about ±2 %.
- 75 % of missed penalties are saved (FPL counts saved penalties as missed), so the simulator gives the opposing keeper on the pitch that share.

| Save points (log loss) | n | Model | Baseline | Difference (95 % CI) |
|---|---|---|---|---|
| vs league rate per 90 | 2,316 | 0.9683 | 1.0188 | −0.049 [−0.066, −0.033] |
| vs opponent only (g_k = 1) | 2,316 | 0.9683 | 0.9701 | −0.002 [−0.004, 0.001] |

The opponent's rate carries the model. The keeper multiplier is not significant, and shrinkage keeps it near 1. Mean saves: 3.10 predicted, 3.05 observed.

**M10 bonus** (`models/bonus.py`): BPS = effective weights × simulated events + a residual by position × minutes band + the player's shrunk mean residual (δ_p) + noise. Bonus then comes from the official allocation over players who played (`assign_bonus_array(eligible = minutes > 0)`).
- Effective weights come from least squares over the last two seasons, because tables change; for example a forward's goal is about 23, against the table's 24.
- δ_p (spread about ±1 BPS) carries passing and other actions the simulator does not generate.
- Baselines: "official" applies the YAML table to the same events with no residual; "position" is the position's bonus rate per appearance.

| Target | Model | Official table | Position | Model − official (95 % CI) |
|---|---|---|---|---|
| P(bonus > 0) (Brier) | 0.0356 | 0.0695 | 0.0939 | −0.034 [−0.037, −0.031] |
| E[bonus] (squared error) | 0.1428 | 0.2476 | 0.4464 | −0.106 [−0.115, −0.095] |
| E[bonus] (absolute error) | 0.1345 | 0.1493 | 0.3756 | −0.015 [−0.019, −0.011] |

These cover 34,295 player-fixtures. Against the position rate, every difference is −0.058 or larger, and every CI excludes 0.

**M7 defensive actions** (`models/defence.py`). Data exist only for 2016/17–2018/19 and from 2025/26, so the model is evaluated walk-forward within 2018/19 and within 2025/26. That is the documented exception to the untouched holdout (open question 1c). It is active in the simulator only when the season's rules score defensive contribution.

Definition drift:
- Each (position, action) rate per 90 is compared between eras. More than 10 % apart rescales the old counts.
- Tackles per 90 doubled in 2025/26 (×2.06 for DEF and MID); recoveries fell to about 0.73×; DEF CBI stayed within tolerance.
- The definitions changed, not the football: DEF CBI+T is about 7.5 per 90 in both eras.

The model:
- per-action decayed Gamma–Poisson rates, shrunk to position × role;
- an opponent multiplier (±5 %);
- a Gamma frailty per player-match shared by the actions, so the group total is NegBin (size 7.5 for DEF, 9.5 for MID). It is fitted by method of moments.
- No game-state term, because action timing is not in our sources.

Threshold Brier at the 2025/26 thresholds, against the position's mean rate with the same dispersion:

| Target | Window | n | Model | Position | Difference (95 % CI) | Mean predicted / observed |
|---|---|---|---|---|---|---|
| DEF CBI+T ≥ 10 | 2018/19 | 3,542 | 0.1225 | 0.1383 | −0.016 [−0.020, −0.012] | 0.210 / 0.171 |
| DEF CBI+T ≥ 10 | 2025/26 | 3,950 | 0.1297 | 0.1482 | −0.019 [−0.022, −0.015] | 0.215 / 0.208 |
| MID/FWD CBI+T+R ≥ 12 | 2018/19 | 6,174 | 0.0745 | 0.0885 | −0.014 [−0.017, −0.011] | 0.131 / 0.118 |
| MID/FWD CBI+T+R ≥ 12 | 2025/26 | 6,775 | 0.0596 | 0.0692 | −0.010 [−0.012, −0.008] | 0.096 / 0.088 |

- The 2018/19 over-prediction for DEF follows a fast fall in defenders' CBI+T across 2016–19 (8.2 → 7.4 → 6.5 per 90), which a 365-day half-life lags.
- In 2025/26 the bias is small (ECE 0.024 DEF, 0.015 MID/FWD). Its half-life is not tuned on 2025/26, to keep the holdout exception narrow.

**Simulator.** M7–M10 are optional inputs; without them those events stay at zero. Property tests (Hypothesis) check:
- at most one yellow, none with a red;
- saves only by keepers who played;
- saved penalties never exceed the opponent's misses;
- defensive actions only when the rules score them;
- no bonus for players who did not play, with at least 6 bonus points per match;
- points equal the sum of components;
- simulated saves and yellow rates match their means.

10 fixtures × 5,000 simulations with every component take 3.8 s.

### 4.6 The exit gate: the player simulator walk-forward

`fplh evaluate phase3 --season 2022-23 --season 2023-24 --season 2024-25` runs every model walk-forward at horizon 1. All are scored on the same 80,973 player-fixtures (33,208 with minutes) over 110 gameweek blocks. Runs are cached by predictor version, data and configuration. The first run takes about 80 minutes on 4 cores, including both ablations.

**The predictor** (`models/player_sim.py`), at each deadline:
- **Fits on 𝓘(D) only:** M4 (refit every 4 deadlines), M5/M6, M7–M10 and M8's pre-match rates (the M1 filter at D).
- **Team rates:** takes the Phase 2 fused rates (`FusedRates`, the fusion fitted on the five seasons before the first deadline) and inverts them to G6 nominal rates with the emulator.
- **Simulates** every fixture 2,000 times, giving per player:
  - expected points and the points pmf (−4…25);
  - P(60+), P(play), P(haul ≥ 10);
  - goals, assists, saves, bonus and clean-sheet probability.

| Model | MSE | MAE | Spearman within position | Spearman, played | Top-10 precision |
|---|---|---|---|---|---|
| **Simulator** | **3.633** | **0.970** | 0.700 | **0.379** | **0.441** |
| A5: raw goal rates | 3.648 | 0.972 | 0.699 | 0.374 | 0.440 |
| OpenFPL replica | 3.661 | 0.996 | 0.696 | 0.371 | 0.429 |
| A4: naive minutes | 3.833 | 0.987 | 0.700 | 0.350 | 0.435 |
| A0 | 4.015 | 1.001 | **0.714** | 0.331 | 0.410 |
| Last 5 | 4.354 | 1.050 | 0.681 | 0.283 | 0.384 |

**Gate.** Each benchmark needs an MSE difference with CI upper bound < 0, and a negative point estimate in at least 2 of 3 seasons. 95 % gameweek-block CI and DM p-values:

| Simulator − | Difference | 95 % CI | DM p | 2022/23 | 2023/24 | 2024/25 |
|---|---|---|---|---|---|---|
| OpenFPL replica | **−0.029** | [−0.048, −0.011] | 0.003 | −0.060 | −0.024 | +0.0005 |
| A0 | −0.403 | [−0.489, −0.331] | < 1e-4 | −0.409 | −0.392 | −0.345 |
| Last 5 | −0.736 | [−0.808, −0.667] | < 1e-4 | −0.779 | −0.709 | −0.677 |

**The gate passes.**
- **Against the replica the margin is narrow (0.8 % of MSE)** and 2024/25 is a tie. The simulator's clearer advantages are MAE, ranking among players who played, and top-10 precision, which are what transfers and captaincy use.
- **One guardrail is not ours:** A0 still ranks all rows best (Spearman 0.714). Its separation of benched from starting players matters most on the many rows with 0 points.

**Guardrails** (calibration of the simulator's distribution):

| Quantity | Mean predicted | Observed | ECE |
|---|---|---|---|
| P(60+ minutes) | 0.2825 | 0.2819 | 0.006 |
| P(haul ≥ 10) | 0.0167 | 0.0164 | 0.001 |
| Points | 1.1298 | 1.1307 | — |

CRPS of the points pmf is 0.627.

**Ablations** (simulator − variant, MSE):
- **A4 naive minutes** (shares of the last three matches): −0.200 [−0.216, −0.182]. Minutes is by far the largest single contribution, about 7 times the margin over the replica.
- **A5 raw goal rates** (no shrinkage): −0.016 [−0.023, −0.008].
- **A7 and A8 are not run.** A7 needs a substitution hazard, and the simulator draws exits from the empirical timing instead. A8 needs epistemic posterior draws, which are not built. Both move to Phase 5.

**Attribution** (`fplh evaluate attribution`, every second deadline, 1,000 simulations): the error y − ŷ⁰ splits exactly into three parts:
- **minutes** (ŷ¹ − ŷ⁰): the actual minutes imposed;
- **goal events** (ŷ² − ŷ¹): actual goals, assists, own goals, conceded and missed penalties also imposed;
- **the rest** (y − ŷ²): saves, cards, defensive actions, bonus and sampling.

The table gives the MSE of each forecast on 39,330 player-fixtures:

| Position | n | Forecast ŷ⁰ | Actual minutes ŷ¹ | Plus actual goal events ŷ² | Mean abs. minutes part | Mean abs. goal-events part | Mean abs. rest |
|---|---|---|---|---|---|---|---|
| All | 39,330 | 3.561 | 2.749 | 0.156 | 0.548 | 0.705 | 0.143 |
| GK | 4,282 | 2.334 | 1.856 | 0.449 | 0.225 | 0.492 | 0.224 |
| DEF | 13,174 | 3.398 | 2.618 | 0.130 | 0.576 | 0.758 | 0.141 |
| MID | 17,181 | 3.700 | 2.843 | 0.109 | 0.596 | 0.692 | 0.127 |
| FWD | 4,693 | 4.628 | 3.586 | 0.132 | 0.587 | 0.798 | 0.127 |

How the error divides:
- **Minutes, about 23 %.** Knowing who plays and for how long removes 0.81 of the 3.56. This part is partly reducible with team news (gap 1, the captured chance of playing).
- **Goal events, about 73 %.** Knowing the goals, assists and goals conceded removes most of the rest. This is mostly the irreducible luck of scoring, which better attack inputs (player-prop odds) can narrow only at the margin.
- **The rest, about 4 %.** Saves, cards, bonus and defensive actions. It is largest for goalkeepers (0.45: saves and bonus).

**`ep_next`** (`fplh evaluate ep-next --season 2026-27`):
- The bootstrap captures run every 3 hours from 30 September 2026 on the `Collect` workflow (`data-bronze` branch until a bucket is configured).
- The comparison takes the last capture at or before each deadline, no older than 4 days, and sums the simulator's expected points over a player's fixtures in the gameweek.
- It is tested on synthetic captures and runs as live gameweeks with results accrue (gap 2).

**Tuning experiments on 2021/22** (not a gate season, so the gate stays clean):
- The attack half-life and finishing shrinkage are already at their best.
- Penalty takers are the weakest role input. The predicted taker on the pitch takes 66–69 % of penalties; in 15–19 % of penalties the taker had no recorded attempt. This needs news, not modelling.
- Longer card memory (730 days) and one season of BPS weights each improve their Brier score in the fourth decimal. They are left for the next model version.

**Where the gains will come from.** Tuning is close to exhausted on these inputs. The next gains need new information:
- the captured news and chance of playing (the news overlay of gap 1);
- player-prop odds (anytime scorer, which also reveals penalty duty);
- stacking the simulator with the replica.

## 5. Phase 4: Decisions

| # | Ticket | Acceptance criteria |
|---|---|---|
| 4.1 | `optimize/milp.py` (HiGHS): §10.1 variables and constraints; FT banking linearised; chips; free-hit squad copy; sell-price formula; top-k pool | Toy instances with known optima; every constraint has a violating-input test |
| 4.2 | `evaluate/replay.py`: season replay with each benchmark's forecasts through the same optimizer | Deterministic given seeds |
| 4.3 | `delivery/ledger.py`: paper EV, fractional Kelly, CLV | No code path can place a bet (grep test for bookmaker write endpoints) |

Exit: replay beats the strongest baseline with a bootstrap CI excluding zero.

**Status:** built (PRs #11–#13).
- **v1 failed the exit gate** (§5.3): first in every season, but the margin over the OpenFPL replica was not significant.
- **Model v2 passes it** (§5.6): +3.7 points per gameweek over the replica, CI [+0.25, +7.19].

### 5.1 Game rules, scoring and prices

- **Chip rules corrected per season** (`configs/rules/fpl_*.yaml`, `Chips.windows`):
  - wildcards in two halves (2022/23 [2–16], [17–38]; 2023/24 [2–20], [21–38]; 2024/25 [2–19], [20–38]);
  - one free hit, bench boost and triple captain before 2025/26; two sets from 2025/26;
  - free-transfer banking capped at 2 before 2024/25 and 5 after;
  - special top-ups as `special_free_transfers` (2022/23 GW17: 15 after the World Cup; 2025/26 GW16: 5 for AFCON).
  - Seasons before 2022/23 are marked approximate. Every strategy runs the same rules, so comparisons stay fair.
- **`rules/team.py`:** official auto-substitution (bench order, GK for GK, formation minimums), vice-captain, and the triple-captain and bench-boost multipliers.
- **`features/prices.py`:** the price at each deadline (carried forward over blanks) and the official sell-price formula. Prices are game state, never a model feature.

### 5.2 The optimiser (`optimize/milp.py`)

- **ARCHITECTURE §10.1 in full** over `scipy.optimize.milp` (HiGHS, no new dependency):
  - squad, XI, captain and transfer flow;
  - budget with sell prices, hits and capped free-transfer banking;
  - the free-hit squad copy;
  - chip windows and allowances.
- **The player pool** is the squad plus the top 25 per position by discounted horizon EV, about 3,700 variables at H = 5.
- **Chips are decided by `plan_week`, not jointly** (the joint model was slow):
  1. a base plan without chips;
  2. bench boost and triple captain valued from the base plan;
  3. free hit and wildcard valued by solves forced this week;
  4. a chip is played only if this week is its best in the horizon and its gain beats an a-priori opportunity cost (wildcard 20, free hit and bench boost 15, triple captain 10).
- **Tested** on toy instances against brute force, with a violating input for every constraint.

### 5.3 The season replay: the exit gate

`fplh evaluate replay --season 2022-23 --season 2023-24 --season 2024-25` (about 30 minutes on 3 cores once forecasts are cached).

**The replay.** Every strategy:
- starts at GW2 with a free £100m squad (the spine is empty at GW1, so GW1 is excluded for all);
- solves the same MILP at each deadline (H = 5, δ = 0.9, β = 0.1, fixed a priori);
- trades at that deadline's prices;
- is scored on actual points with official auto-subs.

**The forecasts:**
- the simulator gives native horizon-5 forecasts;
- horizon-1 forecasters repeat their next-week per-fixture forecast for each future fixture (the known schedule, including doubles and blanks);
- "simulator (repeat)" isolates the value of forecasting ahead.

| Strategy | 2022/23 | 2023/24 | 2024/25 | Total | Hits | Transfers | Captain points per GW |
|---|---|---|---|---|---|---|---|
| **Simulator (horizon 5)** | 2,272 | **2,384** | **2,300** | **6,956** | **39** | 250 | 7.80 |
| Simulator (repeat) | **2,276** | 2,304 | 2,271 | 6,851 | 130 | 350 | 8.00 |
| OpenFPL replica (repeat) | 2,185 | 2,250 | 2,264 | 6,699 | 117 | 344 | 7.81 |
| A0 (repeat) | 2,050 | 1,958 | 2,212 | 6,220 | 100 | 312 | 7.82 |
| Last 5 (repeat) | 1,977 | 2,005 | 2,018 | 6,000 | 158 | 378 | 6.89 |

Points are net of hits over GW2–38. Every strategy used all of its chips.

**The gate** (simulator horizon 5 − baseline, points per gameweek, 110 gameweek blocks, 95 % CI):

| Baseline | Per GW | 95 % CI | DM p | Total | 2022/23 | 2023/24 | 2024/25 |
|---|---|---|---|---|---|---|---|
| OpenFPL replica (strongest) | +2.34 | [−1.26, +6.05] | 0.21 | +257 | +87 | +134 | +36 |
| A0 | +6.69 | [+3.11, +10.32] | 0.001 | +736 | +222 | +426 | +88 |
| Last 5 | +8.69 | [+4.84, +12.66] | < 0.001 | +956 | +295 | +379 | +282 |

**Exit gate: FAIL.**
- The simulator leads the replica in all three seasons, by 257 points in total, but the CI includes 0.
- That is as expected from Phase 3: the forecast edge over the replica was 0.8 % of MSE. Gameweek points are noisy (sd about 15), so 110 gameweeks detect about 4 points per gameweek, not 2.
- Forecasting ahead adds +0.95 points per gameweek over repeating ([−1.76, +3.51], p 0.48). It shows mostly in discipline: 39 hits against 130, and 100 fewer transfers.
- **A0 and last 5 are overconfident:** their mean expected XI score is 85–93 against 60 actual. The optimiser chases their noise. Simulator and replica are well calibrated (62 expected, 65–67 actual).

**What could pass the gate next:**
- **Better forecasts.** These are the Phase 3 levers: the news overlay, set-piece order and props.
- **More seasons**, which narrow the CI. 2021/22 has approximate rules; 2025/26 is the holdout.
- **Variance reduction:** a paired replay on common squads, so differences come from forecasts, not path dependence.

**Live comparison with the average manager.** `lake/silver/fpl_live.py` keeps FPL's `average_entry_score` and `highest_score` per gameweek, and `versus_average` scores a replay against them. It runs as live 2026/27 gameweeks finish.

### 5.4 The paper ledger (`delivery/ledger.py`)

`fplh evaluate ledger --season …` compares the fused match probabilities at each deadline with the best pre-match prices observable then (football-data's maximum, not closing):
- EV = p̂·o − 1;
- quarter Kelly above EV 3 %, capped at 5 % of the paper bankroll;
- CLV against the de-vigged closing price (Pinnacle, else the average), de-vigged one market at a time.

**Nothing can place a bet:** `tests/unit/test_ledger.py` fails on any HTTP write verb or bookmaker order endpoint in `src/`.

On 2022/23–2024/25 (499 paper bets):
- mean CLV −1.15 % [−2.16 %, −0.14 %], so **the model has no edge on the market**;
- ROI +2.3 % is noise at this sample size;
- the fused probabilities tie the market (Phase 2), and the bias experiments find a small away bias (+0.6 pp) that does not survive as CLV.

**Research (paper only, not built into the ledger):**
- A no-model consensus strategy bets soft-book prices above the Pinnacle-early fair price by more than 2 %. Over 2016/17–2024/25 it has +3.2 % mean CLV on 1,442 bets, positive in every one of nine seasons.
- Real accounts get limited quickly under such a strategy. It is recorded as a paper benchmark for Phase 5.

### 5.5 Odds collection on the free plan

The Odds API free plan gives 500 credits a month. `fplh collect odds --due` (hourly in `Collect`, secret `FPLH_ODDS_API_KEY`) plans the month with `collectors/odds_budget.py`.

**Spending priorities**, strictly in order:
1. match odds (h2h + totals, 2 credits) in the hour before each kickoff time, after the lineups;
2. the hour before that, to see the line move on team news;
3. gameweek deadlines;
4. anytime-scorer props per fixture (1 credit) at closing;
5. the same props at the deadline;
6. spare snapshots paced to the end of the month.

**Never-exceed guarantee:**
- Every run reads the live `x-requests-remaining` header.
- Every call is refused if it would leave fewer than 10 credits.
- Kickoffs seen earlier are remembered, so a round's later matches do not look like new deadlines.

In simulations of real 2025/26 months, runs spend 489–490 credits and never go below the floor, even with half the hourly runs missed.

### 5.6 Model v2: before Phase 5

Every idea was tested walk-forward and kept only if its 95 % gameweek-block CI excluded 0:
- choices made on 2021/22 (tuning);
- each final candidate measured once on 2022/23–2024/25;
- 2025/26 untouched.

The full log, including what was rejected and why, is in [EXPERIMENTS_V2.md](EXPERIMENTS_V2.md).

**Kept: transfer-news minutes** (`features/transfers.py`, `minutes="news"`).
- **The signal.** FPL's per-round transfer counts are made before the round's deadline (GW1 rows are 0), so they are public at the deadline. Owners selling a player en masse is the crowd's reaction to team news: among players M4 gave P(start) ≈ 0.87, the most-sold 1 % started 35 % of the time.
- **How it is served.** A derived view re-times the counts to the round's deadline, so no silver file changes and no cached run is invalidated. Live, our bootstrap captures stand in for the open window.
- **Leakage check.** It perturbs these columns by when they became public (`transfers.PRE_DEADLINE`).
- **Minutes gain on 2021/22.** M4 gains owners' selling, buying, ownership and team-mates' selling: P(start) Brier −7.9 %, P(appearance) −10.1 %.
- **Points on 2022/23–2024/25:**

| v2 (simulator + news) − | Δ MSE | 95 % CI | 2022/23 | 2023/24 | 2024/25 |
|---|---|---|---|---|---|
| v1 simulator | −0.097 | [−0.112, −0.083] | −0.092 | −0.110 | −0.088 |
| OpenFPL replica | −0.126 | [−0.146, −0.107] | −0.152 | −0.134 | −0.087 |

- v2's MSE is 3.536, against 3.633 for v1 and 3.661 for the replica. The margin over the replica is 4.3× the Phase 3 margin, positive in every season.
- Spearman within position is 0.722, now above A0. Top-10 precision is 0.447.

**Season replay** (2022/23–2024/25; same optimiser and rules; v2 = news-aware simulator, horizon 5):

| Strategy | 2022/23 | 2023/24 | 2024/25 | Total | Hits |
|---|---|---|---|---|---|
| **v2 (news, horizon 5)** | **2,391** | 2,339 | **2,377** | **7,107** | 39 |
| v1 simulator (horizon 5) | 2,272 | 2,384 | 2,300 | 6,956 | 39 |
| OpenFPL replica (repeat) | 2,185 | 2,250 | 2,264 | 6,699 | 117 |
| A0 (repeat) | 2,050 | 1,958 | 2,212 | 6,220 | 100 |
| Last 5 (repeat) | 1,977 | 2,005 | 2,018 | 6,000 | 158 |

| v2 − | Per GW | 95 % CI | DM p | Total |
|---|---|---|---|---|
| OpenFPL replica (strongest baseline) | **+3.71** | **[+0.25, +7.19]** | 0.034 | +408 |
| A0 | +8.06 | [+4.43, +11.66] | < 0.001 | +887 |
| Last 5 | +10.06 | [+6.39, +13.52] | < 0.001 | +1,107 |
| v1 simulator (horizon 5) | +1.37 | [−1.20, +3.97] | 0.31 | +151 |

**The Phase 4 exit gate now passes with v2:** it beats the strongest baseline with a CI excluding 0, and is ahead in all three seasons (+206, +89, +113).

**Kept: consensus-value paper betting** (`ledger --strategy consensus`, `evaluate live-ledger`).
- **Historical rule.** Bet a named soft book's pre-closing price above Pinnacle's power-de-vigged fair price. Over 2016/17–2024/25: 466 paper bets, CLV +2.9 % [+1.6, +4.2].
- **Live rule.** The Odds API's UK region has no Pinnacle, so the Betfair exchange is the anchor. As an anchor on 2022/23–2024/25 it gave +2.9 % [+1.3, +4.5] at EV > 3 %.
- **Why the model has no edge.** Our match model cannot beat the market: its deviations from the early line do not predict the line move (correlation −0.06 to +0.04).

**Rejected:**
- **Stacking layer** (simulator + replica + crowd + price): −0.010 [−0.020, +0.000] on 2021/22, because the news features already carry the crowd.
- **Replica blend:** hurts once news is in.
- **Home/away fix:** season swings, not a bias.
- **Lineup-aware team rates:** known absences are already in the market at the deadline.
- **Penalty-taker variants.**
- **vaastav `xP`:** recorded after the round, so it leaks.
- **Optimiser tuning:** 2021/22 winners lost on the test seasons.
- **Model/market hybrid ledger; news as a line-move signal; Asian handicaps:** small gain, deferred.
- **API-Football injuries:** post-match, so they leak.

## 6. Phase 5: Extensions, each gated by its ablation row

G7 lineup-aware rates · M11 v1 (LightGBM, A9) → v2 (multi-task network, A10) · anytime-scorer props (A11) · `optimize/saa.py` with rank-aware objective and CVaR (A12) · M12 price changes.

## 7. Phase 6: Operations

VPS with Dagster assets partitioned by gameweek; §12.4 schedules; inference at D − 24 h and D − 2 h (§12.5); `delivery/telegram.py`; Healthchecks.io heartbeats; §12.8 alerts; `fplh lake sync` to move `data-bronze` history into the bucket (verified by SHA-256 against the sidecars), then retire the branch fallback.

Exit: stable calibration across 10+ live gameweeks.
