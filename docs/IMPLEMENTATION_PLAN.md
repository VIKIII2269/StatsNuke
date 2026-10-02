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
| 3 | Match and player level (G4–G6, M4–M10, simulator) | Planned |
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
2. **The `ep_next` benchmark (G2) can be measured only live**, from our first captured deadline. The 2022/23–2025/26 tuning and holdout periods benchmark against OpenFPL and the naive floors instead. This changes the §13 Phase 3 exit criterion to "beats OpenFPL and naive floors walk-forward; beats `ep_next` on live gameweeks as they accrue".
3. **Pinnacle closing odds after the July 2025 API closure.** First Phase 1 task for football-data: check whether `PSCH/PSCD/PSCA` are still populated for 2025/26 and 2026/27. If they aren't, the match benchmark becomes de-vigged **market-average closing** (`AvgCH/AvgCD/AvgCA`), and G1 is restated against it.
4. **The shot-rate ↔ goal-rate link is under-specified (§7.3 vs §7.5).** The emulator takes fused **goal** rates as its inputs and maps them to base shot rates, `λ̄^S_k = λ̄_k / (team mean xG per shot × league finishing)`. The inversion and the intensity model then share one parameterisation. Frailty variance and state effects are fixed at their fitted values inside the emulator (open question 6).
5. **Emulator cost.** A grid of 60² points × 10⁵ sims × ~96 steps needs a **team-only simulator path** with no player allocation, lineups or bookings. Scoreline markets depend only on team intensities, red cards and frailty. Budget: 1–2 h on Colab CPU, cached by `goal_process` model version.
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
9. **Walk-forward compute.** A full NUTS refit every 4 gameweeks × 38 × 3 tuning seasons is about 30 refits per configuration. Run the ablation ladders with SVI/Laplace, and run NUTS only on finalists and for the holdout.
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

| # | Ticket | Acceptance criteria |
|---|---|---|
| 3.1 | `models/goal_process.py` (G4–G6): piecewise-exponential Poisson GLM, hierarchical state effects, red-card hazard, frailty, stoppage-time model | Dispersion direction reported (§11.6); each G-step beats the previous one or is dropped |
| 3.2 | `sim/emulator.py`: team-only grid, bicubic spline per market, L-BFGS-B inversion | Market reproduction error within tolerance; cached per model version |
| 3.3 | `models/minutes.py` (M4): staged LightGBM with monotone constraints, isotonic calibration, small-sample prior blend, horizon drift, news overlay (gap 1) | ECE ≤ 0.02 on start and 60+; A4 logged |
| 3.4 | `models/attack.py` (M5/M6): conjugate Gamma-Poisson with decay, Beta shot quality, shrunk finishing, penalties and own goals | A5 logged (low-minutes players) |
| 3.5 | `models/defence.py` (M7): NegBin by game state | Threshold Brier and PIT; walk-forward within 2025/26 + live (open question 1c) |
| 3.6 | `models/gk.py`, `models/cards.py`, `models/bonus.py` (M8–M10; BPS weights from the rules YAML `bps` block) | Save-point log loss; card Brier; bonus accuracy |
| 3.7 | `sim/simulator.py`: vectorised (sims × players) minute stepper, common random numbers, epistemic batches, DGW summation; calls `rules.engine.score_arrays` + `rules.bonus.assign_bonus_array` | §8.5 validation suite passes; property tests (goals conserve, ≤ 11 on pitch, sub limit) |
| 3.8 | `evaluate/attribution.py` (§11.5) | Decomposition sums to total error on every row |

Exit: §8.5 validation passes; the simulator beats OpenFPL and the naive floors walk-forward; the `ep_next` comparison runs on live gameweeks (gap 2).

## 5. Phase 4: Decisions

| # | Ticket | Acceptance criteria |
|---|---|---|
| 4.1 | `optimize/milp.py` (HiGHS): §10.1 variables and constraints; FT banking linearised; chips; free-hit squad copy; sell-price formula; top-k pool | Toy instances with known optima; every constraint has a violating-input test |
| 4.2 | `evaluate/replay.py`: season replay with each benchmark's forecasts through the same optimizer | Deterministic given seeds |
| 4.3 | `delivery/ledger.py`: paper EV, fractional Kelly, CLV | No code path can place a bet (grep test for bookmaker write endpoints) |

Exit: replay beats the strongest baseline with a bootstrap CI excluding zero.

## 6. Phase 5: Extensions, each gated by its ablation row

G7 lineup-aware rates · M11 v1 (LightGBM, A9) → v2 (multi-task network, A10) · anytime-scorer props (A11) · `optimize/saa.py` with rank-aware objective and CVaR (A12) · M12 price changes.

## 7. Phase 6: Operations

VPS with Dagster assets partitioned by gameweek; §12.4 schedules; inference at D − 24 h and D − 2 h (§12.5); `delivery/telegram.py`; Healthchecks.io heartbeats; §12.8 alerts; `fplh lake sync` to move `data-bronze` history into the bucket (verified by SHA-256 against the sidecars), then retire the branch fallback.

Exit: stable calibration across 10+ live gameweeks.
