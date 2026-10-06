# Model v2: night plan and experiment log

Living document, updated as results arrive. **Rule:** a change is kept only if it beats the
current model with a 95 % gameweek-block CI excluding 0.

**Seasons:**
- 2021/22 is the tuning season.
- 2022/23–2024/25 are the test seasons, scored once per final candidate.
- 2025/26 is the holdout and is never touched.

Status key: ✅ kept · ❌ rejected · ⏳ running · 🔜 next · 💤 later (needs data/keys)

## Results so far

### FPL forecasts

| # | Idea | Test | Result | Status |
|---|---|---|---|---|
| F1 | **Transfer-news minutes.** FPL round transfers (owners selling or buying) are public at the deadline and serve as a team-news proxy. They feed the minutes model, plus team-mates being sold. | M4 Brier on 2021/22 | P(start) −7.9 % [CI excludes 0], P(60+) −7.1 %, P(appear) −10.1 % | ✅ (minutes) |
| F1b | **F1 inside the full simulator** | Points MSE on 2021/22 (24,678 player-fixtures, 37 blocks) | **4.051 vs v1 4.194: −0.146 [−0.187, −0.113]; vs replica −0.164 [−0.223, −0.112]**; top-10 precision 0.456 → 0.474; Spearman (played) 0.350 → 0.363 | ✅ |
| F2 | Simulator + OpenFPL replica blend | Points MSE on 2021/22 | Helped v1 (4.173 vs 4.194), but **hurts the news simulator**: w = 0.5 +0.036 [+0.012, +0.062], w = 0.7 +0.011 [−0.003, +0.026] | ❌ (superseded by F1b); the stack starts from the simulator alone |
| F3 | **Stacking layer (M11 v1).** Simulator summary + replica + crowd (selling, buying, ownership) + price, as gradient-boosted trees starting from the news simulator. Walk-forward, trained on 2019/20–2020/21 and earlier 2021/22 rounds. | MSE vs the news simulator, 2021/22 | 300 rounds, depth 3: −0.005 [−0.020, +0.011]. 150 rounds, depth 2: −0.010 [−0.020, +0.000]. Blend 0.8: −0.002 [−0.019, +0.016]. The news features already carry the crowd's information. | ❌ (kept in code, opt-in) |
| F4 | Home/away bias fix | Fused home-goal residual vs results | +0.09, +0.08, −0.20 per season (SE 0.07): season swings, not bias | ❌ |
| F5 | Lineup-aware team rates (G7) | Market share in fused rates | 93 % of fixtures have prices at the deadline (fusion weight 0.83); known absences are already priced | ❌ |
| F6 | Penalty-taker variants (team-only, half-lives 90–730 d) | Taker hit rate 2020/21–2021/22 | ±2–3 pp, inconsistent | ❌ |
| F7 | vaastav `xP` (FPL's expected points) as a feature | Timing test | Correlates more with the same round's points (0.63) than the previous (0.53): recorded after the round, so it leaks | ❌ |

### ✅ Final test: v2 on 2022/23–2024/25

Measured once, on the same 80,973 player-fixtures and 110 gameweek blocks as the Phase 3 gate.

| Model | MSE | MAE | Spearman within position | Spearman (played) | Top-10 precision |
|---|---|---|---|---|---|
| **v2 (simulator + transfer news)** | **3.536** | **0.942** | **0.722** | **0.387** | **0.447** |
| v1 simulator | 3.633 | 0.970 | 0.700 | 0.379 | 0.441 |
| OpenFPL replica | 3.661 | 0.996 | 0.696 | 0.371 | 0.429 |
| A0 | 4.015 | 1.001 | 0.714 | 0.331 | 0.410 |
| Last 5 | 4.354 | 1.050 | 0.681 | 0.283 | 0.384 |

| v2 − | Δ MSE | 95 % CI | 2022/23 | 2023/24 | 2024/25 |
|---|---|---|---|---|---|
| v1 simulator | **−0.097** | [−0.112, −0.083] | −0.092 | −0.110 | −0.088 |
| OpenFPL replica | **−0.126** | [−0.146, −0.107] | −0.152 | −0.134 | −0.087 |
| A0 | −0.500 | [−0.585, −0.430] | −0.501 | −0.502 | −0.433 |
| Last 5 | −0.833 | [−0.904, −0.764] | −0.871 | −0.819 | −0.765 |

- The margin over the replica is 4.3× the Phase 3 margin (−0.029), and it holds in every season (2024/25 was a tie before).
- v2 is now also best on ranking all rows, which A0 held in Phase 3.

### ✅ Season replay: v2 passes the Phase 4 gate

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

### Paper betting ledger

| # | Idea | Test | Result | Status |
|---|---|---|---|---|
| B1 | **Consensus value:** named soft books' pre-closing price > Pinnacle power-de-vigged fair price | CLV 2016/17–2024/25 | 466 bets, +2.9 % [+1.6, +4.2], ROI +8 %; tune +2.9 %, test +3.0 % (n = 60) | ✅ in ledger |
| B2 | Hybrid: blend our model into the fair price | CLV 2022/23–2024/25 | −1.4 % | ❌ |
| B3 | Our model predicts the line move (early → close) | corr(model − early, close − early) | −0.06 to +0.04 (fused and M1) | ❌ (cause of no model edge) |
| B4 | FPL transfer news predicts the line move | corr, 2,280 fixtures | −0.06 (right sign, about 3 SE), top 5 % move only 0.46 pp | ❌ as a bet signal |
| B5 | Asian handicap consensus | CLV 2017/18–2024/25 | Bet365: almost no edges. Best price across books: +2.4 % [+1.0, +3.9], n = 134 | ➖ small |
| B6 | Betfair exchange as the venue (AH) | CLV | No edge | ❌ |
| B8 | **Betfair exchange as the fair-price anchor** (live has no Pinnacle in the UK region) | CLV 2022/23–2024/25 (exchange prices start 2022/23) | EV > 3 %: +2.9 % [+1.3, +4.5], n = 22. EV > 2 %: +1.7 % [+0.5, +2.8], n = 51. Market average anchor on 2016/17–2021/22: +2.9 % [+1.6, +4.3] | ✅ live anchor |
| B7 | **Live consensus tracker** (`fplh evaluate live-ledger`) over The Odds API snapshots (21 UK books) | Paper CLV as snapshots accrue | First 3 snapshots: 3 paper bets (Crystal Palace away at 5.5 at Brighton, EV +4.3 %), CLV so far +2.9 % (not yet against the close) | ⏳ forward test |

## Incidents

- **20:50 UTC: the container restarted.** Every running job was lost before finishing, so the queue restarted at 21:00 as one sequential pipeline with at most 2 processes. Contention had made the jobs about 5× slower. The cached runs and code survived.
- **API-Football probe** (side session, branch `claude/api-football-probe`):
  - Injury records are post-match cleanup, so they leak: unusable for backtests.
  - Lineups and player stats exist for 2022–24, but only after kickoff.
  - The current season needs a paid plan.
  - The account was suspended after a burst of requests; the owner must reinstate it in the dashboard.
  - ❌ No historical feature gain.

### Decisions (optimiser, season replay)

D1: 2021/22 replay with the v1 simulator (repeat mode), one setting changed at a time. Default is horizon 5, δ 0.9, β 0.1.

| Setting | Points | Hits | vs default |
|---|---|---|---|
| default | 1,955 | 81 | — |
| β 0.2 | 1,963 | 83 | +8 |
| β 0.05 | 2,062 | 78 | +107 |
| δ 0.8 | 2,131 | 65 | +176 |
| δ 1.0 | 2,084 | 88 | +129 |
| horizon 3 | **2,181** | **58** | **+226** |
| horizon 8 | 2,021 | 84 | +66 |

**Reading:**
- Shorter or more discounted planning takes fewer hits and scores more.
- But δ 0.8 and δ 1.0 both beat δ 0.9, so a single season is noisy (about ±100 points).
- **Test seasons** (v1 simulator, horizon-5 forecasts, 2022/23 + 2023/24 + 2024/25):

  | Setting | 2022/23 | 2023/24 | 2024/25 | Total | vs default |
  |---|---|---|---|---|---|
  | default | 2,272 | 2,384 | 2,300 | 6,956 | — |
  | horizon 3 | 2,222 | 2,364 | 2,278 | 6,864 | −92 |
  | δ 0.8 | 2,293 | 2,317 | 2,335 | 6,945 | −11 |

  Hits fall to 30 and 22 (from 39), but points do not rise.
- ❌ **Not adopted:** the 2021/22 gains were noise. The defaults stay.

## Tonight's queue (in order)

1. ⏳ F1b: simulator with news on 2021/22. Also 2017/18–2020/21 runs for stack training, and the test-season run.
2. 🔜 F3: tune the stack on 2021/22 (blend weight, depth, rounds, features), then one test on 2022/23–2024/25 against v1 and the replica.
3. 🔜 F8: per-position stack weights; the stack's calibration of P(haul) and ranking (top-10 precision, used by the optimiser).
4. ✅ Season replay with v2 forecasts (passes the gate) (simulator with news at horizon 5, stacked). Does v2 pass the Phase 4 gate against the replica?
5. 🔜 B7: live consensus paper tracker on The Odds API snapshots (about 10–15 UK books), logged per gameweek as data arrives.
6. ❌ D1 (done; defaults kept): optimiser tuning on the 2021/22 replay (bench weight β, discount δ, horizon 3/5/8, chip thresholds ×0.5/×1.5), applied unchanged to the test seasons.
7. 🔜 F9: card memory 730 d and one season of BPS weights (small known gains from the Phase 3 tuning).
8. 🔜 Docs, PR, merge on green CI.

## Data sources

| Source | Free? | What it adds | Status |
|---|---|---|---|
| FPL round transfers (vaastav, our captures) | Yes, already in the lake | News proxy (F1), crowd features (F3) | ✅ used |
| football-data.co.uk Asian handicap and Betfair columns | Yes, already in bronze | AH consensus (B5) | ➖ small; not ingested yet (would invalidate caches) |
| The Odds API (free 500 credits) | Yes, key set | Live 1X2 and totals from about 15 books, anytime-scorer props | ⏳ collecting; forward test B7 |
| ClubElo API | Yes | Team ratings | 💤 low expected gain (market already in fusion) |
| Betfair historical data, Basic plan | Free with a Betfair account | Minute-level exchange prices since 2016, incl. scorer markets | ❌ not accessible to the owner |
| API-Football free key (100 req/day) | Key set (`FPLH_API_FOOTBALL_KEY`) | Injuries, lineups | ❌ injuries post-match (leak); current season paid; account suspended after a burst |
| vaastav `xP` | Yes | — | ❌ leaks (F7) |
