# Open ML Research Problems in Sports Analytics and Forecasting (football-first), as of Nov 2026

Scope note: These notes come from about 20 web searches run on 2026-11-20. arxiv.org could not be fetched directly in this session because it is blocked by the session network allowlist. arXiv details therefore come from search-engine abstracts and snippets, not full-text reads. Items tagged **[prior knowledge, not re-verified]** come from the researcher's background knowledge. They are canonical, but they were not re-checked online this session. The report writer should treat them as lower-confidence or verify them. Market-related items are framed only as forecasting benchmarks: paper evaluation, no wagering.

---

## 1. Beating or matching betting markets (market efficiency as a forecasting benchmark)

### Takeaway
The consensus from 2024–2026 is this. Closing football odds are a near-efficient forecaster. Small, persistent biases exist in the 1X2 market, especially the favourite–longshot bias. These are mostly absorbed by the margin, and the Asian-handicap market looks efficient. "Tying the market on RPS" is therefore the realistic ceiling for public-data models. The open research problems are about *measuring* that tie rigorously and *explaining* where residual information lives, not about "beating" the market.

### Cited Findings
- Hegarty & Whelan, "Forecasting soccer matches with betting odds: A tale of two markets", *International Journal of Forecasting* 41(2):803–820 (2025), DOI 10.1016/j.ijforecast.2024.06.013. They confirm 1X2 inefficiencies and a strong favourite–longshot bias, citing Angelini & De Angelis 2019 for the earlier evidence. They argue the Asian handicap market produces efficient forecasts for the same matches, using a new odds-to-probability mapping. The working-paper version says efficiency holds "taking bookmaker margins as given". — [EconPapers](https://econpapers.repec.org/RePEc:eee:intfor:v:41:y:2025:i:2:p:803-820); [Whelan blog summary](https://karlwhelan.com/sports-betting-asian-handicap-vs-1x2/); [CEPR DP17949](https://repec.cepr.org/repec/cpr/ceprdp/DP17949.pdf)
- Whelan's companion work on returns to complex Asian-handicap bets appears in *Review of Behavioral Finance*. — [Emerald](https://www.emerald.com/insight/content/doi/10.1108/RBF-11-2023-0314/full/html); [MPRA](https://mpra.ub.uni-muenchen.de/117244/1/ComplexBets.pdf)
- An April 2026 arXiv preprint (2604.17194) proposes new methods for converting odds into probabilities. It uses 90,014 football matches across five bookmakers. This is directly relevant to the "de-vigging as a forecasting problem" sub-problem. — [arXiv 2604.17194](https://arxiv.org/abs/2604.17194) (abstract only, via search)
- An Aalto thesis found that a regression model using only public information can, on average, match odds-implied accuracy but cannot generate positive returns. Its conclusion: "slightly inefficient, but not profitably so". — [Aaltodoc](https://aaltodoc.aalto.fi/items/90aa4a21-b870-4662-bf27-b7bee781d04d)
- An Aalborg MSc thesis (2025) used XGBoost with a custom loss designed to decorrelate predictions from market odds, tested on the Betfair exchange for the top five leagues. Decorrelation from the market is an explicit modelling objective here. — [AAU PDF](https://projekter.aau.dk/projekter/files/784199250/MastersThesisGroup24.pdf)
- An NCI MSc thesis (2025) claims identifiable inefficiencies against Betfair pre-kickoff odds. This is unreviewed and contradicts the Aalto result. — [NORMA](https://norma.ncirl.ie/8547/)
- A Southampton PhD thesis (Oikonomidis) studied predicting *closing* odds from early odds, which is the line-movement prediction problem. — [eprints Soton](https://eprints.soton.ac.uk/376060/1/Oikonomidis_2C_20Anastasios_20-_20final_20thesis.pdf)
- Constantinou's QMUL PhD thesis (Bayesian networks) reported forecasts more accurate than market odds. It is older and pre-dates modern sharp markets. — [QMRO](https://qmro.qmul.ac.uk/xmlui/bitstream/handle/123456789/8404/Constantinou_A_PhD_final.pdf?sequence=1)
- LLM forecasting benchmarks around the 2026 World Cup consistently report that the market is unbeaten:
  - WC2026-Agents (arXiv 2607.17765) treats the pre-match market as a competitor. None of its four agents beat the market's Brier score, and flat-staking the market favourite out-earned all agents. — [arXiv 2607.17765](https://arxiv.org/abs/2607.17765)
  - In WorldCupArena (arXiv 2607.18084), the best model improves only slightly on market and human-fan baselines in strict result prediction. — [HF papers](https://huggingface.co/papers/2607.18084.md)
  - LLM-SoccerArena (arXiv 2607.24573) is a prospective live leaderboard for LLM football forecasts. Its prompts deliberately do not supply odds, injuries or lineups, so models must web-search for them. — [arXiv 2607.24573](https://arxiv.org/pdf/2607.24573); [LMU press release](https://www.lmu.de/en/newsroom/news-overview/news/fifa-world-cup-how-well-can-ai-predict-sports-results-675684e5.html)
- **[prior knowledge, not re-verified]**
  - Štrumbelj (2014, *IJF*, "On determining probability forecasts from betting odds") showed that Shin-style de-vigging beats basic normalisation.
  - Constantinou & Fenton (2012, *JQAS*) established RPS as the standard metric for football 1X2 forecasts.
  - Wheatcroft (2020/2021, *IJF*/*JQAS*) studied "profiting from overreaction" and the predictive power of odds.
  - These are the standard citations for any market-anchored paper.

### Inferences
- **Open problem 1a: a rigorous, public "model vs. closing line" forecasting benchmark.**
  - Why open: most efficiency papers are econometric, testing returns. They do not compare proper scores with block-bootstrap/DM uncertainty on a point-in-time feature set. The LLM World Cup benchmarks are tiny (64–104 matches) and single-tournament.
  - Solo contribution: StatsNuke's walk-forward EPL evaluation (RPS, log loss, gameweek-block bootstrap, DM tests, CLV ledger) could be packaged as a multi-season benchmark with frozen point-in-time snapshots. Baselines would be the opening line, the closing line, de-vig variants, Dixon–Coles, Elo, and the hierarchical model.
  - Compute: CPU.
  - Success metric: RPS/log-loss deltas with confidence intervals, plus encompassing-test coefficients (does the model add information beyond the close?).
- **Open problem 1b: opening-to-closing line movement as a forecasting target.** Predict the *closing* probability from information at time t, then ask how much the model's forecast at t explains movement to the close. This is the paper-only analogue of CLV. It is publishable as "information-arrival" research. Opening and closing odds are in football-data.co.uk, and intraday snapshots can come from The Odds API.
- **Open problem 1c: de-vigging and market aggregation.** The best mapping from multi-book odds to probabilities is still contested (Shin, power, margin-proportional, logit, and the new 2026 methods). The Asian-handicap and 1X2 discrepancy found by Hegarty & Whelan suggests a fusion model across markets (1X2, AH, totals, scorer props) is underexplored.
- **Open problem 1d: forecast encompassing and "where is the residual signal".** Known candidate residuals are favourite–longshot bias, draws, early-season priors, promoted teams, and team-news timing. A per-segment encompassing analysis is cheap and publishable in *IJF*/*JQAS*.
- Framing advice: present the work as "the market as the strongest public forecaster; does a transparent generative model encompass it?" This fits the user's no-wagering constraint and current journal tastes.

### Gaps
- Could not read the full text of arXiv 2604.17194 or the full Hegarty & Whelan paper for exact numbers.
- Found no 2025–2026 peer-reviewed study showing a public-data EPL model with a statistically significant proper-score edge over *closing* odds.
- Did not verify Pinnacle-specific closing-line studies.

---

## 2. Player-level forecasting: minutes and rotation (the largest error source)

### Takeaway
No public benchmark or standard model exists for predicting player minutes or starts in football. Published work splits into four strands: load-based regression, informal fantasy demos, binary played/not-played classifiers, and rotation-driver analyses. This looks like the single clearest gap where the StatsNuke minutes model, which uses FPL transfer activity as a team-news proxy, could become a citable contribution.

### Cited Findings
- OpenFPL (Groos, arXiv 2508.09992, 2025) avoids proprietary "expected minutes" entirely and uses the categorical FPL API availability tags instead. It uses position-specific ensembles trained on FPL plus Understat data for 2020/21–2023/24. It reports accuracy comparable to the leading commercial service and better for high-return players. Code and models are MIT-licensed. — [arXiv 2508.09992](https://arxiv.org/html/2508.09992v1); [Emergent Mind summary](https://www.emergentmind.com/topics/openfpl)
- MDPI *Applied Sciences* (2026) predicts the match minutes a player can handle. It uses Random Forest regression on ML fitness and fatigue indices plus 7- and 28-day load variables. This requires private GPS data. — [MDPI 16(4):2139](https://www.mdpi.com/2076-3417/16/4/2139)
- Kitman Labs reports ML models of coach rotation behaviour. They found about 2–3 lineup changes per team match-to-match, with drivers ranked as rest days, recent performance, competition type and travel. This is an industry blog, not peer-reviewed. — [Kitman Labs](https://www.kitmanlabs.com/?p=26152)
- An open-source FPL project recast minutes as a binary target because the many zeros swamp regression. — [GitHub fpl-ai](https://github.com/majiajue/fpl-ai)
- Lineup *optimisation* work (arXiv 2303.12385) notes that the best-rated XI is not necessarily the win-maximising XI. This is relevant to modelling manager choice. — [arXiv 2303.12385](https://ar5iv.labs.arxiv.org/html/2303.12385)
- A Splunk fantasy write-up forecasts minutes with a Kalman-filter state space model (informal). — [Splunk blog](https://www.splunk.com/en_us/blog/platform/how-to-win-at-fantasy-football-with-splunk-and-machine-learning-part-2.html)
- Fixture-congestion evidence is observational:
  - Bengtsson et al. (2013) studied 27 teams over 11 seasons. Muscle injury RR was 1.32 (95% CI 1.15–1.51) with ≤4 days of recovery versus ≥6, with little effect on performance. — [PubMed 23851296](https://pubmed.ncbi.nlm.nih.gov/23851296/)
  - A 2017 analysis of 133,170 player-match observations found no total-injury difference but lower muscle injuries with ≥6 days between matches. — [BJSM](https://bjsm.bmj.com/content/early/2017/11/03/bjsports-2016-097399)
  - A 2023 systematic review found five of eight studies reporting higher match injury incidence under congestion, with "no clear patterns or causes". — [PMC9758680](https://pmc.ncbi.nlm.nih.gov/articles/PMC9758680)

### Inferences
- **Open problem 2a: a public, point-in-time "minutes/start" benchmark.**
  - Task: for each squad player and gameweek, predict P(start), P(sub appearance), P(60+ minutes) and the minutes distribution. The prediction is made at fixed information cut-offs, e.g. T−72h, T−24h and T−1h, before lineup release.
  - Why open: no shared dataset exists, and most models leak post-lineup information.
  - StatsNuke's append-only data lake is exactly what is needed: FPL API snapshots of status, news, chance_of_playing, transfer deltas and prices.
  - Data: FPL API plus vaastav history (public). Compute: CPU.
  - Metrics: Brier/log loss for start probabilities, CRPS for the minutes distribution, calibration curves, and downstream FPL points MSE.
- **Open problem 2b: crowd signals as team news.** Using FPL transfer flows as a revealed-information signal is a novel "wisdom of the crowd" angle and appears unpublished. A paper could quantify information arrival: how much of lineup surprise is predictable from transfer and price-change flow at each lead time.
- **Open problem 2c: hierarchical rotation models.** Possible ingredients: manager-specific rotation propensity, congestion (European fixtures, cups), the substitution-time distribution (the five-sub rule post-2020), and return-from-injury minutes ramps. Partial pooling across managers and players is natural in Bayesian or state-space form.
- **Open problem 2d: causal effect of rest on performance.** The literature is associational (see above). Natural experiments could include TV-driven rescheduling, cup replays and international breaks. These are feasible with public fixtures plus xG, using difference-in-differences or matching.

### Gaps
- Found no paper benchmarking FPL-API-based minutes prediction.
- Found no academic evaluation of FPL Review's (commercial) minutes projections.
- The proprietary "expected minutes" methods from commercial services are undocumented.

---

## 3. Fantasy sports optimisation research (FPL)

### Takeaway
Academic FPL work is thin and mostly evaluates single seasons with deterministic optimisers. OpenFPL (2025) is the only credible open forecaster benchmarked against a commercial service. Several open problems are wide open for a solo researcher with a mature simulator and MILP: stochastic multi-period optimisation, chip timing, decision-quality evaluation, and a standard benchmark.

### Cited Findings
- OpenFPL (Groos, arXiv 2508.09992) is open source and uses public data only. It claims to rival commercial services. — [arXiv](https://arxiv.org/pdf/2508.09992)
- Ramezani & Dinh, "A data-driven framework for team selection in FPL" (arXiv 2505.02170, v1 May 2025, v3 Jan 2026). It builds deterministic and robust MILPs for XI, bench and captain. Tested on 2023/24, it finds ARIMA the most consistent predictor, with weighted averages and Monte Carlo competitive. It cites Matthews (2012), who modelled FPL as a Bayesian belief-state MDP. — [arXiv 2505.02170v3](https://arxiv.org/abs/2505.02170v3); [HTML v2](https://arxiv.org/html/2505.02170v2)
- Frees, Ravella & Zhang (arXiv 2405.02412, 2024) use CNNs and transfer learning for EPL player performance forecasting. — [arXiv 2405.02412](https://arxiv.org/html/2405.02412v1)
- Santoro (Univ. Bologna MSc) combines ML point prediction with MIP team selection, including an optimisation-under-uncertainty section, evaluated on 2023–24. — [AMS Laurea](https://amslaurea.unibo.it/35674/1/Tesi_Lorenzo_Santoro.pdf)
- **[prior knowledge, not re-verified]**
  - Matthews, Ramchurn & Chalkiadakis, "Competing with humans at Fantasy Football", AAAI 2012, is the canonical belief-state MDP paper.
  - Bonomo et al. (2014) studied fantasy football via MILP for the Argentine league.
  - Community MILP tooling exists in Sertalp Cay's "fpl-optimization" GitHub repo (FPL Review collaborations).

### Inferences
- **Open problem 3a: an open FPL benchmark with point-in-time data.**
  - What exists: OpenFPL evaluates on held-out seasons, but no standard frozen dataset of "what was knowable before each deadline" has been published. vaastav's repo is *post-hoc* history; it is not point-in-time and has no news or availability snapshots.
  - Contribution: release StatsNuke's snapshot lake (licensing permitting) with task splits. Tasks would be (i) player points distribution, (ii) minutes, and (iii) end-to-end season management. Baselines would include OpenFPL and the user's simulator.
  - This is a clear gap and a natural MLSA or *JQAS* data/benchmark paper.
- **Open problem 3b: evaluating decision quality, not just prediction error.**
  - Proposed evaluation: counterfactual season replays with a fixed optimiser and swapped forecasters, scored as points regret against hindsight-optimal or against the population. Use multiple seasons and bootstrap.
  - Why it matters: most papers report MSE. The user's finding that the simulator beats OpenFPL should be shown in *decision* terms, because better MSE does not imply better rank.
  - The "predict-then-optimise" and decision-focused learning literature (SPO+, by Elmachtoub & Grigas **[prior knowledge]**) has not, to my knowledge, been applied to FPL.
- **Open problem 3c: stochastic and risk-sensitive multi-period optimisation.**
  - Elements: chip timing (wildcard, free hit, bench boost, triple captain) as an optimal-stopping or dynamic-programming problem; rank-dependent risk (variance-seeking when chasing); template versus differential effective-ownership game theory.
  - The robust MILP in 2505.02170 is a first step, but no paper combines simulator-based scenario sets with multi-gameweek MILP and evaluates rank outcomes.
- **Open problem 3d: rule-change robustness.** FPL scoring and chip rules change between seasons (e.g., 2025/26 defensive contribution points **[prior knowledge, not re-verified]**). Transfer of models across rule regimes is a domain-shift problem, testable with vaastav 2016–2026.

### Gaps
- No public leaderboard comparing FPL Review, OpenFPL and others prospectively was found.
- I did not confirm whether OpenFPL has been updated beyond the 2025 paper, or whether a 2025/26 or 2026/27 evaluation exists.
- Academic FPL papers in 2026 beyond the v3 of 2505.02170 were not found; the search may have missed some.

---

## 4. xG / xT / possession-value models (VAEP, EPV)

### Takeaway
Possession-value models are mature as tools (socceraction/VAEP, xT, EPV), but their *evaluation* is unsolved. Aggregate log loss and calibration do not test whether models rank game states correctly, and uncertainty is rarely quantified. Benchmarks with expert-labelled state pairs (2025) are a fresh, tractable direction on public event data.

### Cited Findings
- Overmeer et al. (arXiv 2502.02565, 2025; also SCITEPRESS 2025), "Revisiting Expected Possession Value in Football".
  - Prior work reports aggregate loss and calibration, which miss whether a model ranks closely related states correctly.
  - They introduce the OJN-Pass-EPV benchmark of 50 expert-validated game-state pairs. Their U-Net picks the higher-value state 78% of the time.
  - Prior models (e.g., Fernández et al.'s 2021 EPV) ignore ball height.
  - The benchmark is extendable to dribbles, shots and overall EPV.
  - [arXiv 2502.02565](https://arxiv.org/pdf/2502.02565); [SCITEPRESS](https://scitepress.org/publishedPapers/2025/137843/pdf/index.html)
- Fernández, Bornn & Cervone (2021) present a decomposed EPV from tracking data with calibrated component merging. — [arXiv 2011.09426](https://arxiv.org/pdf/2011.09426); [PMC8570314](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8570314/)
- A practitioner post argues that VAEP and other event-only models have unquantified epistemic uncertainty from missing tracking context. It demonstrates bagging on VAEP. — [OpenGoalApp blog](https://opengoalapp.github.io/Knowing-When-Your-Model-Doesn't-Have-A-Clue)
- EPV for control and duel actions is also under study (arXiv 2406.00814). — [arXiv 2406.00814](https://arxiv.org/pdf/2406.00814)
- Davis et al., "Methodology and evaluation in sports analytics: challenges, approaches, and lessons learned", *Machine Learning* 113(9):6977–7010 (2024). It flags four issues: dependency-aware data partitioning, the need for calibrated probabilities, uncertainty quantification and explainability, and verification of learned models. — [Springer](https://link.springer.com/article/10.1007/s10994-024-06585-0); [Lirias PDF](https://lirias.kuleuven.be/retrieve/671769)
- MLSA 2025 (12th workshop, Porto, CCIS vol. 2833) had 11 full papers from 24 submissions. Soccer papers included "Pitch-wide space evaluation for soccer transitions" and "What Makes a Dribble Successful? Insights From 3D Pose Tracking Data". — [Springer book](https://link.springer.com/book/9783032151643)
- **[prior knowledge, not re-verified]**
  - Decroos et al., VAEP (KDD 2019), with the socceraction library ([GitHub](https://github.com/ML-KULeuven/socceraction)).
  - Singh's Expected Threat (xT, 2018 blog).
  - Robberechts & Davis on xG model pitfalls and calibration.
  - Mead et al. (2023, PLOS ONE) on xG with player and team effects.

### Inferences
- **Open problem 4a: xG calibration and "finishing skill" shrinkage.** How much player and team finishing skill survives regression to the mean, and how does it differ across providers (Understat vs. StatsBomb vs. Opta)? This is directly useful to a match model that uses xG for team strength. Hierarchical Bayes on Understat shots is CPU-only.
- **Open problem 4b: provider disagreement as noise.** Quantify how xG differs between Understat and StatsBomb on overlapping matches. Then measure how much the choice of xG source changes downstream match-forecast RPS. Few papers do this.
- **Open problem 4c: ranking-based and pairwise evaluation of possession-value models (EPV/VAEP).** Extend the OJN-style expert-pair benchmark to event-only models on StatsBomb open data, and report uncertainty (ensembles or conformal prediction). This is solo-feasible on CPU or one GPU.
- **Open problem 4d: does possession value predict future results better than xG?** Run a predictive-validity test: does team-level VAEP or xT per match improve out-of-sample match-forecast RPS beyond xG in a market-anchored model? StatsNuke is well placed to answer this. A negative result is still publishable.

### Gaps
- Did not find a 2026 survey of possession-value evaluation.
- The StatsBomb Conference 2025/2026 "open problems" agenda was not retrieved.

---

## 5. Event-data and tracking-data foundation models (2024–2026)

### Takeaway
Next-event transformer models on event streams are now an active line (LEMs 2024, "A Foundation Model for Soccer" 2024, RisingBALLER 2024, EventGPT 2025/26, an event-representation transformer in 2026). Open issues include training only on proprietary data, the lack of shared downstream benchmarks, and unproven value for *match-outcome* or *player-points* forecasting versus simple models. A solo researcher can train small models on StatsBomb open data on a single GPU.

### Cited Findings
- Mendes-Neves et al., Large Events Models (arXiv 2402.06820, "Forecasting Events in Soccer Matches Through Language"; and arXiv 2402.06815 on fine-tuned LEMs for player performance in different contexts). They treat a match as a token sequence for next-event prediction. The v1 used three MLPs; the later version uses a single tokenised model. Applications are match simulation and player evaluation. — [arXiv 2402.06820](https://arxiv.org/html/2402.06820v2); [arXiv 2402.06815](https://arxiv.org/html/2402.06815v2); [Springer ML journal](https://link.springer.com/article/10.1007/s10994-024-06606-y)
- "A Foundation Model for Soccer" (arXiv 2407.14558, 2024) is a transformer decoder for next action. It was trained on three seasons of one pro league and compared against Markov and MLP baselines. — [arXiv 2407.14558](https://arxiv.org/html/2407.14558v1)
- RisingBALLER (arXiv 2410.00943, 2024) treats players as tokens with masked-player pretraining. It uses StatsBomb 2015/16 top-five-league event data, which is public. — [arXiv 2410.00943](https://arxiv.org/pdf/2410.00943)
- EventGPT (arXiv 2512.17266, revised Mar 2026) is a player-conditioned autoregressive transformer. It predicts the next action, its location, timing and value. Trained on five EPL seasons, it beats sequence baselines on next-event accuracy and spatial precision. — [arXiv 2512.17266](https://arxiv.org/html/2512.17266v3)
- arXiv 2606.09327 (2026) is a transformer producing dense embeddings of football events from categorical features. — [arXiv 2606.09327](https://arxiv.org/html/2606.09327v1)
- Event2Tracking (Stats Perform/QUT, AAAI 2025) reconstructs multi-agent trajectories from broadcast footage using long-term multimodal context. — [AAAI](https://ojs.aaai.org/index.php/AAAI/article/view/33289/35444)
- **[prior knowledge, not re-verified]** TacticAI (Wang et al., *Nature Communications* 2024, DeepMind and Liverpool FC) uses geometric deep learning on corner-kick tracking data. — [Nature Comms](https://www.nature.com/articles/s41467-024-45965-x)

### Inferences
- **Open problem 5a: a shared downstream benchmark for football sequence models.** Candidate tasks are next-event prediction, xG, possession outcome, and match-outcome forecasting from the first N minutes. Every paper uses different (often proprietary) data. A StatsBomb-open-data benchmark suite with fixed splits is a clear gap.
- **Open problem 5b: do foundation-model embeddings improve season-level or player-level forecasts?** This is untested. For example, use player embeddings (RisingBALLER-style) as priors for promoted or new-transfer players in an FPL model, where data sparsity is worst (the cold-start problem).
- **Open problem 5c: calibrated match simulation.** LEM-style generators simulate matches. Are the simulated scorelines calibrated against closing odds? This is a cheap, rigorous test of generative football models that is missing from the literature.
- Compute: small transformers (≤50M parameters) on StatsBomb open data fit on one consumer GPU in hours to days.

### Gaps
- Could not verify whether any 2026 open-weights "football foundation model" with a public checkpoint exists.
- No large tracking foundation model trained on public data was found.

---

## 6. Calibration and uncertainty in sports predictions

### Takeaway
Calibration is explicitly flagged as under-handled (Davis et al. 2024). Open problems include calibration of *joint* and *derived* quantities (scorelines, player points distributions, captaincy tails), forecast evaluation with dependent data, and distribution-free uncertainty. These have barely been applied to FPL or player props.

### Cited Findings
- Davis et al. (2024, *Machine Learning*) state that well-calibrated probabilities are needed in many sports tasks. They also call for dependency-aware partitioning, uncertainty quantification and verification. — [Springer](https://link.springer.com/article/10.1007/s10994-024-06585-0)
- Hegarty & Whelan (2025) document miscalibration (favourite–longshot bias) in 1X2 odds but not in the AH market. Market-implied probabilities are therefore not uniformly calibrated. — [EconPapers](https://econpapers.repec.org/RePEc:eee:intfor:v:41:y:2025:i:2:p:803-820)
- Overmeer et al. (2025) note that aggregate calibration is insufficient for evaluating possession-value models. — [arXiv 2502.02565](https://arxiv.org/pdf/2502.02565)
- The 2023 Soccer Prediction Challenge's evaluation page defines scoring for exact-score and probability tasks. — [Challenge evaluation](https://sites.google.com/view/2023soccerpredictionchallenge/challenge-evaluation)

### Inferences
- **Open problem 6a: calibration of player-points distributions.** FPL points are zero-inflated, multi-modal, and driven by heavy-tailed hauls. Use PIT histograms, CRPS, and quantile calibration per position, price band and minutes bucket. Optimisers rely on tails (captaincy), yet no FPL paper reports distributional calibration. StatsNuke's simulator can provide this directly.
- **Open problem 6b: proper scoring for correlated multi-target forecasts.** Players within a team are correlated through team goals. Joint calibration matters for squad variance. Energy score and variogram score are standard in weather forecasting, but they are unused in sports.
- **Open problem 6c: evaluation with temporal dependence.** Gameweek-block bootstrap and DM tests are what StatsNuke does. Documenting them as a methodology note (comparison against naïve i.i.d. CIs) would directly answer the call in Davis et al.
- **Open problem 6d: conformal prediction for sports.** Adaptive or online conformal intervals for player points and minutes under distribution shift (transfers, managerial change) are underexplored. This is CPU-feasible.

### Gaps
- No 2025–2026 paper on distributional calibration of fantasy-sports forecasts was found.

---

## 7. Team-news and injury modelling from text (LLMs)

### Takeaway
LLM-extracted context gives small, conditional gains: about 2.6% in one 2026 thesis, and only when claims are checked against statistics. LLM agents searching the web do not beat markets. Structured, timestamped team-news extraction for minutes forecasting, with a leakage-safe point-in-time design, appears unaddressed.

### Cited Findings
- A Porto MSc thesis (2026) found that LLM-extracted pre-match media context helps next-match player-rating prediction. The gain was net positive only when editorial claims were verified against statistical records, at about 2.6%. — [Repositório Aberto U.Porto](https://repositorio-aberto.up.pt/bitstream/10216/175998/2/787181.pdf)
- SUMMIR (arXiv 2604.04947, 2026) ranks LLM-generated sports insights from 7,900 news articles covering 800 matches in four sports, with hallucination detection. — [arXiv 2604.04947](https://arxiv.org/html/2604.04947v1)
- LLM-SoccerArena, WorldCupArena and WC2026-Agents (all July 2026) show that LLM and agent forecasters do not reliably beat markets. — [2607.24573](https://arxiv.org/pdf/2607.24573); [2607.18084](https://huggingface.co/papers/2607.18084.md); [2607.17765](https://arxiv.org/abs/2607.17765)
- Injury forecasting on arXiv mostly uses monitoring data, not text. For example, there is a DeepHit survival model on elite women's football (arXiv 2601.19479). — [arXiv 2601.19479](https://arxiv.org/html/2601.19479v1); older: [arXiv 1609.07480](https://arxiv.org/pdf/1609.07480)
- An ACL 2026 Findings paper appeared in the search results for this topic. Its specific content was not verified. — [ACL Anthology 2026.findings-acl.1851](https://aclanthology.org/2026.findings-acl.1851.pdf)

### Inferences
- **Open problem 7a: a timestamped team-news extraction benchmark.**
  - Proposal: a dataset of press-conference quotes and FPL news strings, each with a timestamp, paired with realised starts and minutes. Tasks would be (i) extraction (player, status, expected return date) and (ii) incremental forecast value over FPL status flags and transfer flows.
  - Leakage control (article timestamps versus lineup release) is the methodological crux. LLM training-data contamination of past news is also a concern, so prospective evaluation is preferred.
- **Open problem 7b: crowd versus text.** Measure whether FPL transfer flow already subsumes news-text signal. StatsNuke's finding suggests the crowd aggregates team news. A direct comparison would be novel.
- **Open problem 7c: injury return-time survival models** from public injury-history sources (e.g., FPL news strings, historical news), with return-to-minutes ramps.

### Gaps
- Did not find any paper using FPL "news" fields or press conferences for minutes forecasting.
- Public injury databases (e.g., Transfermarkt) have unclear licensing; this was not researched.

---

## 8. Counterfactual and causal analyses (managers, rest, rotation)

### Takeaway
Causal sports analytics remains mostly associational. Rest and congestion effects on performance are weak or mixed. Clean natural experiments using public data (rescheduling, manager sackings with regression-to-the-mean controls, substitution timing) are tractable for a solo researcher.

### Cited Findings
- Congestion studies (2013, 2017, the 2023 review) are observational with conflicting total-injury findings. Bengtsson 2013 found little congestion-performance link. — [PubMed](https://pubmed.ncbi.nlm.nih.gov/23851296/); [BJSM 2017](https://bjsm.bmj.com/content/early/2017/11/03/bjsports-2016-097399); [PMC9758680](https://pmc.ncbi.nlm.nih.gov/articles/PMC9758680)
- Davis et al. (2024) emphasise that contextual confounders and dependencies complicate evaluation. — [Springer](https://link.springer.com/article/10.1007/s10994-024-06585-0)
- **[prior knowledge, not re-verified]** Manager-sacking "new manager bounce" studies (e.g., van Ours & van Tuijl, and others) attribute much of the bounce to regression to the mean.

### Inferences
- **Open problem 8a: new-manager effects using market-implied strength as the counterfactual.** Market odds before and after a sacking give a forward-looking counterfactual baseline. Compare realised xG with market-expected xG, which avoids regression-to-the-mean artefacts. Data: football-data.co.uk plus Understat. CPU.
- **Open problem 8b: rest-day effects on xG** using quasi-random TV rescheduling. EPL match times shift for TV picks and European fixtures, so rest differences arise for reasons unrelated to team quality. Use difference-in-differences or a regression discontinuity on days of rest.
- **Open problem 8c: substitution timing and in-match decisions.** StatsNuke's in-match goal process could estimate counterfactual value of substitution timing, given public minute-level event data (StatsBomb).

### Gaps
- No 2024–2026 causal-design paper on EPL rest effects was found in searches.

---

## 9. Public datasets, benchmarks, and venues

### Takeaway
Public data in late 2026 is rich for events (StatsBomb open data, Wyscout public dataset), sparse for tracking (Metrica sample, SkillCorner's 10 A-League matches, the PFF World Cup 2022 set), and excellent for FPL (FPL API, vaastav). The main venues for this work are MLSA@ECML-PKDD, *JQAS*, the *IJF* sports-forecasting track, the *Machine Learning* journal special issues, and MIT Sloan SSAC.

### Cited Findings
- **SkillCorner open data:** 10 matches of broadcast tracking plus derived "Dynamic Events" from the 2024/25 Australian A-League. SkillCorner also opened Dynamic Events for World Cup finals 2006–2022 through an explorer. — [SkillCorner Open Data](https://skillcorner.github.io/opendata/); [SkillCorner WC finals explorer](https://skillcorner.com/articles/world-cup-finals-interactive-explorer)
- **PFF FC (now Gradient Sports):** a free 2022 World Cup tracking plus event dataset, released 2024, with kloppy loaders. — [PFF FC blog](https://www.blog.fc.pff.com/blog/pff-fc-release-2022-world-cup-data); [enhanced dataset](https://www.blog.fc.pff.com/blog/enhanced-2022-world-cup-dataset); [Gradient repost](https://www.gradientsports.com/blog/pff-fc-release-2022-world-cup-data)
- **StatsBomb:** free 2022 World Cup event data, part of StatsBomb open data. — [StatsBomb blog](https://blogarchive.statsbomb.com/news/statsbomb-release-free-2022-world-cup-data/)
- **[prior knowledge, not re-verified]**
  - StatsBomb open data repo: https://github.com/statsbomb/open-data
  - Metrica sample tracking: https://github.com/metrica-sports/sample-data
  - Wyscout public dataset (Pappalardo et al., *Scientific Data* 2019): https://www.nature.com/articles/s41597-019-0247-7
  - kloppy: https://github.com/PySport/kloppy
  - socceraction: https://github.com/ML-KULeuven/socceraction
  - vaastav FPL history: https://github.com/vaastav/Fantasy-Premier-League
  - football-data.co.uk: https://www.football-data.co.uk/
- **SoccerNet 2025** challenge results paper: [arXiv 2508.19182](https://arxiv.org/abs/2508.19182)
- **SoccerNet 2026** results paper (arXiv 2607.07320, July 2026):
  - Five vision tasks: Ball Action Anticipation (a 5s horizon from 30s of footage, 10 classes); Player-Centric Ball Action Spotting (8 classes plus team and jersey); Novel View Synthesis; Spiideo SoccerNet SynLoc (pitch localisation from a single calibrated static camera); and broadcast Visual Question Answering.
  - Participation: 427 teams and 1,129 submissions; leaderboards include only the 28 teams with reviewed reports.
  - Sources: [arXiv 2607.07320](https://arxiv.org/abs/2607.07320); follow-up example [arXiv 2606.09679](https://arxiv.org/pdf/2606.09679)
- **MLSA:**
  - MLSA 2025 (12th, Porto): CCIS 2833, 11 of 24 papers accepted. — [Springer](https://link.springer.com/book/9783032151643)
  - MLSA 2026 (13th) at ECML PKDD 2026 had a 5 June 2026 submission deadline. — [CfP](https://groups.google.com/g/ml-news/c/YOVJDCgGYTk)
  - Publication series index: [researchr](https://researchr.org/conferenceseries/mlsa/publications)
- ***Machine Learning* journal:** ran a "Machine Learning in Soccer" special issue tied to the 2023 Soccer Prediction Challenge. — [CfP](https://link.springer.com/journal/10994/updates/23772252)
- **Dagstuhl Seminar 24081** (2024) on sports analytics produced a report listing open challenges. — [Dagstuhl report PDF](https://drops.dagstuhl.de/storage/04dagstuhl-reports/volume14/issue02/24081/DagRep.14.2.164/DagRep.14.2.164.pdf)
- **OpenSTARLab:** a sports-analytics preprocessing toolkit with football data-provider loaders, including space evaluation. — [OpenSTARLab docs](https://openstarlab.readthedocs.io/en/latest/Pre_Processing/Sports/Space_Eval/Data_Provider/Football/index.html)

### Inferences
- SoccerNet is vision-only (video/CV). It is a poor fit for a forecasting system and needs GPUs, though some tasks (VQA, anticipation) are single-GPU-feasible with baselines.
- No major public *forecasting* benchmark exists for club football with point-in-time features. This is the gap StatsNuke could fill.

### Gaps
- MIT Sloan SSAC 2025/2026 research-paper winners and their stated open problems were not retrieved.
- StatsBomb/Hudl conference 2025/2026 agendas were not retrieved.
- JQAS and *Journal of Sports Analytics* 2026 editorials were not retrieved.
- The contents of the Dagstuhl 24081 report were not read in detail; only its existence was confirmed.

---

## 10. Competitions, leaderboards, and prize problems a solo researcher can enter

### Takeaway
The football *forecasting* competitions are episodic: the 2017 and 2023 Soccer Prediction Challenges by Berrar et al. Prospective LLM arenas (2026) and Kaggle's March Machine Learning Mania (basketball) are the live venues. SoccerNet runs annually but is computer vision. The FPL global leaderboard itself (10M+ managers) is the most natural prospective benchmark for an FPL system.

### Cited Findings
- **2023 Soccer Prediction Challenge.**
  - Format: predict 736 future matches. Task 1 was exact scores; Task 2 was win/draw/loss probabilities.
  - Berrar, Lopes & Dubitzky's (organisers, out of competition) k-NN and neural models achieved top performance. They concluded that simple learners with domain knowledge win.
  - Published in *Machine Learning* 113:8165–8204 (2024), DOI 10.1007/s10994-024-06625-9.
  - Sources: [Springer](https://link.springer.com/article/10.1007/s10994-024-06625-9); [PDF](https://dberrar.github.io/papers/Berrar_etal_Soccer2024.pdf); [challenge site](https://sites.google.com/view/2023soccerpredictionchallenge/challenge-evaluation)
- **A 2023 challenge entry by Fujii et al.** is on arXiv 2309.14807. — [arXiv 2309.14807](https://arxiv.org/html/2309.14807v1)
- **Kaggle March Machine Learning Mania 2026:** NCAA basketball, Brier-scored, 3,485+ entries per one participant's blog. — [participant blog](https://thomaslarock.com/2026/04/fail-better-then-finish-21st/); [CMU MMML](https://www.cs.cmu.edu/~reids/mmml/)
- **SoccerNet 2026** (427 teams) is vision-focused. — [arXiv 2607.07320](https://arxiv.org/abs/2607.07320)
- **Prospective LLM football arenas** from 2026 (LLM-SoccerArena with a live leaderboard, WorldCupArena, WC2026-Agents) compare against market baselines. Statistical models could be entered as reference competitors. — [2607.24573](https://arxiv.org/pdf/2607.24573); [2607.17765](https://arxiv.org/abs/2607.17765)
- **[prior knowledge, not re-verified]** The 2017 Soccer Prediction Challenge (Berrar, Lopes, Davis, Dubitzky) was published in *Machine Learning* 108 (2019) and used the "Open International Soccer Database".

### Inferences
- A successor Soccer Prediction Challenge has not been announced (none found). It may recur around ECML or *Machine Learning* special issues.
- A strong, low-risk entry would be a *prospective, timestamped* public forecast log: EPL match probabilities and FPL player distributions posted before each deadline (e.g., a GitHub release or hash commitments). This produces community-recognised evidence and a reusable benchmark.

### Gaps
- Did not confirm whether a 2025 or 2026 Soccer Prediction Challenge exists.
- Did not check the status of the NFL Big Data Bowl 2026.
- Did not check for any FPL-specific prediction competitions.

---

## 11. Where publishing StatsNuke would fill a genuine gap (synthesis, ranked by gap and feasibility)

### Takeaway
Ranked: (1) a point-in-time FPL benchmark plus a minutes/start benchmark; (2) a "market-anchored generative model encompasses or ties the closing line" forecasting paper with block-bootstrap/DM methodology; (3) crowd transfer-flow as team-news information; (4) decision-quality evaluation of FPL forecasters. Each rests on the gaps documented above. All are CPU-scale and use public data.

### Cited Findings
- Absence of a minutes benchmark and reliance on availability tags: OpenFPL dispenses with expected minutes. — [arXiv 2508.09992](https://arxiv.org/html/2508.09992v1)
- FPL academic work evaluates single seasons with deterministic or robust MILP. — [arXiv 2505.02170v3](https://arxiv.org/abs/2505.02170v3)
- Market-as-benchmark framing is current: the IJF 2025 paper and the 2026 LLM arenas all treat the market as the bar. — [EconPapers](https://econpapers.repec.org/RePEc:eee:intfor:v:41:y:2025:i:2:p:803-820); [2607.17765](https://arxiv.org/abs/2607.17765)
- The methodology call (calibration, dependency-aware evaluation, uncertainty) comes from Davis et al. 2024. — [Springer](https://link.springer.com/article/10.1007/s10994-024-06585-0)

### Inferences
Each proposed paper below gives the problem, contribution, venue, compute and measures of success.

1. **"FPL-PIT: a point-in-time benchmark for fantasy football forecasting and decision-making."**
   - Contents: data (snapshots), tasks (points distribution, minutes, season management), baselines (OpenFPL, simple form, StatsNuke) and metrics (CRPS, Brier, regret, rank percentile).
   - Venue: MLSA 2027, *JQAS*, or a NeurIPS Datasets & Benchmarks track.
   - Risk: FPL API terms and redistribution licensing; check them first.
2. **"Who starts? Forecasting football minutes from crowd and availability signals."**
   - Contents: lead-time-indexed start and minutes forecasts, with information-arrival curves (how predictability rises as the deadline approaches).
   - This is the largest error source, novel, and cheap to produce.
3. **"Does a transparent generative model encompass the closing line? Evidence from N EPL seasons."**
   - Contents: RPS and log loss with gameweek-block bootstrap and DM tests; encompassing regressions; per-segment residual analysis (draws, promoted teams, early season); de-vig method comparison; Asian-handicap fusion.
   - Venue: *IJF* or *JQAS*.
   - A "tie" result is publishable if the methodology is rigorous and the data and code are open.
4. **"Decision-focused evaluation of fantasy forecasters."**
   - Contents: show that MSE improvements do or do not translate into points and rank, using the vectorised simulator plus the MILP.
   - Optional extension: decision-focused learning (SPO+).
5. Secondary projects:
   - xG-source sensitivity of match forecasts (Understat vs. StatsBomb).
   - Possession value (VAEP/xT) as a predictive feature against the market.
   - Calibrated LEM-style match simulation tested against odds.
   - Rest effects via TV rescheduling.
   - A prospective public forecast log entered into LLM arenas as a statistical baseline.

General lessons from other sports:
- Kaggle March Mania's long-run evidence is that Brier-scored, well-calibrated simple models plus market or rating info win. This mirrors Berrar et al.'s conclusion that domain knowledge beats algorithm complexity. [participant blog](https://thomaslarock.com/2026/04/fail-better-then-finish-21st/); [Berrar 2024](https://link.springer.com/article/10.1007/s10994-024-06625-9)
- The NFL spread market appears economically efficient, with statistical efficiency unclear. This parallels football. [Hanken thesis via Helda](https://helda.helsinki.fi/server/api/core/bitstreams/db1e1612-b57e-4d2b-84d2-4c2c68962e29/content)

### Gaps
- Licensing terms for redistributing FPL API snapshots and Understat-derived data were not checked. This is critical for a benchmark release.
- No inventory of existing FPL "expected minutes" community datasets (e.g., FPL Review's) was found. FPL Review is commercial and its methods are unpublished.
- arxiv.org full texts were unreachable in this session. Exact metrics from OpenFPL (e.g., MAE values and the comparison set) should be verified from the PDF.
