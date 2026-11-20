# Open Research Problems: Tabular FMs, Time-Series FMs, Probabilistic Forecasting/Calibration, Conformal Prediction (state as of Nov 2026)

> **Method note (read first).** This session could run web searches but could **not** fetch arxiv.org (the host is blocked by the network settings). Most 2026 claims below therefore come from **search-result snippets and abstracts, not full texts**. Numbers (Elo, leakage %, ROI) are snapshot values and differ between leaderboard views. Items marked **[prior knowledge]** are well-known papers cited from memory with their canonical arXiv IDs. They were not re-verified in this session. Several 2026 arXiv IDs (26xx.xxxxx) were seen only as search hits. Treat them as "exists per search index" and check them before formal citation.

## Q1. Where do tabular foundation models still fail vs. GBDTs, and what is open (scale, shift, temporal/panel, grouped, calibration, fine-tuning vs ICL)?

### Takeaway
By late 2026 TFMs (the TabPFN-2.5/2.6/3/3.5 line, TabICLv2, RealTabPFN-2.5, LimiX, Mitra-v2) top the IID TabArena leaderboard by large Elo margins. The main remaining failures are **non-IID regimes**: temporal splits, grouped/panel splits, very large n, and high dimensionality. On these, tuned GBDTs, RealMLP and TabM still win. The non-IID regime is exactly where a walk-forward sports panel sits. Open problems with a cheap entry point are calibration under shift, fine-tuning vs in-context adaptation, and context/row selection for panel data.

### Cited Findings
**State of the art and leaderboards**
- On TabArena (51-dataset leaderboard, v0.1.4 snapshot) the top four single models are all TFMs: TabPFN-3, TabPFN-2.6, RealTabPFN-2.5 and TabICLv2. TabPFN-3 default scores about 1673 Elo versus 1375 for tuned and post-hoc-ensembled XGBoost. The TabPFN-3 report lists 1677 and 1631 in different tables. — [CodeSOTA TabArena summary](https://www.codesota.com/tasks/tabular-ml); [TabPFN-3 Technical Report, arXiv 2605.13986](https://arxiv.org/pdf/2605.13986)
- TabICLv2 (arXiv 2602.11139) is described as "better, faster, scalable, and open". It sits around 1596–1613 Elo with no tuning. Its paper notes that RealTabPFN-2.5's numbers rely on tuning and ensembling. — [TabICLv2](https://arxiv.org/pdf/2602.11139)
- The TabPFN-3.5 technical report (arXiv 2609.17895) reportedly puts a "TabPFN-3.5-Thinking" variant first, at about 1910 Elo. This was seen in a snippet only. — [TabPFN-3.5 report](https://arxiv.org/pdf/2609.17895)
- LimiX default sits only around 14th on one TabArena view. LimiX-2M (arXiv 2606.04485) targets "low-rank collapse and attention bottlenecks" in TFMs. A separate entry lists LimiX-2 at 1,929–2,269 Elo, which could not be reconciled with the other views. — [LimiX-2M](https://arxiv.org/pdf/2606.04485); [CodeSOTA](https://www.codesota.com/tasks/tabular-ml)
- Other 2026 TFM entries: Mitra-v2 Technical Report (arXiv 2609.04540), TAFFY task-adaptive TFM (arXiv 2610.07559), GIST generator-state-informed supervision (arXiv 2610.12904), TabAbax (arXiv 2611.00562). TabAbax is about 10th of 41 on BeyondArena, +68 Elo over TabICLv2. — [Mitra-v2](https://arxiv.org/pdf/2609.04540); [TAFFY](https://arxiv.org/pdf/2610.07559); [GIST](https://arxiv.org/pdf/2610.12904); [TabAbax](https://arxiv.org/html/2611.00562v1)
- TabArena itself: Erickson et al. 2025, a living benchmark with curated datasets, tuning and ensembling protocols. arXiv 2506.16791. **[prior knowledge]** — [TabArena](https://arxiv.org/abs/2506.16791)

**Failure regimes (the core open problem)**
- "Beyond IID: How General Are Tabular Foundation Models, Really?" (Purucker et al., arXiv 2606.30410, June 2026) evaluates 11 models on 142 curated datasets and introduces **BeyondArena**. Open-source TFMs are competitive or better up to about 100k rows. **Tree-based and deep-learning models still dominate on grouped or large data.** — [arXiv 2606.30410](https://arxiv.org/abs/2606.30410)
- Industry summary: on temporal and grouped splits the 2026 leaders trail tuned RealMLP, TabM and the GBDTs. On very large tables TFMs fall below default XGBoost. — [Qlik blog "Five Things that 2026 Changed"](https://www.qlik.com/blog/results-for-tabular-machine-learning); [CodeSOTA](https://www.codesota.com/tasks/tabular-ml)
- An MDPI *MAKE* critical review (13 Aug 2026) lists open problems. The three visible in the snippet: (i) generalisation under non-IID, large or high-dimensional data; (ii) a generation fidelity gap (multi-table structure, numeric fidelity, privacy); (iii) **benchmark contamination from real-data pretraining overlapping evaluation sets**. — [Critical Review, doi 10.3390/make8080244](https://doi.org/10.3390/make8080244)
- Position paper: TFMs cut off from operational context (business rules, domain knowledge) have inherent limits. The paper proposes "Semantically Linked Tables". — [Klein & Hoffart, arXiv 2505.19825](https://arxiv.org/abs/2505.19825)
- A theoretical paper claims a "formal barrier" ("Statistically Indistinguishable, Operationally Distinct"). Its content was not visible. — [arXiv 2606.29091](https://arxiv.org/pdf/2606.29091)
- TFMs are being applied beyond tables: "Tabular foundation models for non-tabular tasks" (arXiv 2608.22594). — [link](https://arxiv.org/pdf/2608.22594)

**Temporal shift, calibration, fine-tuning**
- Drift-Resilient TabPFN (NeurIPS 2024) learns temporal distribution shift in context from a prior with shifting structural causal models. It beats baselines out-of-distribution with strong calibration and needs no tuning. *(Older; partially superseded by newer TFMs, but no successor that targets drift is clearly established.)* — [NeurIPS 2024 paper](https://proceedings.neurips.cc/paper_files/paper/2024/file/b2e2774c8e76afe191b5bf518f5cb727-Paper-Conference.pdf); [code](https://github.com/automl/Drift-Resilient_TabPFN)
- "In-context learning of evolving data streams with tabular foundational models" (arXiv 2502.16840) covers TFMs for streams. — [link](https://arxiv.org/pdf/2502.16840)
- Fine-tuning most TFMs on a downstream table reportedly **lowers accuracy and raises ECE together**. TabPFNv2 is the exception: full fine-tuning works best, converges faster than LoRA and keeps calibration. The proposed explanation is that TabPFN acts as an implicit retriever whose similarity structure gradient updates can destroy. *Blog-level source; verify against arXiv 2608.10280.* — [Qlik blog](https://www.qlik.com/blog/results-for-tabular-machine-learning)
- Putting TabPFN in place of the linear head on fine-tuned embeddings substantially reduces calibration error with no gradient updates. The same paper cites earlier work in which TabPFN keeps better ECE than XGBoost under temporal shift. — [arXiv 2607.11007](https://arxiv.org/html/2607.11007v2)
- Label-shift correction for ICL (arXiv 2605.04363). A reported "CALD" method *detects* harmful shift for TabPFN/TabICL but *does not correct* it. This is GitHub-issue-level sourcing and unverified. — [lazy-tfm issue #302](https://github.com/biprateep/lazy-tfm/issues/302)
- TabPFN v2 (Hollmann et al., *Nature* 2025) was originally limited to about 10k samples and 500 features. Later versions (2.5/3) raised these limits a lot. Exact current limits were not verified in this session. **[prior knowledge]** — [Nature 2025](https://www.nature.com/articles/s41586-024-08328-6)

### Inferences
**Concrete open problems for a solo researcher (Q1)**
1. **TFMs on walk-forward panel data (temporal + grouped split together).** *Why open:* BeyondArena and Qlik both name temporal and grouped splits as where TFMs lose. Almost no public benchmark combines both with point-in-time features. *Solo work:* build a walk-forward benchmark from sports panel data (team-match and player-gameweek rows). Compare TabPFN-2.5/3 (or the open TabICLv2) zero-shot, TabPFN fine-tuned, CatBoost/LightGBM, and TabM/RealMLP. *Compute:* one GPU; most seasons fit in a 10k–50k-row context. *Success:* log-loss/RPS/CRPS per season, Elo-style aggregate, and a gap closed against the betting-market baseline.
2. **Context (row) selection and recency weighting for ICL under drift.** Options: which past rows to put in context (most recent, same team or player, kNN retrieval), and whether time-aware positional features help. *Why open:* TFMs act as implicit retrievers (per the Qlik summary), but there is no principled retrieval or recency policy for non-stationary panels. *Success:* improvement over the "all history" context at the same compute; ablations by season distance.
3. **Calibration of TFM outputs under temporal shift, and cheap post-hoc fixes.** Compare TFM ECE/reliability against temperature scaling, Venn-Abers and conformal on a walk-forward split. Test whether fine-tuning degrades calibration on real temporal data, which is the Qlik claim. *Compute:* CPU/GPU, hours.
4. **Fine-tuning vs ICL vs "TFM-as-head" for small grouped data.** Measure when a 20–200-row per-entity context beats global fine-tuning. Player-level FPL data gives thousands of small groups.
5. **Contamination audit for TFMs pretrained on real data** (RealTabPFN, Mitra, LimiX). Check whether OpenML/TabArena tables overlap with real-data pretraining (the review's problem iii). *Solo work:* membership-style probes, or comparing performance on post-cutoff tables (for example, a 2026/27 season not yet in existence at pretraining time). *Success:* a measured performance gap between pre- and post-cutoff data.
6. **Ordinal/structured targets with TFMs.** Examples: football home/draw/away ordered outcomes, scorelines as bivariate counts, FPL points as zero-inflated counts. TFMs regress or classify without structure. Test whether output-head constraints (cumulative-link, Poisson/NB parameterisation) improve proper scores.

### Gaps
- Could not read full texts of BeyondArena (2606.30410), TabPFN-3/3.5 or the MAKE review, so the exact BeyondArena margins and the review's full list of seven open problems are unknown.
- Current TabPFN-3/3.5 row and feature limits and licensing terms were not verified.
- No source found that benchmarks TFMs directly on sports or betting data.

## Q2. Time-series foundation models: open problems (covariates, multivariate/hierarchical, calibration, contamination, intermittent/count, events)

### Takeaway
Chronos-2 (Oct 2025) made covariate-aware, multivariate zero-shot forecasting mainstream. TiRex-2, Toto-2.0, TimesFM-3, Moirai-2 and t0 now compete. But **covariates often do not help**, static covariates are not supported, and **benchmark leakage is endemic**. Examples: Moirai-2.0 about 28% and TimesFM-2.5 about 10% overlap on fev-bench, and TimesFM 1.0/2.0 are flagged on GIFT-Eval. Intermittent/sparse series are poorly evaluated. These are good targets for a solo researcher: evaluation, auditing and small adapters rather than pretraining.

### Cited Findings
**Models and capabilities**
- Chronos-2 ("From Univariate to Universal Forecasting", arXiv 2510.15821) supports past and future covariates and multivariate inputs via group attention. Its multivariate skill depends on synthetic multivariate pretraining data: an ablation trained on univariate data only drops sharply on fev-bench. **Static covariates are not supported.** It beats TabPFN-TS and TiRex "by a wide margin" on the fev-bench covariate subset. — [Chronos-2 arXiv 2510.15821](https://arxiv.org/pdf/2510.15821); [ECB conference copy](https://www.ecb.europa.eu/press/conferences/shared/pdf/20260323_forecasting/Guerron_paper.pdf)
- A further Chronos-2 paper ("A foundation model for multivariate time series forecasting", arXiv 2610.41956) appeared. Only the title was seen. — [link](https://arxiv.org/html/2610.41956)
- TiRex-2 generalises TiRex to multivariate data and streaming. It argues Chronos-2 is "neither target-causal nor constant-memory". It reports the lowest mean MASE with about 2.2x fewer mean active parameters than Chronos-2. — [TiRex-2 (ResearchGate)](https://www.researchgate.net/publication/408340932_TiRex-2_Generalizing_TiRex_to_Multivariate_Data_and_Streaming)
- In fev-bench, TiRex is univariate with direct quantile outputs. Toto-1.0 has cross-variate attention but no known or categorical covariates. TabPFN-TS supports known covariates. Most other models ignore covariates. — [fev-bench arXiv 2509.26468](https://arxiv.org/html/2509.26468v2)
- t0 (The Forecasting Company, arXiv 2609.24559) targets "forecasting with context". It reports that **11 of 30 known-future-covariate tasks score worse with covariates**, by up to 15% worse (15-min electricity with weather covariates). Toto-2.0 variants sit close to Chronos-2 on the covariate subset. — [t0 arXiv 2609.24559](https://arxiv.org/pdf/2609.24559)
- TS-ICL (arXiv 2606.05878): a time-indexed TSFM via in-context learning. — [link](https://arxiv.org/pdf/2606.05878)

**Benchmark contamination**
- GIFT-Eval (arXiv 2410.10393) concedes *partial leakage*. Retraining Moirai on a clean split showed leakage inflates scores, especially at long horizons. The benchmark ships a non-leaking pretraining corpus of about 230B points. — [GIFT-Eval](https://arxiv.org/pdf/2410.10393)
- fev-bench distinguishes training on a benchmark's train split from exposure to its test split. It relies on **self-reported** pretraining overlap and discounts overlapping tasks. — [fev-bench](https://arxiv.org/html/2509.26468)
- A later paper cites **fev-bench leakage of 28% for Moirai-2.0 and 10% for TimesFM-2.5** and excludes them from its main baselines. — via search snippet referencing [t0 arXiv 2609.24559](https://arxiv.org/pdf/2609.24559)
- t0 also claims the GIFT-Eval pretraining corpus contains a demand series twice, and that 2014 overlap leaks for most TSFMs. Vendor-authored; unverified. — [t0](https://arxiv.org/pdf/2609.24559v1); [TDS explainer](https://towardsdatascience.com/time-series-llms-explained-with-t0-alpha/)
- **TSFMAudit** (arXiv 2605.26161): LLM contamination-detection methods do not transfer to time series. Low loss does not prove contamination because smooth or seasonal series are easy anyway. — [TSFMAudit](https://arxiv.org/pdf/2605.26161)
- A "source-first taxonomy and audit framework" for evidence behind zero-shot TS forecasting (arXiv 2609.21425). — [link](https://arxiv.org/pdf/2609.21425)
- TSFM-Bench: a unified benchmark of TSFMs. — [ResearchGate](https://www.researchgate.net/publication/394261019_TSFM-Bench_A_Comprehensive_and_Unified_Benchmark_of_Foundation_Models_for_Time_Series_Forecasting)

**Intermittent/sparse/count data**
- "When, Not How Much: Evaluating TSFMs on Sparse Events" (arXiv 2609.39386) argues sparse-series decisions hinge on *which* periods are active. MAE-type scores are minimised by the median, so they reward all-zero forecasts. The paper proposes occurrence-focused evaluation grounded in hurdle and zero-inflated models. — [arXiv 2609.39386](https://arxiv.org/html/2609.39386v1)
- TimesFM-3 vs classical methods on simulated intermittent data: TimesFM-3 is roughly on par with, but slightly behind, iETS, negative binomial and a simple history benchmark on scaled pinball loss and RMSSE. — [arXiv 2610.00589](https://arxiv.org/pdf/2610.00589)
- More accurate intermittent-demand forecasters can give lower order-level service (Chin et al. 2026, as cited in 2609.39386). — [arXiv 2609.39386](https://arxiv.org/pdf/2609.39386)
- Robustness to sensor dropout: intermittent dropout barely moves average accuracy, but individual channels degrade (notably TiRex). — [MDPI Future Internet doi 10.3390/fi18100550](https://doi.org/10.3390/fi18100550)
- Bayesian zero-inflated count forecasting (IJF 2025) is a classical strong baseline. — [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0169207025001220)
- Review placing TSFMs against VARs: "From Vector Autoregressions to AI-based Time Series Forecasting" (arXiv 2607.14279). — [link](https://arxiv.org/pdf/2607.14279)

### Inferences
**Concrete open problems (Q2)**
1. **When do covariates help a TSFM? A diagnostic or gating method.** *Why open:* t0 shows covariates hurt on 11/30 tasks, and there is no principled test for covariate usefulness at inference time. *Solo work:* a cheap held-out-window gate (use covariates only if they improve backtest CRPS), or covariate dropout ensembles. Evaluate on the fev-bench covariate subset. *Compute:* inference only, one GPU. *Success:* the share of tasks where gated ≥ max(with, without), and the aggregate skill score versus Chronos-2 with covariates always on.
2. **Static covariates and entity metadata for TSFMs.** Chronos-2 lacks them explicitly. Test injection via pseudo-covariates (constant channels), TFM residual stacking (TabPFN on entity features + TSFM residuals), or lightweight adapters. Sports panels are a natural testbed: team strength, player position and price are static or slowly varying.
3. **Contamination-robust evaluation via post-cutoff data.** *Why open:* fev-bench relies on self-reporting, and TSFMAudit shows loss-based detection fails. *Solo work:* build a "fresh" benchmark slice from series whose observations postdate all model cutoffs (sports, weather, prices after mid-2026), or apply a TSFMAudit-style difficulty-controlled comparison (pre vs post cutoff, matched by forecastability such as spectral entropy). *Success:* the estimated leaderboard rank change after controlling for leakage.
4. **Intermittent/count-aware evaluation and output heads.** Apply occurrence-focused metrics (Brier on P(y>0), plus CRPS conditional on activity) to TSFMs. Test whether a hurdle post-processor on top of TSFM quantiles beats iETS/NB. FPL player points (many zeros, spiky hauls) and goal counts are ideal. *Compute:* CPU-feasible.
5. **Probabilistic calibration of TSFM quantiles across horizons.** Most leaderboards report WQL/CRPS aggregates. Measure PIT uniformity and quantile coverage by horizon and domain, then fix with lightweight conformal or isotonic recalibration (see Q4). *Success:* coverage error reduced at matched CRPS.
6. **Event-driven/irregular series.** Sports matches are irregular events with known future schedule covariates (fixtures, opponent, home/away). Map these to TSFM inputs (regular-grid "gameweek" time vs event time), compare against tabular models, and publish the negative or positive result.
7. **Hierarchical coherence.** Examples: player → team → league totals, or minutes × points-per-minute. Test whether TSFM forecasts + MinT/probabilistic reconciliation beat joint multivariate Chronos-2/TiRex-2. *Note:* this session found no 2026 source on hierarchical TSFMs specifically.

### Gaps
- No primary 2026 source found on hierarchical/grouped forecasting with TSFMs, Sundial, Moirai-MoE or Lag-Llama status (Lag-Llama is likely superseded; flag as older).
- The current fev-bench and GIFT-Eval leaderboard top-5 as of Nov 2026 could not be fetched.
- TimesFM-3 architecture and covariate support details were not verified.

## Q3. Probabilistic forecasting and calibration: multiclass/ordinal/joint calibration, scoring rules for structured outputs, recalibration under shift, LLM probability forecasts

### Takeaway
Multiclass calibration is moving from "top-label ECE" to local, utility-aware and multicalibration notions, with several 2025–26 papers but no consensus metric. Multivariate scoring rules (energy score, variogram, CRPS-sum) have known weak discrimination for dependence structure, and finite-sample behaviour is under-characterised. LLM forecasting is now near superforecaster parity on ForecastBench. That shifts the open problems to **leakage-free evaluation**, **calibration aggregation** and **beating markets**. A 2026 World Cup benchmark found frontier agents could not beat the bookmaker.

### Cited Findings
**Multiclass/structured calibration**
- Multiclass local calibration with Jensen-Shannon distance (AISTATS 2026, arXiv 2510.26566). It flags extending local calibration beyond neural networks as open. — [arXiv 2510.26566](https://arxiv.org/abs/2510.26566)
- "Truthful Calibration Errors for Multi-Class Prediction" (arXiv 2510.06388) addresses binning-dependence and truthfulness of calibration-error measures. — [link](https://arxiv.org/html/2510.06388)
- Scalable utility-aware multiclass calibration (arXiv 2510.25458); Divide et Calibra, region-wise calibration via vector quantization (arXiv 2605.21060); instance-adaptive online multiclass multicalibration (arXiv 2610.15954); hierarchical utility calibration for structured multiclass decisions (arXiv 2609.36532); linear-log-odds multiclass recalibration (arXiv 2602.18573). — [2510.25458](https://arxiv.org/pdf/2510.25458); [2605.21060](https://arxiv.org/abs/2605.21060); [2610.15954](https://arxiv.org/html/2610.15954); [2609.36532](https://arxiv.org/abs/2609.36532); [2602.18573](https://arxiv.org/pdf/2602.18573)
- This session found **no ordinal-specific calibration-error work**. That is a gap and probably an opportunity (see Inferences). — (search null result)

**Multivariate scoring rules**
- The energy score has limited ability to discriminate dependence structure, with a derived theoretical upper bound for Gaussian processes. — [DTU Orbit](https://orbit.dtu.dk/en/publications/discrimination-ability-of-the-energy-score/)
- Variogram scores (p = 0.5, 1) discriminate better than the energy score. — [Springer, Annals of OR](https://link.springer.com/article/10.1007/s10479-022-04611-9); [Scheuerer & Hamill 2015, MWR](https://journals.ametsoc.org/view/journals/mwre/143/4/mwr-d-14-00269.1.xml)
- CRPS-Sum can rank random noise above state-of-the-art methods. — [arXiv 2201.08671](https://arxiv.org/pdf/2201.08671)
- "Propriety is not sufficient to guarantee good discrimination in the non-asymptotic regime" (Marcotte et al., ICML 2023, "Regions of Reliability"). — [PMLR v202](https://proceedings.mlr.press/v202/marcotte23a/marcotte23a.pdf)
- Energy-score discrimination improves with smaller β. — [Siegen thesis](https://dspace.ub.uni-siegen.de/entities/publication/72cb9f78-0a0e-4384-8a7a-12ff90110cfa)
- MVG-CRPS as a robust multivariate loss (arXiv 2410.09133); survey of proper scoring rules (arXiv 2504.01781). — [2410.09133](https://arxiv.org/pdf/2410.09133); [2504.01781](https://arxiv.org/html/2504.01781v1)
- *Flag:* most of this literature is 2015–2023. No 2026 resolution was found, so the problem appears still open.

**LLM probability forecasts**
- ForecastBench (Karger et al., arXiv 2409.19839; ICLR 2025) is a dynamic benchmark of unresolved questions. — [arXiv 2409.19839](https://arxiv.org/pdf/2409.19839); [about](https://www.forecastbench.org/about/)
- FRI introduced the "Brier Index": superforecasters at 70.6% versus 67.9% for the best LLM. — [FRI Substack](https://forecastingresearch.substack.com/p/introducing-the-brier-index)
- Sources conflict on parity. ForecastBench/Lightningrod (July 2026) say top systems (Gemini, Cassi ensemble, Foresight-v3) sit within the superforecaster CI (superforecaster median 69.3 vs Foresight-v3 67.7; vendor source). Parallect (May 2026) reports superforecasters at about 70.2 vs the best LLM at about 67.9. — [Lightningrod blog](https://blog.lightningrod.ai/p/ai-reaches-human-superforecaster-range); [Parallect](https://parallect.ai/reports/ai-superforecasters-parity-may-2026-fb32ac); [LEAP question](https://leap.forecastingresearch.org/forecasts/forecastbench-performance)
- The AIA Forecaster report calls ForecastBench "relatively easy". — [arXiv 2511.07678](https://arxiv.org/pdf/2511.07678)
- **Temporal leakage in LLM backtesting** (arXiv 2608.02985) estimates a post-cutoff "jump" using ForecastBench's resolved-question archive (Polymarket, Metaculus, Manifold, INFER, 2024Q3–2026Q3). — [arXiv 2608.02985](https://arxiv.org/pdf/2608.02985)
- Date-filtered web search still returns post-dated articles. — [arXiv 2602.00758](https://arxiv.org/pdf/2602.00758)
- The "Lookahead Propensity" statistic for detecting lookahead bias. — [arXiv 2512.23847](https://arxiv.org/pdf/2512.23847)
- "Forecasting ability of LLMs depends on what we're asking". — [arXiv 2511.18394](https://arxiv.org/pdf/2511.18394)
- Metaculus FutureEval and bot tournaments; frozen-corpus benchmarks (Bench to the Future 2, FutureSim). — [arXiv 2604.18576](https://arxiv.org/pdf/2604.18576); [Metaculus "AI Forecasting in 2026"](https://www.metaculus.com/notebooks/43363/ai-forecasting-in-2026/)
- Survey of LLM forecasting agents (arXiv 2608.23058). — [link](https://arxiv.org/pdf/2608.23058)
- **FIFA World Cup 2026 as a contamination-free benchmark** (arXiv 2607.17765): four LLM agents vs a bookmaker over 104 matches. Frontier agents predict alike and **cannot beat the market**. — [arXiv 2607.17765](https://arxiv.org/pdf/2607.17765)

### Inferences
**Concrete open problems (Q3)**
1. **Ordinal calibration metrics and recalibration for ordered outcomes.** Home/draw/away is ordinal. RPS is the standard proper score, but there is no well-established ordinal calibration-error estimator or reliability diagram, and this search found none. *Solo work:* define cumulative-threshold calibration (calibrate P(Y ≤ k) for each k), study estimator bias and variance, and compare to top-label and classwise ECE on football data. *Compute:* CPU. *Success:* the estimator's power to detect known miscalibration in simulation, plus a real-data recalibration gain in RPS.
2. **Discrimination power of multivariate scores for joint match outcomes** (scoreline = bivariate counts; correlated player returns). Simulate known dependence misspecification (for example, Dixon-Coles ρ wrong) and measure which score (energy, variogram, log, MMD-type) detects it at realistic n (about 380 matches per season). This extends Marcotte et al. to discrete-count settings. *Compute:* CPU.
3. **Recalibration under drift with a market anchor.** Sequential or online recalibration (rolling Platt, beta calibration, online multicalibration per 2610.15954) of model probabilities, using de-vigged market probabilities as a covariate. Measure the benefit across season boundaries.
4. **LLM forecasting vs markets in sports, with leakage controls.** Replicate and extend 2607.17765 on EPL with point-in-time retrieval, lookahead-propensity checks (2512.23847) and model-plus-LLM ensembles. *Compute:* API cost rather than GPU.
5. **Aggregation and extremizing of heterogeneous probabilistic forecasters** (TFM, GBDT, TSFM, LLM, market). The question is when the "wisdom of models" beats the market, using proper-score stacking in walk-forward.

### Gaps
- The current ForecastBench leaderboard could not be accessed directly. Parity claims conflict and some are vendor-sourced.
- No 2026 primary work found on calibration of *joint* (multivariate) forecasts beyond scoring-rule discrimination.
- The Metaculus AI benchmark (Q3/Q4 2026 bot-tournament) results were not retrieved.

## Q4. Conformal prediction: time series/non-exchangeable, conditional coverage, decision-making (CRC, conformal decision theory), multivariate/structured, online under shift

### Takeaway
The core theory is mature: ACI, conformal PID, weighted and non-exchangeable CP, and CRC. Open problems are now (a) **tightening coverage-gap bounds adaptively** under non-exchangeability, (b) conditional/subgroup coverage (impossible distribution-free in finite samples, so approximate notions are needed), (c) CRC for non-monotone, unbounded or selection-affected losses, and (d) feasibility limits under shift. Panel data and multi-step forecasts are active 2026 fronts, and both match the sports setting well.

### Cited Findings
- The ACM Computing Surveys "Conformal Prediction: A Data Perspective" (Jan 2026) says non-exchangeable CP bounds the weighted-CP coverage gap by a distribution distance, and **adaptively minimising that gap remains open**. Extending CP to general data imperfections with tractable gaps is also open. — [ACM CSUR](https://dl.acm.org/doi/full/10.1145/3736575)
- "A Gentle Introduction to Conformal Time Series Forecasting" (arXiv 2511.13608) derives split-conformal guarantees under weak dependence. It also gives a simulation comparison (coverage, width, compute) of reweighting, residual-updating and online-α methods, and lists open directions. — [arXiv 2511.13608](https://arxiv.org/pdf/2511.13608)
- Multi-step conformal methods seldom give guarantees without exchangeability. Distribution-free conditional coverage is impossible in finite samples, and SPCI is only asymptotically conditional. — [Online conformal for multi-step TS, arXiv 2410.13115](https://arxiv.org/pdf/2410.13115); [EnbPI arXiv 2010.09107](https://arxiv.org/pdf/2010.09107); [SPCI arXiv 2206.07851](https://arxiv.org/pdf/2206.07851)
- JANET: joint adaptive prediction regions for multi-step/multivariate TS (arXiv 2407.06390). — [link](https://arxiv.org/pdf/2407.06390)
- **Online CP for non-exchangeable panel data** (arXiv 2605.17705, 2026). — [link](https://arxiv.org/pdf/2605.17705)
- Graph-structured multivariate TS conformal with "spectral graph conditional exchangeability" (arXiv 2605.04957). — [link](https://arxiv.org/abs/2605.04957)
- CRC with non-monotone losses: risk control is no longer intrinsic but depends on how n grows relative to grid resolution m. Listed future work: continuous parameters, heavy-tailed/unbounded losses, sharper structural bounds. — [arXiv 2604.01502](https://arxiv.org/html/2604.01502v2)
- When CRC can certify LLM outputs: feasibility/impossibility bounds. ACI cuts cross-dataset risk violations from 71% to 21% but cannot overcome the feasibility limit. — [arXiv 2606.29054](https://arxiv.org/html/2606.29054v2)
- Selection breaks calibration/test exchangeability, which motivates Selective CRC (arXiv 2512.12844). Non-exchangeable CRC (arXiv 2310.01262). Risk-controlled post-processing of decision policies (arXiv 2605.06479). Conformal Policy Control: indicator-type safety losses cannot be parameterised monotonically (arXiv 2603.02196). — [2512.12844](https://arxiv.org/pdf/2512.12844); [2310.01262](https://arxiv.org/pdf/2310.01262); [2605.06479](https://arxiv.org/pdf/2605.06479); [2603.02196](https://arxiv.org/html/2603.02196v1)
- Foundational methods **[prior knowledge]**: ACI (Gibbs & Candès 2021, arXiv 2106.00170); Conformal PID control (Angelopoulos, Candès, Tibshirani 2023, arXiv 2307.16895); CRC (Angelopoulos et al. 2022, arXiv 2208.02814); Conformal Decision Theory (Lekeufack et al. 2023, arXiv 2310.05921); CP beyond exchangeability (Barber et al. 2022, arXiv 2202.13415); weighted CP under covariate shift (Tibshirani et al. 2019, arXiv 1904.06019). — [ACI](https://arxiv.org/abs/2106.00170); [Conformal PID](https://arxiv.org/abs/2307.16895); [CRC](https://arxiv.org/abs/2208.02814); [CDT](https://arxiv.org/abs/2310.05921); [Barber et al.](https://arxiv.org/abs/2202.13415); [Weighted CP](https://arxiv.org/abs/1904.06019)
- Curated paper list (useful for a literature sweep). — [awesome-conformal-prediction](https://github.com/valeman/awesome-conformal-prediction)

### Inferences
**Concrete open problems (Q4)**
1. **Online conformal for panel data with entity heterogeneity.** Examples: hundreds of players, each a short non-stationary series. Compare pooled ACI/PID, per-entity, and hierarchical/shrinkage α-updates (building on 2605.17705). *Compute:* CPU. *Success:* long-run and per-entity coverage error, interval width, and worst-decile-entity coverage.
2. **Conformalising TSFM/TFM outputs.** Calibrate quantiles from Chronos-2/TiRex or TabPFN with conformalized quantile regression plus online updates across horizons. Joint multi-step regions (JANET-style) for gameweek horizons 1–6. Success: coverage at all horizons with less width than the raw quantiles or classical baselines.
3. **Conformal decision theory / risk control for betting and FPL decisions.** Control a decision loss directly (for example, bankroll drawdown or FPL transfer regret) instead of coverage. Losses are non-monotone (Kelly staking) and selection-affected (bet only when edge > threshold), which are exactly the open issues in 2604.01502 and 2512.12844. *Solo work:* empirical study of how often naive CRC violates guarantees under selection, plus a selective-CRC fix. *Success:* realised risk ≤ target across seasons, with ROI or regret reported.
4. **Prediction sets for ordinal/categorical outcomes with conditional (Mondrian) coverage** by subgroup (home/away, favourite/underdog, odds bucket). Quantify the gap between marginal and subgroup coverage. Market-implied probability gives a natural conditioning variable.
5. **Adaptive minimisation of the non-exchangeable coverage gap** (the ACM survey's open problem). Learn weights from time and similarity features (recency, same opponent, same manager) in walk-forward, and report the measured coverage gap versus fixed exponential-decay weights.

### Gaps
- No 2026 source specific to conformal decision theory follow-ups was retrieved.
- No source found on conformal prediction for multivariate *count* outputs (for example, scorelines).

## Q5. Which problems best leverage an EPL/FPL panel dataset with point-in-time features, betting-market prices and walk-forward evaluation?

### Takeaway
That dataset fits the **non-IID regime where TFMs are known to lose** (temporal + grouped) and gives an **unusually strong, leakage-free external benchmark**: closing odds, which bookmaker-level RPS rarely beats. It also gives **genuinely post-cutoff data** (2026/27 season) for contamination studies of TFMs, TSFMs and LLMs. The highest-leverage contributions are evaluation and benchmark papers plus calibration and conformal methods, not new pretraining.

### Cited Findings
- Calibration-based model selection gave higher betting returns than accuracy-based selection. A corrigendum revised the best calibration-driven ROI from 36.93% to 23.13%, with average ROI −9.77% vs −26.78% for accuracy-driven. The mean difference was **not statistically significant**. — [Walsh & Joshi, arXiv 2303.06021](https://arxiv.org/html/2303.06021v3); [MLWA 2024](https://www.sciencedirect.com/science/article/pii/S266682702400015X); [Corrigendum](https://www.sciencedirect.com/science/article/pii/S2666827025000106)
- Bookmaker odds are a strong benchmark: models in a 2026 thesis approached but did not beat the market RPS of 0.2005. — [Bamberg thesis 2026](https://www.uni-bamberg.de/fileadmin/xai/studies/theses/2026/2026_Bachelorthesis_Di_Bao.pdf)
- In-play football (2026): an uncalibrated Weibull model reached 56.4% pre-match accuracy vs 61.4% for Betfair. — [arXiv 2605.16066](https://arxiv.org/html/2605.16066v1)
- LLM agents could not beat the bookmaker on World Cup 2026. — [arXiv 2607.17765](https://arxiv.org/pdf/2607.17765)
- Systematic review of ML in sports betting (techniques, challenges). — [arXiv 2410.21484](https://arxiv.org/pdf/2410.21484)
- TFMs trail GBDTs on temporal and grouped splits. — [BeyondArena arXiv 2606.30410](https://arxiv.org/abs/2606.30410); [Qlik](https://www.qlik.com/blog/results-for-tabular-machine-learning)
- TSFM leakage requires post-cutoff evaluation. — [TSFMAudit](https://arxiv.org/pdf/2605.26161); [fev-bench](https://arxiv.org/html/2509.26468)

### Inferences
**Ranked shortlist for the user's codebase (highest expected value first)**
1. **"Sports-Arena": a walk-forward, market-benchmarked tabular benchmark.** Compare TabPFN-2.5/3, TabICLv2, LimiX, CatBoost, LightGBM, TabM and RealMLP on match outcome (ordinal 1X2), goals (counts) and FPL points (zero-inflated). Report RPS/log-loss/CRPS against de-vigged closing odds. This directly tests the BeyondArena finding in a new domain and could be submitted to the NeurIPS/ICML TRL (Table Representation Learning) workshop. *Compute:* one GPU, days.
2. **TFM context-selection and recency policies for drifting panels** (Q1 item 2). This is a method contribution with a clear ablation story.
3. **Ordinal calibration metric and recalibration for 1X2 forecasts** (Q3 item 1), with market probabilities as a calibrated reference. Theory plus simulation on CPU.
4. **Selective conformal risk control for value-betting / FPL transfer decisions** (Q4 item 3). It addresses open problems in non-monotone and selection-affected CRC.
5. **Contamination-free TSFM evaluation on post-cutoff sports series** (FPL player points per gameweek as short, intermittent, covariate-rich series). It tests covariate gating (Q2 item 1) and hurdle post-processing (Q2 item 4).
6. **LLM + model + market ensembling with leakage audits** (Q3 item 4), extending the World Cup 2026 study to a full EPL season.

**Measurement of success across all of these:**
- Proper scores relative to the market, i.e. a skill score where RPS_skill = 1 − RPS_model/RPS_market, with a block-bootstrap CI by gameweek.
- Calibration (reliability, ECE variants).
- Coverage and width for conformal methods.
- Economic metrics (CLV, closing-line value, and ROI with significance tests), noting the Walsh & Joshi lesson that ROI differences are often not significant.

**Caveats:**
- Market efficiency means the realistic win is often matching the market with public data, or adding value in ensemble, not beating it outright.
- Negative results ("TFMs do not beat GBDTs or the market on walk-forward sports panels") are still publishable at workshops.
- Point-in-time feature hygiene is the critical validity condition.

### Gaps
- No source found evaluating TabPFN/TabICL or Chronos-2/TiRex on football or FPL data, which suggests novelty but could not be confirmed exhaustively.
- Workshop calls for 2026 (TRL @ NeurIPS 2026, time-series workshops) were not retrieved, so venue fit and deadlines are unverified.
- Licensing of betting odds data for publication was not researched.
