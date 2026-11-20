# Attack the time axis, not model scale

The best open ML problems for a strong solo researcher in late 2026 are measurement problems, not modelling problems, and most of them break at the same point: data whose *availability time* differs from its *event time*. The top pick is a **public benchmark of temporal and point-in-time leakage bugs, built by mutating a realistic feature pipeline and paired with an as-of differential detector**. A March 2026 survey reports that **no existing tool checks temporal or target-encoding leakage** ([arXiv 2603.17893](https://arxiv.org/pdf/2603.17893)). The largest empirical leakage study (2,047 datasets) covers only i.i.d. tables ([arXiv 2604.04199](https://arxiv.org/abs/2604.04199)). The whole project runs on a CPU. Second is an audit of whether data-science and ML-engineering agents win benchmarks by leaking. Third is an inference-time gate that decides when covariates help time-series foundation models; one 2026 model found **11 of 30 covariate tasks scored worse with covariates** ([arXiv 2609.24559](https://arxiv.org/pdf/2609.24559)). The same pattern shows up in every area covered:

- Tabular foundation models top i.i.d. leaderboards but lose on temporal and grouped splits.
- Text-to-SQL gold labels are wrong more than half the time.
- Agent-built pipelines that pass snapshot tests at 86–100% pass only 7–79% under replayed late data.
- Optimizer speedups shrink to about 1.1x at 1.2B parameters.
- Random sparse autoencoders (SAEs) score almost as well as trained ones.

Each of these is a measurement gap that one person with one GPU can close and publish. A caveat applies to everything below. The researchers could not reach arxiv.org, so most 2026 claims rest on search snippets and abstracts. arXiv IDs of the form 26xx.xxxxx should be checked before formal citation. Causal ML and decision-focused learning are not covered.

## Twenty-eight problems ranked by tractability, impact and novelty

Each problem gets a score from 1 to 5 on three axes, and the three scores are multiplied:

- **Tractability (T):** 5 means a CPU-only minimum viable result in a weekend and a publishable result within about three months; 2 means many GPU-weeks.
- **Impact (I):** how many practitioners or researchers would change what they do if it worked.
- **Novelty (N):** 5 means the notes found no existing work on the core question.

Ties are broken by novelty first, then by how directly a public benchmark can be reused. The scores are my judgment calls from the evidence below. The ordering of the top eight is robust to ±1 changes on any single axis. Positions 9–20 are effectively a tier and could be reshuffled.

| Rank | Problem | Area | T·I·N | Score | Compute |
|---|---|---|---|---|---|
| 1 | Temporal/point-in-time leakage mutation benchmark plus as-of differential detector | Data-centric / data systems | 5·4·5 | 100 | CPU |
| 2 | Methodological-validity audit of DS/MLE agents ("do agents leak?") | LLM agents for data | 4·4·5 | 80 | CPU + API |
| 3 | Covariate gating for time-series foundation models | TS foundation models | 5·4·4 | 80 | 1 GPU, inference only |
| 4 | Drift-alarm → degradation forecasting benchmark | Data quality | 5·4·4 | 80 | CPU |
| 5 | Tabular FMs on walk-forward temporal + grouped panels, with context selection | Tabular FMs | 4·4·4 | 64 | 1 GPU |
| 6 | Ordinal calibration-error estimator and recalibration | Calibration | 5·3·4 | 60 | CPU |
| 7 | Coverage-checked eval error bars plus simultaneous rank CIs for leaderboards | Evaluation | 5·4·3 | 60 | CPU |
| 8 | Dictionary learning and in-context-algorithm identification inside TabPFN | Mech interp | 4·3·5 | 60 | CPU / 1 GPU |
| 9 | Bitemporal point-in-time reproducibility checker over Iceberg/Delta | Data systems | 4·3·4 | 48 | CPU |
| 10 | Replay robustness of agent-built SQL/dbt pipelines with an as-of oracle | Agents / data systems | 3·4·4 | 48 | CPU + API |
| 11 | Transfer and variance audit of NanoGPT-speedrun techniques | Efficient training | 3·4·4 | 48 | 1 GPU, weeks |
| 12 | Hashed n-gram lookup memory vs parameters at small scale | Efficient training | 4·3·4 | 48 | 1 GPU |
| 13 | Mechanism of verbalised confidence and latent probability in small LLMs | Mech interp | 4·3·4 | 48 | 1 GPU |
| 14 | Faithfulness audit of attribution graphs on known-mechanism prompts | Mech interp | 4·4·3 | 48 | 1 GPU (24 GB) |
| 15 | Model-conditional LLM-judge correction with valid CIs | Evaluation | 4·4·3 | 48 | CPU + API |
| 16 | Post-cutoff contamination-controlled evaluation of TSFMs/TFMs | FMs / evaluation | 4·4·3 | 48 | 1 GPU |
| 17 | Automated gold-SQL auditing by differential execution | LLM agents for data | 4·4·3 | 48 | CPU + API |
| 18 | Occurrence-aware evaluation and hurdle heads for intermittent series | TS foundation models | 5·3·3 | 45 | CPU |
| 19 | Ground-truth toy testbeds for SAE absorption, seed consistency, SPD | Mech interp | 5·3·3 | 45 | CPU / 1 GPU |
| 20 | Event-sequence transformer as an Othello-style world-model testbed | Mech interp | 5·2·4 | 40 | 1 GPU, hours |
| 21 | Online and selective conformal methods for heterogeneous panels and decisions | Conformal | 4·3·3 | 36 | CPU |
| 22 | FP4 optimizer states for matrix optimizers under simulated quantization | Efficient training | 3·3·4 | 36 | 1 GPU |
| 23 | Staleness of IRT/anchor item sets across model generations | Evaluation | 4·3·3 | 36 | CPU + API |
| 24 | Discrimination power of multivariate scoring rules for discrete joint outcomes | Probabilistic forecasting | 5·2·3 | 30 | CPU |
| 25 | Continuous forecasting benchmark with de-vigged market baselines | Evaluation / forecasting | 3·3·3 | 27 | CPU + data cost |
| 26 | Time-aware data valuation | Data-centric | 3·3·3 | 27 | CPU / 1 GPU |
| 27 | Evaluation metrics for synthetic temporal multi-table data | Data-centric | 3·3·3 | 27 | CPU / 1 GPU |
| 28 | Proxy-scale transfer of LLM data-mixture rankings | Data curation | 2·4·3 | 24 | 1 GPU, weeks |

The ranking favours a particular kind of work. Leaderboard-chasing areas score low on novelty: tabular FMs, optimizers and SAEs each have well-funded groups shipping several papers a month. Benchmark-validity and temporal-correctness problems score high on both tractability and novelty because they need careful engineering rather than compute, and because the systems community and the ML community each assume the other has solved them. The sections below go area by area. For each problem they cover why it is open, the current state of the art, what a solo contributor would build, the compute needed and the success metric.

## Data-centric ML: temporal leakage is the clearest uncovered gap

### Rank 1 — A mutation benchmark and as-of detector for temporal leakage

**Why it is open.** Static leakage detection in notebooks is an active niche, but every tool targets the same three i.i.d. leakage types: train/test overlap, preprocessing leakage and multi-test leakage. That framing comes from Yang et al.'s ASE 2022 analysis ([arXiv 2209.03345](https://arxiv.org/abs/2209.03345)). The tools that followed are:

- LeakageDetector, a PyCharm plugin (SANER 2025). Its successor LeakageDetector 2.0 is a VS Code/Jupyter extension with LLM-driven quick fixes. **Its authors did not separately measure detection accuracy** ([arXiv 2509.15971](https://arxiv.org/html/2509.15971)).
- Microsoft's NBLyzer, which uses abstract interpretation to *prove the absence* of leakage ([ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0167642325000772)).
- A tool combining static slicing with structured LLM prompts ([ACM DL](https://dl.acm.org/doi/pdf/10.1145/3808199)).
- A 2026 "grammar of ML workflows" that makes leakage impossible to express in the API ([arXiv 2603.10742](https://arxiv.org/pdf/2603.10742)).

Against this backdrop, the scicode-lint survey found no tool that checks temporal or target-encoding leakage ([arXiv 2603.17893](https://arxiv.org/pdf/2603.17893), snippet-level only). The strongest empirical study, Roth's four-class causal taxonomy tested on **2,047 OpenML/PMLB binary classification datasets**, found that selection leakage matters most and normalization leakage least ([arXiv 2604.04199](https://arxiv.org/abs/2604.04199)). Those datasets are i.i.d., so temporal leakage was never in scope.

Theory explains why the gap persists. "Look-Ahead-Freedom as Temporal Non-Interference" defines a verifiable correctness property for windowing, resampling, joins and point-in-time reads. It then shows that **full look-ahead-freedom is undecidable when availability depends on data values** ([arXiv 2607.04958](https://arxiv.org/html/2607.04958v1)). Static analysis alone therefore cannot be complete, and dynamic testing is required. A third finding sharpens the need for ground truth. "Temporal Leakage in LLM Backtesting" showed that leakage tests can themselves be invalid: **4 of 5 flagship models "failed" a leakage check on the official ForecastBench panel with spurious gaps of up to 0.061**. Clean reference data must therefore come from outside the system being tested ([arXiv 2608.02985](https://arxiv.org/html/2608.02985)). The older Kapoor–Narayanan taxonomy already listed temporal leakage among eight types that undermined results across 17 fields ([Princeton reproducibility project](https://reproducible.cs.princeton.edu/)). What still does not exist is a benchmark with ground-truth temporal bugs.

**What to build.** First, a clean reference pipeline over data where every row carries an entity ID, an `event_at` timestamp and an `available_at` timestamp. Second, a set of leakage *mutation operators*, each of which rewrites the pipeline to introduce one known bug:

- An as-of join turned into a plain join.
- `available_at` swapped for `event_at`.
- Centered rolling windows instead of trailing ones.
- Target encoding fit on the full history.
- Post-event statistics used as pre-event features.
- Late corrections back-filled into history.
- Entity-resolution merges that use future information.
- Random-split cross-validation in place of walk-forward.

Each mutant gets a ground-truth label and a measured inflation of the backtest score. The detector is a hybrid. It combines static column lineage with **dynamic differential testing**: re-materialize every feature at as-of time *t* using only rows with `available_at ≤ t`, then diff against what the pipeline produced. Any mismatch is a concrete witness of leakage.

Three secondary contributions come almost for free:

- An inflation-magnitude model that predicts how much each leak type inflates offline metrics, extending Roth's question to temporal data.
- A mutation score for leakage CI suites: what fraction of injected leaks does a team's existing test suite catch? This directly answers the concern raised in 2608.02985.
- A lineage-only variant that tests whether OpenLineage-style column lineage plus timestamps is enough to flag leaks (rank 9 extends this).

**Compute and data.** CPU only. Use public or synthetic data so the benchmark is redistributable. A synthetic generator gives exact ground truth, and one or two public datasets with real publication lags or late corrections add realism.

**How to measure success.** Detector precision and recall per mutation class (the first published numbers on how detectable temporal leaks are), the R² of the inflation model, and the mutation kill rate of common CI test styles. The notes name DMLR, the NeurIPS Datasets & Benchmarks track, DEEM@SIGMOD and MSR/ICSE-SEIP as venues. Their 2026–27 calls were not checked.

### Rank 4 — Do drift alarms actually predict degradation?

Drift detection is benchmarked as *detection*, not as *prediction of performance loss*. A KDD '26 framework crosses **14 detectors × 7 real datasets × 4 drift types**. It finds detectors do better on feature-space drift than on label changes, and that the strongest (SEED, STEPD) degrade under gradual drift ([arXiv 2606.07789](https://arxiv.org/pdf/2606.07789)). A separate 2026 paper treats false alarms, which trigger needless retraining, as a research problem in their own right ([arXiv 2607.17336](https://arxiv.org/pdf/2607.17336)). Label-free performance estimators such as CBPE and DLE, the industry answer to delayed labels, have only vendor-level evidence behind them ([labelyourdata guide](https://labelyourdata.com/articles/machine-learning/data-drift)). The notes found no independent benchmark of them under real temporal drift.

**What to build.** Assemble real temporal datasets with natural regime changes, freeze a model at each origin, and define ground truth as the realized degradation over the next *k* periods. Score detectors and label-free estimators by AUROC for "degradation > δ within *k* periods", by the calibration of the metric they estimate, and by false-alarm cost. Relaxed functional dependencies, a database-theory approach to drift, are a useful extra detector family ([EDBT 2025](https://openproceedings.org/2025/conf/edbt/paper-246.pdf)).

**Compute.** CPU. This ranks below the top three only because δ and *k* are judgment calls a reviewer can challenge. Sweep them and report how sensitive the conclusions are.

### Rank 9 — Bitemporal reproducibility is not snapshot time travel

Lakehouse table formats add versioning and time travel on top of open files ([Armbrust et al., CIDR 2021](https://people.eecs.berkeley.edu/~matei/papers/2021/cidr_lakehouse.pdf)). A CIDR 2023 study found the design tradeoffs among Delta, Hudi and Iceberg "not well studied" ([CIDR 2023](https://www.cidrdb.org/cidr2023/papers/p92-jain.pdf)). Work since then covers training reproducibility across formats ([arXiv 2508.13396](https://arxiv.org/pdf/2508.13396)) and a combined data, model and code lineage proposal ([arXiv 2503.22754](https://arxiv.org/pdf/2503.22754)). In practice, teams still pin training runs to snapshot IDs ([datafield.dev](https://datafield.dev/advanced-data-science/part-05/chapter-25/)).

The open conceptual gap is this. Time travel returns *transaction* time, meaning what the lake contained. Forecasting needs *availability* time, meaning what was knowable when the prediction was made. Late corrections break the equivalence silently.

**What to build.** Formalize point-in-time reproducibility as a property over a table format plus an availability column, then build a checker that finds divergences on real late-correction histories. A follow-on predicts a retrained model's metric change from a snapshot diff alone, without retraining.

**Compute.** Laptop. Novelty is real: the notes found no 2025–26 VLDB, SIGMOD or CIDR paper on training-data reproducibility over open table formats. Impact is narrower than rank 1, which is why it ranks lower.

### Ranks 26–28 — Valuation, synthetic data and mixtures remain harder solo bets

Three further data-centric problems are open but score lower.

**Time-aware data valuation (rank 26).** Data Shapley in one training run ([arXiv 2406.11011](https://arxiv.org/pdf/2406.11011)) and Data Banzhaf ([arXiv 2205.15466](https://arxiv.org/pdf/2205.15466)) cut the cost of valuation but ignore temporal dynamics. A survey abstract lists temporal dynamics as one of five persistent limitations (the attribution to a specific survey is unverified). Under differential privacy, influence-based valuation fails structurally ([arXiv 2603.00342](https://arxiv.org/html/2603.00342)). A solo study would ask "which past periods or sources help next-period forecasts" using time-aware utilities, and check the answers with retrain-with-top/bottom-k-removed counterfactuals.

**Synthetic temporal multi-table data (rank 27).** Evaluation lacks validated metrics. A systematic review finds that inconsistent metrics lead to "substantially disparate conclusions" ([arXiv 2504.18544](https://arxiv.org/pdf/2504.18544)). TabStruct argues that structural fidelity must be a core evaluation axis across 13 generators and 29 datasets ([arXiv 2509.11950](https://arxiv.org/html/2509.11950v2)). A top synthesizer carries significant membership-privacy risk, and adding differential privacy erases its advantage ([arXiv 2402.06806](https://arxiv.org/html/2402.06806v3)).

**Data-mixture optimization (rank 28).** The field has moved from static methods (RegMix, [arXiv 2407.01492](https://arxiv.org/abs/2407.01492)) to dynamic and proxy-free ones: RegMix-D ([arXiv 2606.18663](https://arxiv.org/pdf/2606.18663), ID flagged as possibly inconsistent), DeMix via model merging ([arXiv 2602.00747](https://arxiv.org/html/2602.00747v2)), and FastMix at a claimed **~1.3 GPU-hours** (author figures; [ICLR 2026](https://proceedings.iclr.cc/paper_files/paper/2026/file/01d845baaa31b1043e5f280592d57dc4-Paper-Conference.pdf)). The core open question, when mixture rankings flip across scale, needs many proxy runs, which is why tractability is low. A cheaper variant has not been explored: apply RegMix-style regression to source and period weights for gradient-boosted models on temporal data. It runs on a CPU and overlaps with rank 5.

**Label errors and data contracts (unranked).** Label-error detection under temporal structure is a gap. Current decomposition benchmarks ([arXiv 2510.16211](https://www.arxiv.org/abs/2510.16211)) and a survey of 34 tabular noisy-label filters ([Springer](https://link.springer.com/article/10.1007/s10462-025-11293-9)) assume i.i.d. data, with Confident Learning still the baseline ([arXiv 1911.00068](https://arxiv.org/abs/1911.00068)). Learned data contracts are another gap: the notes found no strong 2025–26 primary literature on inferring schema and invariant constraints from data history. Both are good side projects for anyone building rank 1, but neither is ranked because the evidence base is too thin to score.

## LLM data agents: benchmarks fail faster than agents improve

### Rank 2 — Do data-science agents win by leaking?

**Why it is open.** Kaggle-derived agent suites reward leaderboard score, and a solution that leaks can score *higher*. The common suites are:

- MLE-bench: 75 competitions ([arXiv 2410.07095](https://arxiv.org/abs/2410.07095)).
- DSBench: 540 tasks.
- DA-Code: 500 tasks.
- DataSciBench: 222 tasks. o1-mini scored **29.77%** overall, under scoring that mixes LLM judging, human judgment and programmed rules ([ICLR 2026 comparison](https://proceedings.iclr.cc/paper_files/paper/2026/file/f72d4fdfd5eb425cd81df9fe6272a533-Paper-Conference.pdf); [DataSciBench](https://aclanthology.org/2026.findings-acl.181.pdf)).

New agents keep arriving: MLE-STAR ([arXiv 2506.15692](https://arxiv.org/pdf/2506.15692)), GRACE-DS ([arXiv 2606.16000](https://arxiv.org/pdf/2606.16000)) and a blackboard multi-agent system reporting 13–57% relative gains on DA-Code ([arXiv 2510.01285](https://arxiv.org/pdf/2510.01285)). None of these benchmarks scores *methodological validity*, and the notes found no benchmark that does.

**What to build.** Run several open and commercial agents on a subset of MLE-bench or DSBench tasks, plus a handful of temporal forecasting tasks where leakage is easy and tempting. Lint every trajectory with LeakageDetector 2.0 for i.i.d. leaks and with a temporal checker (rank 1's detector, or a simple as-of oracle). Then re-score each solution after removing the leak. Report three things: the leak rate per agent, the score inflation the leak buys, and whether agent rankings change (Kendall τ).

**Compute.** CPU plus an API budget. For scale, Spider-Agent with GPT-4o averaged about 43.7 steps and **$2.03 per task** on ELT-Bench ([arXiv 2504.04808](https://arxiv.org/pdf/2504.04808)), so a few hundred agent runs cost hundreds of dollars, not thousands. A paper reporting "agent X's leaderboard lead is mostly leakage" would land at the intersection of the agent and data-quality communities, and no one in the notes has claimed it.

### Rank 10 — Replay testing exposes pipelines that only pass snapshots

ArrivalBench (NeurIPS 2026, 40 tasks) re-runs agent-built pipelines under replayable delivery schedules with late, duplicated, out-of-order and retried records. Single-snapshot grading certifies **86–100%** of pipelines; adversarial replay certifies **7.0–79.2%** across 11 models. Pipelines "repaired" against the snapshot test fail replay about as often as first-pass ones ([NeurIPS 2026](https://neurips.cc/virtual/2026/173048); [arXiv 2610.02363](https://arxiv.org/pdf/2610.02363)). ELT-Bench supplies 100 end-to-end ELT pipelines ([PVLDB](https://www.vldb.org/pvldb/vol19/p84-jin.pdf)). kRAIG reports limited agent performance on 8 of them ([arXiv 2603.20311](https://arxiv.org/pdf/2603.20311)).

**What to build.** Extend replay testing to dbt/SQL bronze-to-gold pipelines over real data with late corrections. Use an as-of oracle plus data contracts as the grader, and report replay pass rate against snapshot pass rate. This reuses rank 1's oracle almost unchanged. It ranks lower only because ArrivalBench already establishes the core finding.

### Rank 17 — Text-to-SQL gold labels are wrong more often than right

A UIUC audit found gold-annotation error rates of **52.8% on BIRD Mini-Dev and 62.8% on Spider 2.0-Snow**, enough to cause "severe misestimation and misranking." The CIDR version reports 66.1% for Spider 2.0-Snow, so cite carefully. Gold SQL is public for only 121 of Spider 2.0-Snow's 547 examples ([arXiv 2601.08778](https://arxiv.org/html/2601.08778v3); [CIDR 2026](https://vldb.org/cidrdb/papers/2026/p5-jin.pdf)). One error, for example, swapped longitude and latitude in ST_POINT. Reported Spider 2.0-Snow scores rose from 23.8% to 96.7%, but that figure comes from a search summary and its primary source is unverified.

**What to build.** GBV-SQL already gives a typology of gold errors ([ACL 2026](https://aclanthology.org/2026.acl-long.837.pdf)), and SpotIt applies formal verification ([arXiv 2510.26840](https://arxiv.org/pdf/2510.26840)). That leaves automated gold auditing by **differential execution across perturbed database instances**, plus released corrected subsets. Success means agreement with expert adjudication and leaderboard ranking stability before and after correction.

**Two further ideas.** Grounding agents in a semantic layer (dbt metric definitions) and scoring data-cleaning agents against verifiable oracles (contracts plus entity-resolution ground truth) are reasonable follow-ups. The notes contain no direct evidence for either, so they are left unranked.

## Tabular and time-series foundation models lose where real data lives

### Rank 3 — Gate covariates at inference time

**Why it is open.** Covariate-aware zero-shot forecasting went mainstream with Chronos-2. It handles past and future covariates and multivariate inputs through group attention, but **does not support static covariates**, and its multivariate skill depends on synthetic multivariate pretraining data ([arXiv 2510.15821](https://arxiv.org/pdf/2510.15821)). Its competitors include:

- TiRex-2, which reports the lowest mean MASE with about 2.2x fewer active parameters and argues Chronos-2 is "neither target-causal nor constant-memory" ([ResearchGate](https://www.researchgate.net/publication/408340932_TiRex-2_Generalizing_TiRex_to_Multivariate_Data_and_Streaming)).
- Toto, TabPFN-TS and TimesFM. fev-bench catalogues which models ignore covariates entirely ([arXiv 2509.26468](https://arxiv.org/html/2509.26468v2)).

The key finding is from t0: **11 of 30 known-future-covariate tasks score worse with covariates, by up to 15%** (15-minute electricity with weather covariates) ([arXiv 2609.24559](https://arxiv.org/pdf/2609.24559); vendor-authored). No principled test exists for deciding at inference time whether covariates will help.

**What to build.** A held-out-window gate: use covariates only if they improve backtest CRPS on the most recent windows. Compare it against covariate-dropout ensembles and against always-on and always-off covariates, on the fev-bench covariate subset. A natural extension injects static covariates as constant pseudo-channels, or stacks a tabular FM on entity features with TSFM residuals.

**Compute.** Inference only, on one GPU. No data needs to be collected, because the benchmark already exists.

**How to measure success.** The share of tasks where the gated forecast is at least as good as the better of with-covariates and without, plus the aggregate skill score against Chronos-2 with covariates always on. This ranks third because it is the most "plug into an existing leaderboard" project in the report.

### Rank 5 — Tabular FMs on walk-forward panels and the context-selection question

**State of the art.** On the TabArena leaderboard (v0.1.4 snapshot) the top four single models are all tabular FMs:

- **TabPFN-3 scores about 1673 Elo against 1375 for tuned, ensembled XGBoost** ([CodeSOTA](https://www.codesota.com/tasks/tabular-ml); [TabPFN-3 report](https://arxiv.org/pdf/2605.13986)). The report itself lists 1677 and 1631 in different tables.
- TabICLv2 sits around 1596–1613 with no tuning ([arXiv 2602.11139](https://arxiv.org/pdf/2602.11139)).
- A "TabPFN-3.5-Thinking" variant reportedly reaches about 1910 Elo, but this comes from a snippet only ([arXiv 2609.17895](https://arxiv.org/pdf/2609.17895)).
- LimiX-2 figures could not be reconciled across leaderboard views.

TabArena itself is a living benchmark ([arXiv 2506.16791](https://arxiv.org/abs/2506.16791)).

**Why it is open.** These models fail outside the i.i.d. setting. BeyondArena evaluates 11 models on 142 datasets. Open-source tabular FMs are competitive up to about 100k rows, but **tree-based and deep-learning models still dominate on grouped or large data** ([arXiv 2606.30410](https://arxiv.org/abs/2606.30410)). An industry summary adds that on temporal and grouped splits the 2026 leaders trail tuned RealMLP, TabM and gradient-boosted trees, and on very large tables fall below default XGBoost ([Qlik](https://www.qlik.com/blog/results-for-tabular-machine-learning)).

The same blog makes two further claims, which should be checked against arXiv 2608.10280. First, fine-tuning most tabular FMs *lowers accuracy and raises calibration error together*, with TabPFNv2 the exception. Second, TabPFN behaves like an implicit retriever. Drift-Resilient TabPFN (NeurIPS 2024) learned temporal shift in context ([paper](https://proceedings.neurips.cc/paper_files/paper/2024/file/b2e2774c8e76afe191b5bf518f5cb727-Paper-Conference.pdf)), but no newer model that targets drift is clearly established.

**What to build.** A walk-forward benchmark on public panel datasets that combines temporal *and* grouped splits with strictly point-in-time features, a combination almost no public benchmark offers. On it, compare TabICLv2 or TabPFN zero-shot, fine-tuned, and used as a head on embeddings, against CatBoost, LightGBM, TabM and RealMLP. Using TabPFN in place of a linear head on fine-tuned embeddings cut calibration error with no gradient updates in one study ([arXiv 2607.11007](https://arxiv.org/html/2607.11007v2)).

The method contribution is **context selection under drift**: which past rows go into the in-context window. Candidates are the most recent rows, same-entity rows, kNN retrieval, or recency-weighted sampling. If these models really behave like retrievers, this choice is the main lever.

**Compute.** One GPU. Contexts of 10k–50k rows are feasible.

**How to measure success.** Proper scores against the "all history" context at matched compute, ablations by temporal distance, and calibration error before and after cheap fixes such as temperature scaling, Venn-Abers and conformal. Negative results ("tabular FMs do not beat tuned trees on drifting panels") are publishable at workshops such as Table Representation Learning; its 2026 call was not checked.

### Rank 16 — Contamination needs post-cutoff data, not loss probes

Benchmark leakage is widespread in time-series FMs:

- GIFT-Eval concedes partial leakage and ships a non-leaking pretraining corpus of about 230B points ([arXiv 2410.10393](https://arxiv.org/pdf/2410.10393)).
- fev-bench relies on self-reported pretraining overlap ([arXiv 2509.26468](https://arxiv.org/html/2509.26468)).
- A 2026 paper cites **28% fev-bench overlap for Moirai-2.0 and 10% for TimesFM-2.5** (via snippet of [arXiv 2609.24559](https://arxiv.org/pdf/2609.24559)).
- TSFMAudit shows that LLM contamination detectors do not transfer to time series, because smooth or seasonal series are easy to forecast anyway ([arXiv 2605.26161](https://arxiv.org/pdf/2605.26161)).
- For tabular FMs pretrained on real data, a 2026 critical review lists contamination between pretraining and evaluation sets as an open problem ([MAKE review](https://doi.org/10.3390/make8080244)).

**What to build.** A "fresh" evaluation slice of series and tables whose observations postdate every model's cutoff, matched to older slices by forecastability (for example, spectral entropy). The deliverable is the leaderboard rank change after controlling for leakage. A source-first audit framework already exists ([arXiv 2609.21425](https://arxiv.org/pdf/2609.21425)), so novelty scores 3.

### Rank 18 — Intermittent series need occurrence-aware scoring

MAE-type scores are minimized by the median, so on sparse series they reward all-zero forecasts. A 2026 paper proposes occurrence-focused evaluation grounded in hurdle and zero-inflated models ([arXiv 2609.39386](https://arxiv.org/html/2609.39386v1)). TimesFM-3 roughly matches, but slightly trails, iETS and negative-binomial baselines on simulated intermittent data ([arXiv 2610.00589](https://arxiv.org/pdf/2610.00589)).

**What to build.** A hurdle post-processor on top of TSFM quantiles, scored by Brier score on P(y>0) plus CRPS conditional on activity. CPU work. Novelty is moderate because the framing paper already exists. Hierarchical coherence of TSFM forecasts is another candidate, but the notes found no 2026 source on it, so it is unranked.

## Calibration and conformal prediction: mature theory, open structured cases

### Rank 6 — Nobody has an ordinal calibration-error estimator

Multiclass calibration has had a burst of 2025–26 work, none of it ordinal:

- Local calibration with Jensen-Shannon distance ([arXiv 2510.26566](https://arxiv.org/abs/2510.26566)).
- Truthful calibration errors ([arXiv 2510.06388](https://arxiv.org/html/2510.06388)).
- Utility-aware calibration ([arXiv 2510.25458](https://arxiv.org/pdf/2510.25458)).
- Region-wise calibration via vector quantization ([arXiv 2605.21060](https://arxiv.org/abs/2605.21060)).
- Online multicalibration ([arXiv 2610.15954](https://arxiv.org/html/2610.15954)).
- Hierarchical utility calibration ([arXiv 2609.36532](https://arxiv.org/abs/2609.36532)).
- Linear-log-odds recalibration ([arXiv 2602.18573](https://arxiv.org/pdf/2602.18573)).

The researchers' search found **no ordinal-specific calibration-error work**. That is a search null result, not proof of absence, so the first step is a literature sweep that includes the ordinal-regression literature.

**What to build.** Ordered outcomes such as ratings, severity grades, Likert responses and win/draw/loss already have a standard proper score (the ranked probability score), but no standard calibration diagnostic. Define cumulative-threshold calibration (calibrate P(Y ≤ k) for every k), derive the estimator's bias and variance, build reliability diagrams, and test power against planted miscalibration in simulation.

**Compute.** CPU.

**How to measure success.** Detection power at realistic sample sizes, and the gain in ranked probability score after recalibration on real ordinal data. This is a clean theory-plus-simulation paper with a short path to a usable library.

### Rank 24 — Multivariate scoring rules barely discriminate dependence

Most of this literature dates from 2015–2023, and the notes found no 2026 resolution:

- The energy score has a derived upper bound on how well it can discriminate dependence structure ([DTU Orbit](https://orbit.dtu.dk/en/publications/discrimination-ability-of-the-energy-score/)).
- Variogram scores discriminate better ([Scheuerer & Hamill 2015](https://journals.ametsoc.org/view/journals/mwre/143/4/mwr-d-14-00269.1.xml)).
- **CRPS-Sum can rank random noise above state-of-the-art methods** ([arXiv 2201.08671](https://arxiv.org/pdf/2201.08671)).
- Propriety does not guarantee good discrimination at finite sample sizes ([Marcotte et al., ICML 2023](https://proceedings.mlr.press/v202/marcotte23a/marcotte23a.pdf)).

**What to build.** Extend the Marcotte analysis to *discrete* joint outcomes such as bivariate counts. Measure which score detects planted dependence misspecification at a few hundred observations. CPU only and tractable, but the audience is narrow.

### Rank 21 — Conformal methods: the gaps are adaptive weighting, selection and panels

The core toolkit is mature. It includes ACI ([arXiv 2106.00170](https://arxiv.org/abs/2106.00170)), conformal PID ([arXiv 2307.16895](https://arxiv.org/abs/2307.16895)), conformal risk control (CRC, [arXiv 2208.02814](https://arxiv.org/abs/2208.02814)), conformal decision theory ([arXiv 2310.05921](https://arxiv.org/abs/2310.05921)) and CP beyond exchangeability ([arXiv 2202.13415](https://arxiv.org/abs/2202.13415)).

Four problems remain open:

- **Adaptive weighting.** A January 2026 ACM survey says that non-exchangeable CP bounds the coverage gap by a distribution distance, and that *adaptively minimizing* that gap is still open ([ACM CSUR](https://dl.acm.org/doi/full/10.1145/3736575)).
- **Conditional coverage.** Distribution-free conditional coverage is impossible in finite samples ([arXiv 2410.13115](https://arxiv.org/pdf/2410.13115)).
- **Non-monotone losses.** For CRC with non-monotone losses, risk control depends on how the sample size grows relative to the grid resolution ([arXiv 2604.01502](https://arxiv.org/html/2604.01502v2)).
- **Feasibility.** For LLM certification, ACI cuts cross-dataset risk violations **from 71% to 21%** but cannot overcome a feasibility limit ([arXiv 2606.29054](https://arxiv.org/html/2606.29054v2)).

Selection breaks exchangeability, which motivated selective CRC ([arXiv 2512.12844](https://arxiv.org/pdf/2512.12844)). Online CP for non-exchangeable panel data is a 2026 front ([arXiv 2605.17705](https://arxiv.org/pdf/2605.17705)).

**What to build.** Any of three projects:

- Hierarchical or shrinkage α-updates for panels of hundreds of short, heterogeneous entity series, scored by worst-decile per-entity coverage.
- Conformalized multi-horizon intervals on top of TSFM quantiles, in the style of JANET ([arXiv 2407.06390](https://arxiv.org/pdf/2407.06390)).
- An empirical study of how often naive CRC breaks its guarantees when the pipeline acts only on predictions above a threshold, with selective CRC as the fix.

**Compute.** CPU. The low rank reflects crowding (several 2026 papers per sub-problem), not unimportance.

### Rank 25 — Market-baselined forecasting benchmarks exist but lack power

Live-event forecasting is an established benchmark genre:

- ForecastBench ([arXiv 2409.19839](https://arxiv.org/html/2409.19839v5)).
- Prophet Arena, which draws events from Kalshi and Polymarket ([arXiv 2510.17638](https://arxiv.org/html/2510.17638v2)).
- Foresight Arena ([arXiv 2605.00420](https://arxiv.org/pdf/2605.00420)).
- A wave of 2026 sports papers.

The "superforecaster parity" claim is disputed. The Forecasting Research Institute concedes the confidence intervals overlap substantially ([FRI](https://forecastingresearch.substack.com/p/ai-models-have-likely-reached-parity)). Good Judgment argues the preliminary board favours AI-friendly question types and uses a frozen 2024 human baseline ([Good Judgment](https://goodjudgment.substack.com/p/not-so-fast)). One non-peer-reviewed analysis found **442 of 527 ForecastBench forecasters scored worse than the market price they were shown** ([EA Forum](https://forum.effectivealtruism.org/posts/8YuG9EuhyYqaNBt7F/the-copier-line-what-forecastbench-s-market-scores-actually)).

Sports results show the power problem directly:

- LLM-SoccerArena's best model scored a **Brier of 0.497 versus 0.498 for de-margined closing odds**, which is not significant ([arXiv 2607.24573](https://arxiv.org/pdf/2607.24573)).
- A 104-match World Cup 2026 study found no LLM agent beat the bookmaker ([arXiv 2607.17765](https://arxiv.org/pdf/2607.17765)).
- Econometric work finds the Asian-handicap market efficient even where 1X2 odds show favourite–longshot bias ([Hegarty & Whelan, IJF 2025](https://econpapers.repec.org/RePEc:eee:intfor:v:41:y:2025:i:2:p:803-820)).

**What to build.** A continuous, multi-domain benchmark with de-vigged opening *and* closing lines, Diebold–Mariano-style paired tests clustered by date, and a published minimum detectable edge. A cheaper variant would test whether scoring a forecast against the market price at t+k predicts its final proper score, which would allow evaluation before questions resolve. The low rank comes from odds licensing and cost, and from at least five ad-hoc 2026 prototypes already in the field. The statistical sub-problem is stronger than the benchmark itself and folds naturally into rank 7.

## Evaluation methodology: the statistics are settled, adoption and hard cases are not

### Rank 7 — Test whether eval error bars actually cover

**Why it is open.** The basic recipe is known. Miller recommends clustered standard errors and paired question-level tests, and shows **clustered SEs can be more than 3x naive SEs on DROP** ([arXiv 2411.00640](https://arxiv.org/abs/2411.00640v1)). Bowyer et al. argue that CLT intervals "dramatically underestimate uncertainty" on small benchmarks and recommend Bayesian methods instead ([arXiv 2503.01747](https://pdf.arxiv.org/abs/2503.01747)). An `errorbars` package already implements cluster-robust SEs for lm-eval-harness and Inspect logs ([PyPI](https://pypi.org/project/errorbars/)).

The hard cases are what remain:

- **Saturation.** **29 of 60 benchmarks show high or very high saturation**, and private test sets do not prevent it ([arXiv 2602.16763](https://arxiv.org/html/2602.16763); [ICML 2026 slides](https://icml.cc/media/icml-2026/Slides/63311.pdf)).
- **Agent time horizons.** METR's 50% time horizon for Claude Opus 4.5 is about 4h49m, with a 95% CI of 1h49m to 20h25m, a span of more than 10x ([METR note](https://metr.org/notes/2026-01-22-time-horizon-limitations/)). A regularization change lowered some horizons by up to 20% ([METR](https://metr.org/notes/2026-03-20-impact-of-modelling-assumptions-on-time-horizon-results/)).
- **Multiple comparisons.** The notes found no 2025–26 paper establishing false-discovery-rate (FDR) control or rank confidence sets for leaderboards.

**What to build.** Resample public item-level logs to measure the *empirical coverage* of nominal 95% intervals for five methods: naive CLT, clustered CR1, cluster bootstrap, hierarchical Beta-Binomial and IRT posteriors. Then compute simultaneous rank confidence sets for existing leaderboards and count how many state-of-the-art claims survive FDR 0.05. A hierarchical Bayesian refit of the time-horizon curve with task-family random effects is a natural third act.

**Compute.** CPU.

**How to measure success.** Coverage close to nominal, a lower rank-flip rate, and a merged pull request to a mainstream harness. This applies equally to the forecasting leaderboards in rank 25 and to speedrun records in rank 11.

### Rank 15 — Judge corrections fail exactly when judges differ across models

High test-retest agreement can coexist with severe position bias across 21 judge models, and verbosity bias is much smaller than 2023-era reports suggested ([arXiv 2606.19544](https://arxiv.org/abs/2606.19544), abstract only). Existing correction estimators become biased when the test and calibration distributions differ ([arXiv 2511.21140](https://arxiv.org/html/2511.21140v3)). Model comparisons fail when judge calibration differs across the models being compared ([arXiv 2605.06939](https://arxiv.org/html/2605.06939v1)). Swapping judges changes the construct being measured ([arXiv 2607.08535](https://arxiv.org/pdf/2607.08535)).

**What to build.** A prediction-powered inference estimator ([arXiv 2301.09633](https://arxiv.org/abs/2301.09633)) with a *per-model* judge confusion matrix, plus active allocation of a small human-label budget.

**How to measure success.** Bias and CI coverage against full human labels on public human-judgment datasets. CPU plus modest API.

### Rank 23 — Do cheap IRT item sets go stale?

tinyBenchmarks estimates MMLU accuracy from about 100 IRT-selected items ([arXiv 2402.14992](https://arxiv.org/html/2402.14992v2)). LEGO-IRT ([arXiv 2510.04051](https://arxiv.org/html/2510.04051v1)) and the ATLAS adaptive-testing method ([arXiv 2511.04689](https://arxiv.org/html/2511.04689v2)) extend the approach. IRT fitting remains "computationally expensive and numerically unstable" ([arXiv 2605.07046](https://arxiv.org/html/2605.07046)).

**What to build.** Measure how fast anchor-item estimates degrade on models released after the items were calibrated, then propose item banks that recalibrate themselves. A related idea is an IRT-based early warning of benchmark saturation.

**How to measure success.** Mean absolute error at a 100-item budget on post-calibration models.

## Efficient training: a single GPU can audit records, not set them

### Rank 11 — Do speedrun tricks survive scale and seeds?

The NanoGPT speedrun trains a 124M model to 3.28 FineWeb validation loss on 8xH100 ([modded-nanogpt](https://github.com/KellerJordan/modded-nanogpt)). Its record has fallen from **45 minutes in May 2024 to 0.665 minutes (record #92, 30 Aug 2026)**. Record #92 combined sampled softmax, full FP8 MLPs, an 84.6M-row hashed n-gram table, an "ANVIL" optimizer stack and full CUDA-graph capture. The 350M track has been stuck at 17.35 minutes since December 2025.

Optimizer claims are much smaller once baselines are tuned. Against tuned AdamW, speedups **never exceed about 1.4x and fall to about 1.1x at 1.2B**, and setting the learning rate at 2x its optimum erases them ([Wen et al., arXiv 2509.02046](https://arxiv.org/html/2509.02046v2)). Most speedrun-born optimizer variants have not been validated beyond 1B parameters against tuned baselines.

**What to build.** A transfer audit of the top 10–15 speedrun innovations against a tuned baseline at two or three scales and token budgets, with seed variance. Add a significance protocol for time-to-target, since records often improve by less than 1% (for example, 1.266 to 1.256 minutes).

**Compute.** One GPU for weeks, or a few rented 8xH100 hours for official-scale variance runs.

**How to measure success.** Speedup-versus-scale curves with confidence intervals.

### Rank 12 — Lookup memory as a substitute for parameters

Record #92's hashed n-gram table and the earlier bigram hash embedding (which reached 1.655 minutes in January 2026) suggest that cheap lookup memory substitutes for compute at small scale ([modded-nanogpt](https://github.com/KellerJordan/modded-nanogpt)). This bears directly on artifact-size rules such as OpenAI's Parameter Golf: a 16MB artifact trained in under 10 minutes ([rules mirror](https://github.com/sekinal/parameter-golf)), which drew 1,000+ participants and 2,000+ submissions ([OpenAI](https://openai.com/index/what-parameter-golf-taught-us/), snippet only).

**What to build.** Characterize how the benefit of lookup memory scales and where it disappears, and how it should count toward parameter totals. One GPU.

### Rank 22 — Fully-FP4 training is still unsolved where solo work fits

NVIDIA trained a 12B model on 10T tokens in NVFP4 at FP8 parity ([arXiv 2509.25149](https://arxiv.org/html/2509.25149v2)). Yet TetraJet-v2 states that fully-FP4 optimization problems "remain unsolved," citing weight oscillation and outliers ([arXiv 2510.27527](https://arxiv.org/html/2510.27527v1)). Full-Stack FP4 notes that optimizer states and attention remain underexplored, and reports a **1.47% loss gap versus BF16 at 3B parameters and 64B tokens** ([arXiv 2607.04422](https://arxiv.org/abs/2607.04422)).

**What to build.** Fake-quantized FP4 optimizer states for Muon or SOAP at 350M parameters or fewer. Matrix optimizers keep state with different statistics from Adam, which is why they need separate study.

**Limits.** Kernel-speed claims cannot be tested without Blackwell-class hardware.

**How to measure success.** Loss gap versus BF16 and divergence rate.

**Not researched.** Speculative decoding, KV-cache compression, small-scale MoE, distillation and test-time compute were not searched. The report has no verified 2026 state of the art for them.

## Mechanistic interpretability: ground-truth testbeds beat another SAE variant

The field's agenda is still Sharkey et al.'s "Open Problems in Mechanistic Interpretability" ([arXiv 2501.16496](https://arxiv.org/abs/2501.16496)). The GDM team has since pivoted toward "pragmatic" interpretability judged by proxy tasks ([LessWrong](https://www.lesswrong.com/posts/StENzDcD3kpfGJssR/a-pragmatic-vision-for-interpretability)). A November 2026 survey names evaluation methodology the most pressing open problem ([arXiv 2611.08438](https://arxiv.org/pdf/2611.08438), snippet).

Sparse autoencoders have lost their default status:

- SAE probes across 113 datasets never consistently beat baseline-only ensembles ([arXiv 2502.16681](https://arxiv.org/pdf/2502.16681.pdf)).
- Simple baselines beat SAEs at steering ([AxBench](https://export.arxiv.org/pdf/2501.17148)).
- **Random baselines nearly match trained SAEs**: 0.87 vs 0.90 on interpretability and 0.69 vs 0.72 on sparse probing ([arXiv 2602.14111](https://arxiv.org/pdf/2602.14111)).

The free tooling is strong. Gemma Scope 2 provides SAEs and transcoders on every layer of every Gemma 3 size from 270M to 27B ([DeepMind](https://deepmind.google/blog/gemma-scope-2-helping-the-ai-safety-community-deepen-understanding-of-complex-language-model-behavior)). Anthropic's open-source circuit-tracer supports Gemma-2-2B, Llama-3.2-1B and Qwen3-4B ([Anthropic](https://anthropic.com/research/open-source-circuit-tracing)). Every project below runs on one GPU.

### Rank 8 — Open TabPFN with dictionaries and patching

Tabular FM interpretability began in 2026:

- Probing found regression coefficients decodable in TabPFN v2's middle layers ([arXiv 2601.08181](https://arxiv.org/html/2601.08181v1)).
- Researchers searched for concept-selective "grandmother" neurons ([arXiv 2601.03657](https://arxiv.org/pdf/2601.03657)).
- An ICML 2026 paper localized attention-head function causally in TabPFN 2.5. It found that **contrastive steering fails to transfer across samples**, because in-context learning relies on context-dependent attention ([ICML 2026](https://icml.cc/virtual/2026/71542)).

The notes found no dictionary-learning or circuit-level study of a tabular FM.

**What to build.** Train SAEs on TabPFN residuals over synthetic priors and check whether the latents match known generating parameters: coefficients, noise level and feature relevance. Then use activation patching to test whether TabPFN implements ridge regression, kNN or Bayesian model averaging in context.

**Compute.** Small. The notes estimate CPU for small tables or one GPU, but this was not verified.

**Why it ranks highly.** The work also informs rank 5. If TabPFN is a retriever, context selection is the lever.

### Rank 13 — Find the circuit behind "how confident are you?"

Linear probes explain variance in verbalized confidence beyond token log-probabilities ([arXiv 2603.17839](https://arxiv.org/html/2603.17839v2)). Cross-layer agreement matches probes in-distribution ([arXiv 2603.22299](https://arxiv.org/abs/2603.22299)). Confidence probes transfer across languages ([arXiv 2605.31220](https://arxiv.org/html/2605.31220)). No mechanism has been identified.

**What to build.** Attribution graphs on confidence prompts for Gemma 3 or Qwen3-4B. Add synthetic Bayesian tasks with known posteriors, to test whether models represent log-odds linearly. A third experiment compares a probe-read probability against the verbal one on post-cutoff resolved forecasting questions.

**How to measure success.** Patching that changes stated confidence without changing the answer, and Brier scores for probe-read versus verbal probabilities.

### Rank 14 — Check attribution graphs against real interventions

Transcoders produce features that are significantly more interpretable than SAE features ([arXiv 2501.18823](https://arxiv.org/abs/2501.18823)), and they underpin attribution graphs. Neuronpedia frames those graphs as hypotheses to be validated ([Neuronpedia](https://neuronpedia.org/graph/info)). Manual tracing costs about two hours per prompt ([LessWrong](https://www.lesswrong.com/posts/zQqGhKPqaCBZZDCge)). One 2026 paper finds neuron bases more faithful than sparse-dictionary bases ([arXiv 2601.22594](https://arxiv.org/html/2601.22594)).

**What to build.** Build graphs for prompt families with known mechanisms, such as indirect-object identification and two-hop recall. Correlate the intervention effects each graph predicts with actual ablation effects, and measure how much of the logit effect routes through error nodes. Anthropic's October 2026 J++ lens has only internal comparisons ([Circuits Update](https://transformer-circuits.pub/2026/october-update/)), so independent lens benchmarking is a cheap add-on.

### Ranks 19–20 — Toy worlds with known answers

**Rank 19: toy ground-truth testbeds.** Three findings motivate them:

- Feature absorption occurs in every LLM SAE tested and is not fixed by changing size or sparsity ([arXiv 2409.14507](https://arxiv.org/abs/2409.14507v6)).
- Matryoshka SAEs claim a fix, but only the authors have tested it ([arXiv 2503.17547](https://arxiv.org/html/2503.17547v1)).
- SAEs fail to find canonical units ([arXiv 2502.04878](https://arxiv.org/abs/2502.04878)).

Weight-sparse transformers have small circuits by construction ([arXiv 2511.13653](https://arxiv.org/pdf/2511.13653)), and stochastic parameter decomposition has been validated mostly on toy models ([arXiv 2506.20790](https://arxiv.org/html/2506.20790v2)). A solo researcher can plant hierarchical features or train a weight-sparse model on a synthetic task, then score SAE variants, transcoders and SPD on recovery of the known structure.

**Rank 20: an event-sequence world model.** Train a 2–8-layer transformer on public sequential event data (for example, open football event logs) and probe for latent state it never sees directly. This is the Othello-GPT recipe ([arXiv 2309.00941](https://arxiv.org/pdf/2309.00941)) in a new domain. Event models such as EventGPT exist ([arXiv 2512.17266](https://arxiv.org/pdf/2512.17266)), but the notes found no interpretability work on them. It is novel and trivially cheap, but its impact outside that domain is limited.

## Pick temporal leakage first; here is the weekend

**#1: the temporal-leakage mutation benchmark and as-of detector.** It has the best combination of three properties. It addresses a gap that a 2026 survey names explicitly. It runs entirely on a CPU. And it produces reusable infrastructure: the as-of oracle that re-materializes features from `available_at ≤ t` is the same component that ranks 2, 9 and 10 need. A paper, a pip-installable checker and a public mutant suite give three routes to impact.

The main risk is that the scicode-lint claim of "no temporal tool" rests on a snippet. The first task of the weekend is therefore to verify it. A second risk is that reviewers see mutation benchmarks as "just engineering". Shipping the inflation-magnitude analysis and detector recall numbers alongside the benchmark answers that.

**#2: the agent methodological-validity audit.** It has the highest novelty in the report. It answers a question agent-leaderboard maintainers will care about ("how much of this score is leakage?"), and a few hundred dollars of API calls cover it. It stands alone using existing i.i.d. linters, and it gets stronger once #1's temporal checker exists.

**#3: covariate gating for time-series FMs.** It is the best pick for someone who wants a modelling result rather than infrastructure. The benchmark (fev-bench's covariate subset) and models (Chronos-2, TiRex-2, Toto) are public, the motivating failure is documented, and the minimum viable experiment is inference-only. Its weakness is that the motivating 11-of-30 result comes from a vendor paper, so replicating that finding is the first week's job.

The first-weekend plan for #1 aims at a go/no-go decision by Sunday night, not a finished benchmark.

| Block | Work | Output |
|---|---|---|
| Saturday morning (3–4 h) | Read the full texts of scicode-lint (2603.17893), Roth (2604.04199), the look-ahead-freedom paper (2607.04958), the LLM-backtest leakage paper (2608.02985) and LeakageDetector 2.0, and confirm that no temporal-leakage tool or benchmark exists. Fix the data model: every row carries `entity_id`, `event_at`, `available_at` and `ingested_at`. | One-page spec plus a written novelty check |
| Saturday afternoon (4–5 h) | Build a clean reference pipeline in DuckDB SQL plus Python. Start with a synthetic generator that plants a signal, reporting lags, late corrections and entity merges. Add one public dataset with real publication lags or revisions; any source with documented revisions works, and public results-plus-closing-odds files where post-event statistics can leak into pre-event features are one option. Implement rolling-window features, an as-of join, target encoding and a walk-forward split. Implement the **as-of oracle**: re-materialize each feature at *t* from rows with `available_at ≤ t`. | Clean pipeline with zero oracle mismatches |
| Sunday morning (4 h) | Implement eight mutation operators as SQL or Python rewrites: plain join instead of as-of join; `event_at` instead of `available_at`; centered windows; full-history target encoding; post-event statistics as pre-event features; back-filled late corrections; a future-aware entity merge; and random-split CV. Run each mutant through the oracle diff, then through walk-forward evaluation with logistic regression and LightGBM. Compute score inflation with block-bootstrap CIs. | Table: mutant × oracle-detected? × log-loss/Brier inflation with CI |
| Sunday afternoon (3–4 h) | Run LeakageDetector 2.0, scicode-lint if it is installable, and a prompted LLM code reviewer on every mutant's source. Record recall per mutation class. Push the repo with a README and draft a 150-word abstract. | First detectability table, public repo, go/no-go note |

**Go/no-go rule.** Proceed if two conditions both hold. First, at least three temporal mutation classes produce inflation whose confidence interval excludes zero. Second, static tools and the LLM reviewer miss most of those classes while the oracle catches them. If static tools already catch the temporal mutants, pivot. Either keep the benchmark and make the inflation-magnitude model the headline, or move to the agent audit (#2), which reuses the same mutants as probes.

**After the weekend.** Over four to eight weeks:

- Grow the operators to about twenty.
- Add mutation scoring for common CI styles: pytest assertions, dbt tests and expectation suites.
- Fit the inflation model across operator × lag × model family.
- Reproduce at least two documented real-world temporal bugs.
- Submit to a datasets-and-benchmarks or data-management-for-ML venue.

## What this report cannot tell you

Several limits bound these conclusions:

- **Causal ML and decision-focused learning are not covered.** The user stopped that research thread before notes were saved. Problems such as predict-then-optimize evaluation, uplift modelling and causal discovery on small data were not assessed, and their absence from the ranking does not mean they are less promising.
- **The sports-analytics thread was stopped partway.** This report keeps only its standalone findings: market efficiency, the power problem in market-baselined benchmarks, and event-sequence models as interpretability testbeds.
- **Earlier project-fit framing was dropped.** The notes originally assessed fit to a specific existing codebase. That framing was removed at the user's request, and every problem here is scoped to public data.
- **The evidence base is thin in places.** The researchers could not fetch arxiv.org, metr.org, openai.com or lesswrong.com, so most 2026 claims come from search snippets and abstracts. Several numbers are flagged as unverified or vendor-sourced: TabPFN-3.5's ~1910 Elo, the LimiX-2 Elo values, Spider 2.0-Snow's 96.7% score and 62.8%-vs-66.1% error rate, the t0 covariate and leakage figures, the Qlik fine-tuning claims, FastMix's 1.3 GPU-hours, and the RegMix-D ID.
- **Absence claims rest on limited searches.** Statements of the form "no existing tool or benchmark was found" include no temporal-leakage tool, no ordinal calibration metric, no leaderboard FDR paper and no interpretability of event models. Each rests on a small number of searches and should be re-checked before it is used to claim novelty.
- **Some areas were not researched at all.** Speculative decoding, KV-cache compression, MoE, distillation and test-time-compute efficiency are among them. Venue deadlines and data-licensing terms were not checked either.

## Conclusion

The common thread is that **"time" is the missing dimension in ML's evaluation infrastructure**. Leakage linters, tabular-FM leaderboards, agent benchmarks, drift benchmarks, conformal guarantees and LLM-forecasting arenas all break at the same place: the gap between when something happened and when it could have been known. The systems community has bitemporal tables and the ML community has walk-forward splits, but no shared, tested component connects them. A solo researcher who builds one rigorously validated as-of oracle can reuse it across at least five of the top ten problems. That is a stronger compounding position than any single modelling result.

The second implication concerns what counts as a contribution in 2026. Many headline numbers in this report are being deflated: optimizer speedups, SAE utility, text-to-SQL scores, snapshot-certified pipelines and AI-forecasting parity. In each case a careful outsider with modest compute showed that the measurement, not the model, was wrong. Well-resourced labs are under-incentivized to produce these negative and validity results, which leaves them open to anyone who brings statistical care and systems discipline. A solo project here is likely to end with a corrected measurement rather than a new state of the art, and in these areas the corrected measurement is the more lasting contribution.
