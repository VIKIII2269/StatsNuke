# Open Problems in Data-Centric ML, ML Data Infrastructure, and LLM Agents for Data Engineering/Science (as of Nov 2026)

Method note: research ran on 2026-11-20 with web search only. **arxiv.org could not be fetched from this session** (DNS blocked), so arXiv-hosted claims come from search-result abstracts and snippets, not full-text reads. Some search snippets had inconsistent dates or IDs, and those are flagged. Items marked "(prior knowledge)" are well-known papers cited from memory with their canonical arXiv URL. Their content was not re-read this session.

Each problem card below lists: **Why open / SOTA / Solo contribution / Compute / Success metric**.

---

## 1. Data leakage and point-in-time (PIT) correctness

### Takeaway
Static leakage detection in notebooks is a busy 2025–2026 niche (LeakageDetector 1/2, abstract interpretation, slicing+LLM, and "make leakage inexpressible" grammars). Existing tools still target only overlap, preprocessing and multi-test leakage. A March 2026 survey says **no existing tool checks temporal or target-encoding leakage**, and formal work (2607.04958) shows full look-ahead-freedom is undecidable when availability depends on data values. That leaves temporal/PIT leakage in real feature pipelines as the clearest open gap, and a solo builder with a production PIT lake is well placed to fill it.

### Cited Findings
- Yang et al. (ASE 2022) introduced static detection of three leakage types in notebooks (overlap, preprocessing, multi-test). Later tools build on this analysis. — [Yang et al., arXiv 2209.03345 (prior knowledge)](https://arxiv.org/abs/2209.03345); described as the base analysis in [LeakageDetector 2.0, arXiv 2509.15971](https://arxiv.org/html/2509.15971)
- LeakageDetector (SANER 2025, arXiv 2503.14723) is a PyCharm plugin wrapping a Dockerized static analyzer. LeakageDetector 2.0 (Sep 2025, arXiv 2509.15971) is a VS Code/Jupyter extension that adds quick fixes and an LLM-driven correction path. The authors did **not** separately measure detection accuracy. — [alphaxiv 2503.14723](https://www.alphaxiv.org/abs/2503.14723); [arXiv 2509.15971](https://arxiv.org/html/2509.15971)
- Abstract interpretation can *prove absence* of leakage by tracking whether train/test data come from disjoint, untainted sources. It is implemented in Microsoft's NBLyzer (Science of Computer Programming, 2025). — [ScienceDirect S0167642325000772](https://www.sciencedirect.com/science/article/abs/pii/S0167642325000772)
- Static slicing combined with structured LLM prompts (ACM TOSEM-style, 2025/26): its authors argue pure static analysis misses leakage when an API is not explicitly modeled. — [ACM DL 10.1145/3808199](https://dl.acm.org/doi/pdf/10.1145/3808199)
- "A Grammar of ML Workflows: Rejecting Data Leakage at Call Time" (arXiv 2603.10742, 2026) argues for *prevention* by making leakage inexpressible in the API, since detection finds errors only after the fact. — [arXiv 2603.10742](https://arxiv.org/pdf/2603.10742)
- scicode-lint (arXiv 2603.17893, Mar 2026) is an LLM-generated-pattern linter for methodology bugs in scientific Python. Its survey reports **no existing tool checks temporal or target-encoding leakage**. — [arXiv 2603.17893](https://arxiv.org/pdf/2603.17893) (snippet-level only; full text not fetched)
- Kapoor & Narayanan (Patterns 2023) give an 8-type leakage taxonomy that includes temporal leakage and illegitimate features, plus "model info sheets". Secondary sources disagree on paper counts (294 vs 329 papers across 17 fields). — [reproducible.cs.princeton.edu](https://reproducible.cs.princeton.edu/); [arXiv 2207.07048 (prior knowledge)](https://arxiv.org/abs/2207.07048)
- "Which Leakage Types Matter?" (Roth, arXiv 2604.04199, Apr 2026) defines a 4-class causal taxonomy and tests it empirically on 2,047 OpenML/PMLB binary classification datasets. Selection leakage matters most at practical sizes and normalization leakage least. It is a companion to 2603.10742, with v1.1 on Zenodo (May 2026). — [arXiv 2604.04199](https://arxiv.org/abs/2604.04199); [emergentmind](https://www.emergentmind.com/papers/2604.04199); [Zenodo 20451150](https://zenodo.org/records/20451150)
- Note: the 2,047 datasets are i.i.d. tabular, so the study does not cover temporal/PIT leakage. That is an inference from the dataset sources; I did not read the full text.
- "Look-Ahead-Freedom as Temporal Non-Interference" (arXiv 2607.04958, 2026) proposes a verifiable correctness property for backtesting and agentic trading pipelines. It covers windowing, resampling, joins, PIT/vintage reads and agentic retrieval for value-independent cases, and states that **full look-ahead-freedom is undecidable when availability depends on data values**. — [arXiv 2607.04958](https://arxiv.org/html/2607.04958v1)
- Look-Ahead-Bench (arXiv 2601.13770) is a standardized benchmark of look-ahead bias in PIT LLMs for finance, scored by performance decay across market regimes. — [arXiv 2601.13770](https://arxiv.org/abs/2601.13770)
- "Temporal Leakage in LLM Backtesting" (arXiv 2608.02985) finds that leakage-detection tests themselves can be invalid. On the official ForecastBench panel, 4 of 5 flagship models "failed" the check with spurious gaps of up to 0.061, so clean reference data must come from outside the benchmark. — [arXiv 2608.02985](https://arxiv.org/html/2608.02985)
- "Does Training on Future Data Pay?" (arXiv 2609.20554) separates *input-time* from *parameter-time* integrity for pretrained forecasters. Excluding the target market does not remove parameter-time contamination. — [arXiv 2609.20554](https://arxiv.org/html/2609.20554)
- ArrivalBench (arXiv 2610.02363, NeurIPS 2026, 40 tasks) re-runs agent-built pipelines under replayable delivery schedules (late, duplicated, out-of-order and retried records). Single-snapshot grading certifies 86–100% of pipelines, but adversarial replay drops this to **7.0–79.2%** across 11 models. Pipelines "repaired" against the snapshot test fail replay about as often as first-pass ones. — [NeurIPS 2026 listing](https://neurips.cc/virtual/2026/173048); [arXiv 2610.02363](https://arxiv.org/pdf/2610.02363)

### Inferences
**Problem cards**
- **P1.1 Temporal/PIT leakage detection in real feature pipelines (not just notebooks).**
  - *Why open:* Static tools model only sklearn-style split/fit/transform leakage. Temporal leakage depends on event time vs. availability time, which code alone doesn't reveal, and the general case is undecidable (2607.04958).
  - *SOTA:* Yang et al. analysis, LeakageDetector 1/2, NBLyzer abstract interpretation, slicing+LLM, and scicode-lint, which flags this as uncovered.
  - *Solo:* Build a hybrid checker that combines static lineage (which columns derive from which timestamps) with **dynamic differential testing**. Re-materialize features at as-of time t using only data with `available_at <= t`, then diff against production features. A mismatch is a witness of leakage.
  - *Compute:* CPU only.
  - *Metric:* Precision/recall on a seeded-bug benchmark (see P1.2) and on real historical bugs.
- **P1.2 A public benchmark of temporal leakage bugs in realistic pipelines.**
  - *Why open:* Existing leakage evaluations use Kaggle notebooks with i.i.d. leakage types. Roth's 2,047-dataset study is i.i.d. tabular. Look-Ahead-Bench targets LLM pretraining contamination rather than pipeline code. No pipeline-level temporal leakage benchmark turned up.
  - *Solo:* Use mutation testing on a real PIT pipeline. Inject leakage operators (as-of join changed to plain join, `available_at` changed to `event_at`, centered rolling windows, target encoding fit on the full history, post-match stats used pre-match, late-arriving corrections back-filled, entity-resolution merges that use future information). Label each mutant with ground truth and measured metric inflation.
  - *Compute:* CPU.
  - *Metric:* Adoption, detector recall per mutation class, and the correlation between injected leakage and the backtest skill inflation it causes.
- **P1.3 Leakage "magnitude" estimation:** Predict how much a given leak inflates offline metrics without fixing it, extending Roth's "which types matter" from i.i.d. to temporal data. Solo work is a sweep over mutation types × lag sizes × model families. CPU/1 GPU. The metric is the predictive R² of an inflation model.
- **P1.4 Validating leakage tests themselves:** 2608.02985 shows detection tests can give false positives without outside clean references. An open question is how to certify that a CI-blocking leakage test has power and isn't vacuous. A solo contributor could apply test-suite mutation scoring (does the suite kill injected leak mutants?) to leakage CI. CPU. The metric is mutation kill rate.
- **P1.5 Agent pipelines under time:** ArrivalBench shows agents' pipelines break under late or out-of-order data. A replay-based PIT-correctness harness is an evaluation primitive that most data-agent benchmarks lack (see §4).

### Gaps
- No Kaggle-specific leakage audit or benchmark (2024–2026) was found. Kaggle "leak" exploitation in competitions remains anecdotal in my results.
- I found no peer-reviewed paper on automatically verifying feature-store as-of join correctness (Feast/Tecton/Databricks). Only formal-property work (2607.04958) turned up.
- No Kapoor–Narayanan follow-up by the original authors (2025–2026) was found.
- Full texts of 2603.17893, 2607.04958 and 2610.02363 weren't read because arxiv.org was blocked, so their exact coverage claims are taken from abstracts.

---

## 2. Data quality and validation (learned validation, contracts, label errors, drift, valuation)

### Takeaway
Label-error detection has mature baselines (Confident Learning/cleanlab) and 2025 decomposition benchmarks, but no clear successor. Drift detection is well benchmarked as *detection*, yet the field still lacks evidence that drift alarms *predict degradation*; false alarm rates are themselves a 2026 research topic. Data valuation still hits scalability, violated-assumption and temporal-dynamics limits. Schema/contract inference turned up no strong 2025–2026 primary literature.

### Cited Findings
- Confident Learning (Northcutt et al., JAIR 2021) and its cleanlab implementation remain the standard baseline. A 2026 tabular paper still calls it "a widely adopted baseline". — [arXiv 1911.00068 (prior knowledge)](https://arxiv.org/abs/1911.00068); [Adaptive Ensemble Selection for Noisy Labels on Tabular Data, arXiv 2609.32976 (ID unverified)](https://arxiv.org/pdf/2609.32976)
- Pickler et al. (arXiv 2510.16211, Oct 2025) decompose label-noise detectors into an agreement function, aggregation, and in-sample vs. out-of-sample information gathering. On vision and tabular data, in-sample methods tend to outperform out-of-sample ones. — [arXiv 2510.16211](https://www.arxiv.org/abs/2510.16211)
- A Springer AI Review survey compares 34 tabular noisy-label filtering methods and deep-benchmarks 20 of them, on performance and runtime. — [Springer s10462-025-11293-9](https://link.springer.com/article/10.1007/s10462-025-11293-9)
- AQuA (NeurIPS 2023 D&B) is an older label-quality benchmarking tool and is image-heavy. — [arXiv 2306.09467](https://arxiv.org/pdf/2306.09467)
- Cross-modal error detection with tables and images (arXiv 2510.12383) extends error detection beyond single-modality. — [arXiv 2510.12383](https://arxiv.org/pdf/2510.12383)
- Concept-drift benchmarking framework (arXiv 2606.07789, listed KDD '26): 14 detectors × 7 real datasets × 4 drift types. Detectors do better on feature-space drift than on label-based changes, and SEED/STEPD have the highest medians but degrade under gradual drift. — [arXiv 2606.07789](https://arxiv.org/pdf/2606.07789)
- "When Drift Detectors Cry Wolf" (arXiv 2607.17336) studies false alarm rates in continuous monitoring, where alarms cause unnecessary retraining. — [arXiv 2607.17336](https://arxiv.org/pdf/2607.17336)
- RFD-based concept drift detection (EDBT 2025) uses relaxed functional dependencies, a DB-theory angle on drift. — [OpenProceedings EDBT 2025](https://openproceedings.org/2025/conf/edbt/paper-246.pdf)
- Label-free performance estimation (CBPE, DLE, as in NannyML) is the industry answer to delayed labels. Sources here are vendor/guide-level. — [labelyourdata guide](https://labelyourdata.com/articles/machine-learning/data-drift); [Conduktor glossary](https://www.conduktor.io/glossary/model-drift-in-streaming)
- Data valuation: Data Shapley in One Training Run (arXiv 2406.11011) and Data Banzhaf (arXiv 2205.15466) target cost and robustness. LLM-scale influence (arXiv 2405.13954) notes that Shapley requires many retrainings and that influence functions struggle with high-dimensional gradients and large-gradient-norm outliers. — [2406.11011](https://arxiv.org/pdf/2406.11011); [2205.15466](https://arxiv.org/pdf/2205.15466); [alphaxiv 2405.13954](https://www.alphaxiv.org/abs/2405.13954)
- Rescaled Influence Functions (arXiv 2506.06656) target accuracy in high dimension. Versatile IF for non-decomposable loss (arXiv 2412.01335). Efficiently verifiable proofs of data attribution (arXiv 2508.10866). — [2506.06656](https://arxiv.org/pdf/2506.06656); [2412.01335](https://arxiv.org/pdf/2412.01335); [2508.10866](https://arxiv.org/pdf/2508.10866)
- "Challenges in Enabling Private Data Valuation" (arXiv 2603.00342): under DP, the failure of influence-based valuation is *structural*. Open problems include decoupling individual records from global dataset geometry and valuation objectives with bounded sensitivity. — [arXiv 2603.00342](https://arxiv.org/html/2603.00342)
- A survey-style abstract surfaced in search lists five persistent valuation limitations: scalability, violated assumptions, **temporal dynamics**, inter-data interactions, and approximation accuracy. — via search results around [arXiv 2405.13954](https://www.alphaxiv.org/abs/2405.13954) (attribution to a specific survey not verified)
- Datamodels (Ilyas et al. 2022) and TRAK (Park et al. 2023) are standard attribution baselines. — [arXiv 2202.00622 (prior knowledge)](https://arxiv.org/abs/2202.00622); [arXiv 2303.14186 (prior knowledge)](https://arxiv.org/abs/2303.14186)

### Inferences
**Problem cards**
- **P2.1 Drift → degradation prediction:** Benchmarks score whether detectors *detect* synthetic drift (2606.07789), not whether alarms *forecast* metric loss, and false alarms are costly (2607.17336).
  - *Solo:* Build a benchmark of real temporal datasets (sports seasons are ideal: rule changes, promotions/relegations, transfer windows, COVID empty stadiums). Ground truth is the realized degradation of a frozen model. Score detectors and label-free estimators (CBPE/DLE) by AUROC for "degradation > δ in the next k periods" and by calibration of the estimated metric.
  - *Compute:* CPU.
- **P2.2 Label-error detection under temporal and non-i.i.d. structure:** Current benchmarks (Pickler 2025, Springer survey) assume i.i.d. tabular data. Errors in time-stamped or event data (mis-recorded scores, a goal credited to the wrong player, late corrections) are under-studied. Solo work: inject realistic noise processes (source-specific, correlated in time) and test whether in-sample vs. out-of-sample findings hold. CPU. The metric is precision@k against real correction logs.
- **P2.3 Learning data contracts from history:** Infer pandera/Great Expectations-style constraints (ranges, FDs, cross-table invariants, temporal monotonicity) from data plus change history, then measure false-positive and false-negative rates on future batches. The RFD angle (EDBT 2025) suggests functional-dependency mining is a usable primitive. CPU. The metric is caught-real-incident rate vs. alert noise.
- **P2.4 Temporal data valuation:** Valuation ignores "temporal dynamics", a listed limitation, and the value of old seasons decays. Solo work: run TRAK/Data-Shapley-in-one-run-style valuation on GBMs or small nets with time-aware utility, and answer "which seasons or sources help next-season forecasts". 1 GPU or CPU. The metric is the retrain-with-top/bottom-k-removed delta (datamodels-style counterfactual check).

### Gaps
- No 2025–2026 primary paper on *learned* schema/contract inference (e.g., LLM-inferred data contracts evaluated on real incidents) was found. This may be a genuine gap or a search miss.
- No leaderboard comparing cleanlab successors on a common real-noise tabular benchmark was found beyond the Springer survey.
- No independent benchmark of CBPE/DLE accuracy under real temporal drift was found. Only vendor claims turned up.

---

## 3. Dataset curation, data selection, mixture optimization, synthetic data evaluation

### Takeaway
Mixture optimization moved from static (DoReMi, RegMix) to dynamic and proxy-free methods in 2026 (RegMix-D, DeMix via model merging, FastMix at ~1.3 GPU-hours). The open issues are proxy reliability across scales, time-varying mixtures, and multimodal/post-training data. Synthetic tabular evaluation still lacks standard metrics: papers using different metrics reach "substantially disparate conclusions", and structural fidelity and membership privacy are unresolved.

### Cited Findings
- RegMix (arXiv 2407.01492, ICLR 2025) frames mixture selection as regression over small proxy runs. DoReMi (arXiv 2305.10429) uses group-DRO proxy reweighting. — [arXiv 2407.01492](https://arxiv.org/abs/2407.01492); [DoReMi notes](https://hunterheidenreich.com/notes/natural-language-processing/language-models/doremi-data-mixture-optimization/)
- RegMix-D (arXiv 2606.18663): challenges the "single optimal mixture" assumption shared by RegMix and DoReMi. It proposes a dynamic mix with offline and online variants and reports gains on 13 tasks (1B model, 25B Pile tokens). Results are self-reported. — [arXiv 2606.18663](https://arxiv.org/pdf/2606.18663)
- DeMix (arXiv 2602.00747): existing approaches either rely on unreliable tiny-scale proxies or need prohibitively expensive exploration. It decouples search from training via weighted model merging. — [arXiv 2602.00747](https://arxiv.org/html/2602.00747v2)
- FastMix (ICLR 2026, gradient-based) claims optimized mixtures with ~1.3 GPU-hours, best on 9 benchmarks. These are author figures. — [ICLR 2026 proceedings](https://proceedings.iclr.cc/paper_files/paper/2026/file/01d845baaa31b1043e5f280592d57dc4-Paper-Conference.pdf)
- Multimodal mixtures: DoReMi/RegMix assumptions are unimodal-specific, and a loss-regression alternative is proposed (Algorithms 2026). — [doi 10.3390/a19060482](https://doi.org/10.3390/a19060482)
- Survey: "Data Mixing for LLM Pretraining: A Survey and Outlook" (arXiv 2604.16380, 2026). — [arXiv 2604.16380](https://arxiv.org/pdf/2604.16380)
- Synthetic tabular data: TabStruct (rev. Mar 2026, arXiv 2509.11950) argues structural fidelity must be a core axis, testing 13 generators across 29 datasets. Du & Li (arXiv 2402.06806) find a top synthesizer has significant membership-privacy risk and that adding DP negates its advantage. A systematic review (arXiv 2504.18544) finds inconsistent metrics lead to disparate conclusions. A health framework finds DP lowers fidelity and utility. — [2509.11950](https://arxiv.org/html/2509.11950v2); [2402.06806](https://arxiv.org/html/2402.06806v3); [2504.18544](https://arxiv.org/pdf/2504.18544); [PMC12058740](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12058740/); [2504.01908](https://arxiv.org/html/2504.01908v1)

### Inferences
**Problem cards**
- **P3.1 Proxy-scale transfer of mixture rankings:** DeMix names proxy unreliability as the core difficulty. Solo work: run a careful study at 10M–150M parameters of *when* mixture rankings flip across scale and token budget, with a cheap predictor of flip risk. 1 GPU, weeks. The metric is the rank correlation of mixtures across scales.
- **P3.2 Data selection for tabular/time-series (non-LLM) models:** Mixture and selection work is LLM-centric. For forecasting, "which past seasons, leagues and sources to include, with what weights" is a mixture problem with temporal shift. Solo work: adapt RegMix-style regression to source/season weights for GBMs. CPU. The metric is out-of-time log-loss vs. uniform and recency baselines.
- **P3.3 Synthetic data evaluation for temporal relational data:** Metrics disagree (2504.18544), and structure matters (TabStruct). Sequential, multi-table, entity-linked synthetic data (matches × players × events) lacks validated metrics. Solo work: test whether "train on synthetic, test on real" utility rankings agree with structural and fidelity metrics on multi-table temporal data. CPU/1 GPU.
- Deduplication (SemDeDup and similar) is still relevant, but no new 2025–2026 open-problem statements were collected. — [SemDeDup arXiv 2303.09540 (prior knowledge)](https://arxiv.org/abs/2303.09540)

### Gaps
- RegMix-D's arXiv ID/date was flagged by the search summarizer as possibly inconsistent and should be verified.
- No consensus "open problems" list for data selection in 2026 was found. Themes were inferred from abstracts.
- Data pruning (e.g., scaling-law-beating pruning) status in 2026 was not researched due to the tool budget.

---

## 4. LLM agents for data engineering and data science

### Takeaway
Text-to-SQL leaderboards are no longer trustworthy as-is. A 2026 audit found **52.8% (BIRD Mini-Dev) and 62.8% (Spider 2.0-Snow) gold-annotation error rates**; the CIDR version says 66.1% for Spider 2.0-Snow. Errors misrank agents, while reported Spider 2.0-Snow scores went from 23.8% to 96.7%. Data-pipeline agents look good on single-snapshot grading but fail under replayed late, out-of-order data (ArrivalBench). DS agents still score low on broad benchmarks (o1-mini at 29.77% on DataSciBench). Benchmark proliferation in 2026 (AgenticDataBench, CODA-Bench, FDABench, Argo-Bench, DSAEval, ELT-Bench-Verified) has outpaced benchmark *validity* work.

### Cited Findings
- **Annotation-error audit** (UIUC; arXiv 2601.08778, PVLDB 2026): error rates of 52.8% (BIRD Mini-Dev) and 62.8% (Spider 2.0-Snow), found with SAR-Agent plus human SQL experts. Spider 2.0-Snow has 547 examples, with gold SQL public for only 121. The errors cause "severe misestimation and misranking". The CIDR 2026 version reports 66.1% for Spider 2.0-Snow, so cite carefully. — [arXiv 2601.08778](https://arxiv.org/html/2601.08778v3); [CIDR 2026 p5](https://vldb.org/cidrdb/papers/2026/p5-jin.pdf); [PVLDB vol19 p931](https://www.vldb.org/pvldb/vol19/p931-jin.pdf)
- Reported best Spider 2.0-Snow score rose from 23.8% at release to 96.7% (per a 2026 paper's summary; the source paper wasn't fetched). — via search summary citing [arXiv 2601.08778](https://arxiv.org/abs/2601.08778) and related work
- Example gold error: swapped longitude/latitude in ST_POINT. — [arXiv 2601.08778](https://arxiv.org/html/2601.08778v2)
- GBV-SQL (ACL 2026) provides a typology of "Gold Errors" across SQL, NL questions and databases. SpotIt (arXiv 2510.26840) evaluates text-to-SQL evaluation with formal verification and finds many problematic gold SQLs. — [ACL 2026.acl-long.837](https://aclanthology.org/2026.acl-long.837.pdf); [arXiv 2510.26840](https://arxiv.org/pdf/2510.26840)
- Spider 2.0 (ICLR 2025, enterprise warehouses: Snowflake, BigQuery) and BIRD are the canonical text-to-SQL suites. — [Spider 2.0 arXiv 2411.07763 (prior knowledge)](https://arxiv.org/abs/2411.07763); [BIRD arXiv 2305.03111 (prior knowledge)](https://arxiv.org/abs/2305.03111)
- **ELT-Bench** (PVLDB vol 19; arXiv 2504.04808): 100 end-to-end ELT pipelines. The best agent was Spider-Agent with Claude-3.7-Sonnet. Spider-Agent with GPT-4o averaged ~43.7 steps and $2.03 per task. A 2026 "ELT-Bench-Verified" argues that benchmark-quality issues *underestimate* agents. — [PVLDB p84](https://www.vldb.org/pvldb/vol19/p84-jin.pdf); [arXiv 2504.04808](https://arxiv.org/pdf/2504.04808)
- kRAIG (arXiv 2603.20311): on 8 ELT-Bench tasks, existing agents show limited ELT performance. — [arXiv 2603.20311](https://arxiv.org/pdf/2603.20311)
- **ArrivalBench** (NeurIPS 2026): single-run grading certifies 86–100% of pipelines, while replay under adversarial schedules certifies 7.0–79.2%. Snapshot-based repair doesn't fix replay failures. — [NeurIPS](https://neurips.cc/virtual/2026/173048)
- KramaBench (ICLR 2026): data-to-insight pipelines over data lakes, described as having 104 tasks. No 2026 scores were found. — [arXiv 2506.06541 (prior knowledge)](https://arxiv.org/abs/2506.06541)
- Other 2025–2026 data-agent benchmarks: FDABench (heterogeneous analytical queries, arXiv 2509.02473), AgenticDataBench (arXiv 2607.01647), CODA-Bench (arXiv 2606.15300), Argo-Bench (enterprise-scale workflows, arXiv 2610.02122), "Can AI Agents Answer Your Data Questions?" (arXiv 2603.20576), and DSAEval (arXiv 2601.13591). Their contents were not verified beyond titles. — [2509.02473](https://arxiv.org/pdf/2509.02473); [2607.01647](https://arxiv.org/pdf/2607.01647); [2606.15300](https://arxiv.org/pdf/2606.15300); [2610.02122](https://arxiv.org/html/2610.02122v1); [2603.20576](https://arxiv.org/pdf/2603.20576); [2601.13591](https://arxiv.org/pdf/2601.13591)
- DS benchmark sizes (per an ICLR 2026 comparison table): DataSciBench 222 tasks, DSBench 540 (466 analysis + 74 modeling, from ModelOff/Kaggle), DA-Code 500, MLE-bench 75. On DataSciBench, o1-mini scored 29.77% overall. DataSciBench scoring mixes LLM judging, human judgment and programmed rules. — [ICLR 2026 paper](https://proceedings.iclr.cc/paper_files/paper/2026/file/f72d4fdfd5eb425cd81df9fe6272a533-Paper-Conference.pdf); [DataSciBench, ACL Findings 2026](https://aclanthology.org/2026.findings-acl.181.pdf); [DA-Code arXiv 2410.07331](https://arxiv.org/pdf/2410.07331)
- A multi-agent blackboard system on DA-Code reports 13–57% relative end-to-end gains and up to 9% data-discovery F1 gain. — [arXiv 2510.01285](https://arxiv.org/pdf/2510.01285)
- MLE-STAR (arXiv 2506.15692) is a Kaggle-style MLE agent. GRACE-DS (arXiv 2606.16000) is a reward-guided correction environment. FML-bench covers ML research agents. — [2506.15692](https://arxiv.org/pdf/2506.15692); [2606.16000](https://arxiv.org/pdf/2606.16000); [2510.10472](https://arxiv.org/pdf/2510.10472)
- MLE-bench (OpenAI, 2024; 75 Kaggle competitions). — [arXiv 2410.07095 (prior knowledge)](https://arxiv.org/abs/2410.07095)

### Inferences
**Problem cards**
- **P4.1 Benchmark validity for data agents:** Text-to-SQL gold errors exceed 50%. Solo work: build automated gold-query auditing via differential execution across perturbed DB instances (in the spirit of SpotIt's formal verification), and release corrected subsets. CPU plus API budget. The metric is agreement with expert adjudication and ranking stability (Kendall τ) before vs. after correction.
- **P4.2 Temporal robustness of agent-built pipelines:** ArrivalBench covers only 40 synthetic tasks. Solo work: extend replay testing to dbt/SQL/medallion pipelines over real PIT data (bronze to gold), with as-of correctness as the oracle. CPU plus API. The metric is replay pass rate vs. snapshot pass rate.
- **P4.3 Do DS/MLE agents leak?** No benchmark found scores agents on *methodological validity*, such as whether an agent's solution leaks. Kaggle-derived suites (MLE-bench, DSBench) reward leaderboard score, and a leaking solution can score higher. Solo work: run a leakage linter (P1.1) over agent trajectories on MLE-bench/DSBench and report the leak rate and the score inflation it buys. CPU plus API. This is a high-novelty, low-compute project.
- **P4.4 Semantic layer/catalog grounding:** Enterprise text-to-SQL failures concentrate on schema/metric semantics, though this was not directly evidenced in fetched sources. A solo contributor could test whether supplying dbt semantic-layer or metric definitions closes the gap on Spider 2.0-style tasks, but must use audited gold.
- **P4.5 Data-cleaning agents with verifiable oracles:** Pair agents with pandera contracts plus ER ground truth so cleaning can be scored by contract-satisfaction and downstream utility rather than by an LLM judge (which DataSciBench partly relies on).

### Gaps
- No consolidated 2026 leaderboard covering MLE-bench, DSBench, DataSciBench and DA-Code was found, and no cross-benchmark failure taxonomy either.
- No current Spider 2.0/BIRD leaderboard snapshot was fetched. The 96.7% figure comes from a search summary and its primary source is unverified.
- ELT-Bench-Verified's numbers and authors were not retrieved.
- Industry blog statements (Databricks, Snowflake, Netflix) on unsolved data-agent problems were not collected due to the tool budget.

---

## 5. Lakehouse/table formats meet ML (versioning, time travel, lineage, reproducible snapshots)

### Takeaway
Peer-reviewed research here is thin. Practice relies on pinning training runs to Iceberg/Delta snapshot IDs. A 2023 CIDR paper says lakehouse storage design tradeoffs are "not well studied". No 2025–2026 paper was found that standardizes the link between training-data versions, feature stores and model lineage across table formats. This is open, mostly systems-engineering research that is tractable on a laptop.

### Cited Findings
- The Lakehouse concept (CIDR 2021) puts a metadata layer (e.g., Delta) over open files to get versioning and time travel. — [Armbrust et al. CIDR 2021](https://people.eecs.berkeley.edu/~matei/papers/2021/cidr_lakehouse.pdf)
- "Analyzing and Comparing Lakehouse Storage Systems" (CIDR 2023): the tradeoffs among Delta, Hudi and Iceberg are not well studied. — [CIDR 2023 p92](https://www.cidrdb.org/cidr2023/papers/p92-jain.pdf)
- A 2025 comparative study of Delta, Iceberg and Hudi looks at training reproducibility and feature-store creation. — [arXiv 2508.13396](https://arxiv.org/pdf/2508.13396)
- Model Lake (arXiv 2503.22754) proposes combined data, model and code lineage per registered model. It is a design proposal and does not address lakehouse tables directly. — [arXiv 2503.22754](https://arxiv.org/pdf/2503.22754)
- Practitioner guidance trains against snapshot IDs so later arrivals don't change what a model saw. These are practice guides, not research. — [datafield.dev ch.25](https://datafield.dev/advanced-data-science/part-05/chapter-25/); [Medium: data versioning](https://medium.com/@okanyenigun/data-versioning-in-machine-learning-and-data-lakes-aa5ac3b6c33b)
- Late-arriving data strategies (update, drop or delay; reprocess a date partition) are standard practice. — [lakeFS blog](https://lakefs.io/blog/3-strategies-for-late-arriving-data/)

### Inferences
**Problem cards**
- **P5.1 Bitemporal correctness vs. snapshot time travel:** Table-format time travel gives *system/transaction time*. Forecasting needs *valid/availability time*, i.e. bitemporal data. Snapshot pinning reproduces what the lake contained, not what was knowable at prediction time; late corrections break this. Solo work: formalize and test "PIT reproducibility" as a property over Iceberg/Delta plus an availability column, and build a checker. CPU. The metric is detected divergences on real late-correction histories.
- **P5.2 Lineage-aware leakage detection:** Column-level lineage (OpenLineage-style) plus timestamps could feed P1.1. An open question is whether lineage metadata alone suffices to flag temporal leaks, and at what false-positive rate.
- **P5.3 Snapshot-diff impact on models:** Given two snapshots, predict a retrained model's metric change without retraining, linking data diffs to data valuation (P2.4). CPU.

### Gaps
- No 2025–2026 VLDB/SIGMOD/CIDR/DEEM paper specifically on ML training-data reproducibility over open table formats was found. This may be a real gap or a search limitation (DEEM 2025/2026 proceedings were not searched).
- Industry engineering blogs (Netflix, Databricks, Tecton) on unsolved PIT/feature-store problems were not retrieved.

---

## 6. Which open problems the user's football PIT data lake could credibly address

The lake has bronze/silver/gold layers, pandera contracts, CI-blocking leakage tests and cross-source entity resolution.

### Takeaway
The strongest fit is a **public temporal-leakage benchmark built by mutating a real PIT pipeline**. It addresses the gap that existing tools don't check temporal or target-encoding leakage (2603.17893), it complements the i.i.d.-only empirical leakage study (2604.04199), and it supplies the "outside clean reference" that 2608.02985 says leakage tests need. Second-best fits are a sports entity-resolution benchmark (none found in the literature) and an ArrivalBench-style replay harness for data agents over real late-arriving football data.

### Cited Findings
- No existing tool checks temporal or target-encoding leakage. — [arXiv 2603.17893](https://arxiv.org/pdf/2603.17893)
- The empirical leakage-type study uses 2,047 OpenML/PMLB (i.i.d.) datasets. — [arXiv 2604.04199](https://arxiv.org/abs/2604.04199)
- Leakage detection tests need clean references from outside the benchmark. — [arXiv 2608.02985](https://arxiv.org/html/2608.02985)
- Agent pipelines fail under replayed late, out-of-order data. The benchmark has only 40 self-built tasks. — [NeurIPS 2026 ArrivalBench](https://neurips.cc/virtual/2026/173048)
- Recent ER benchmarks cover noisy web identities (MelonBench, EMNLP 2026, 665 items) and long-context mention resolution (IdentifyMe, NAACL 2025). No sports or athlete-record ER benchmark was found. — [MelonBench](https://aclanthology.org/2026.emnlp-main.660/); [IdentifyMe](https://www.cs.toronto.edu/~makarand/papers/NAACL2025_IdentifyMe.pdf)
- LLM ER open issues: self-explanations have weak faithfulness (arXiv 2606.01210), pairwise LLM calls are costly (graph refinement + label propagation, arXiv 2605.25814), and practitioner lessons come from a self-serve ER pipeline (arXiv 2607.26298). Understanding LM-based entity matching beyond scale is covered in arXiv 2607.24688. — [2606.01210](https://arxiv.org/html/2606.01210); [2605.25814](https://arxiv.org/pdf/2605.25814); [2607.26298](https://arxiv.org/pdf/2607.26298); [2607.24688](https://arxiv.org/pdf/2607.24688)
- A practitioner view holds that LLMs can make ER decisions but are weak on consistency, explainability, scale and cost. — [Tilores blog](https://tilores.io/content/can-llms-be-used-for-entity-resolution/)
- Parameter-time contamination: pretrained models used as forecasters can leak future results through their weights even when inputs are clean. This applies directly to LLM-based match prediction. — [arXiv 2609.20554](https://arxiv.org/html/2609.20554); [Look-Ahead-Bench 2601.13770](https://arxiv.org/abs/2601.13770)

### Inferences
Ranked contributions, all CPU-feasible plus a modest LLM API budget:
1. **"PITLeakBench" (P1.2 + P1.3 + P1.4):** Release a de-licensed/public-data version of the pipeline plus about 20 leakage mutation operators, ground-truth labels, and the measured skill inflation (log-loss/Brier/RPS) for each mutant. Evaluate LeakageDetector 2.0, scicode-lint, NBLyzer-style analysis and LLM reviewers on it.
   - *Success:* Detector recall per class, and the first published numbers on temporal-leak detectability.
   - *Venues:* DMLR, the NeurIPS D&B track, DEEM@SIGMOD, MSR/ICSE-SEIP.
2. **Mutation score for leakage CI:** Report what fraction of injected leaks the user's existing CI tests kill. This methodological contribution ("are your leakage tests powerful?") answers 2608.02985's concern for pipelines.
3. **Sports ER benchmark:** Cross-source player/team/match identity (transliteration, name changes, loanees, duplicate IDs, temporal validity of identities), with gold links taken from the user's resolution.
   - *Evaluate:* Classical ER (Magellan/Ditto-style), LLM matchers, and the faithfulness of LLM explanations (extending 2606.01210).
   - *Metric:* Pairwise/cluster F1, cost per 1k pairs, and *temporal* ER errors (merges that are only valid after a date, an under-studied leakage vector).
4. **Replay-robustness track for data agents (P4.2):** Give agents bronze-to-gold tasks on real football feeds with late corrections (e.g., goal reassignments, postponed fixtures). Grade with ArrivalBench-style replay plus pandera contracts plus PIT oracle. Report replay pass rate vs. snapshot pass rate.
5. **Do MLE/DS agents leak on temporal sports tasks? (P4.3):** Run agents on forecasting tasks over the lake and lint their code and features for look-ahead. Report the leak rate and the score inflation.
6. **Drift → degradation benchmark (P2.1):** Seasons and rule changes give natural drift with realized degradation as ground truth.
7. **LLM parameter-time contamination for match outcomes:** Compare LLM predictions on matches before vs. after the training cutoff, following Look-Ahead-Bench/2609.20554 methods, as a sports analogue to the finance work.

Caveats: licensing of football data (commercial providers) may block a full public release. Synthetic or perturbed variants, or only public-source layers, may be needed, and §3's synthetic-fidelity problems then apply. That is an inference; licensing terms were not researched.

### Gaps
- Football data licensing and redistribution constraints were not researched.
- Existence of any sports-specific leakage or ER benchmark: none was found, but only one sports-targeted ER search was run.
- Whether DMLR or DEEM 2026 calls explicitly solicit leakage benchmarks was not verified.
