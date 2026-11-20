# Open Research Problems in ML Evaluation Methodology and Efficient Training/Inference (state as of Nov 2026)

Research notes for the report writer. Network note: during this session, direct fetches of metr.org, openai.com and arxiv.org were blocked by the sandbox network policy. Some facts below therefore come from search-result snippets and abstracts, not full texts, and are marked as such. Items tagged **[prior knowledge, not re-verified this session]** are well-known references cited from background knowledge; their arXiv IDs should be spot-checked before publication.

---

## 1. Evaluation methodology: statistical rigor, contamination/saturation, live benchmarks, IRT/small-sample evaluation, LLM-as-judge, agent capability (METR), probabilistic outputs

### Takeaway
The basic statistics are settled: use clustered standard errors and paired tests (Miller 2024), and use Bayesian or other small-sample methods when benchmarks are small (Bowyer et al. 2025). What is still open is getting the field to adopt them, and handling the harder cases. Those cases include saturated suites (about half of 60 benchmarks studied in 2026 were highly saturated), judge-model measurement error under distribution shift, adaptive/IRT evaluation that stays stable across model generations, and agent "time horizons" that the current task suite can no longer measure for frontier models. Most of this is CPU-tractable work on public item-level eval logs.

### Cited Findings

**Statistical rigor / error bars**
- Miller, "Adding Error Bars to Evals" (arXiv:2411.00640, Nov 2024) recommends CLT standard errors, clustered SEs when questions come in related groups, and inference on question-level paired differences when comparing models. On DROP, clustered SEs can be more than 3x larger than naive SEs. — [arXiv 2411.00640](https://arxiv.org/abs/2411.00640v1)
- Bowyer, Aitchison & Ivanova (arXiv:2503.01747, ICML 2025 spotlight position paper) argue that CLT-based error bars fail on small, specialized benchmarks (common for agent/frontier evals) and "dramatically underestimate uncertainty." They also argue that Bayesian methods make clustered extensions easier. — [arXiv 2503.01747](https://pdf.arxiv.org/abs/2503.01747)
- An `errorbars` PyPI package implements cluster-robust (CR1 sandwich) SEs and reads lm-evaluation-harness and Inspect AI logs. This tooling exists, but its adoption is unclear. — [PyPI errorbars](https://pypi.org/project/errorbars/)
- Background: Madaan et al. 2024, "Quantifying Variance in Evaluation Benchmarks" (arXiv:2406.10229), measures seed and initialization variance in eval scores. — [arXiv 2406.10229](https://arxiv.org/abs/2406.10229) **[prior knowledge, not re-verified this session]**

**Saturation and contamination**
- "When AI Benchmarks Plateau: A Systematic Study of Benchmark Saturation" (arXiv:2602.16763, EvalEval, ICML 2026) defines saturation as the loss of reliable discriminative power among SOTA models. Its slides report that 29 of 60 benchmarks show high or very high saturation, that benchmark age and scale predict saturation, and that private test sets do not prevent it. — [arXiv 2602.16763](https://arxiv.org/html/2602.16763); [ICML 2026 slides](https://icml.cc/media/icml-2026/Slides/63311.pdf)
- "Benchmarking is Broken" (arXiv:2510.07575) argues that laissez-faire benchmarking is untenable and proposes PeerBench, a community-governed, proctored evaluation blueprint. — [arXiv 2510.07575](https://arxiv.org/html/2510.07575v2)
- A 2026 position paper (arXiv:2604.03244) argues that standardized item-level benchmark data should become default evaluation infrastructure, citing saturation and contamination as reasons aggregate scores are untrustworthy. — [arXiv 2604.03244](https://arxiv.org/html/2604.03244v1)
- LiveBench (ICLR 2025, arXiv:2406.19314) is "contamination-limited." Questions are added and updated monthly from recent math competitions, arXiv papers, news and datasets. — [arXiv 2406.19314](https://arxiv.org/abs/2406.19314v2); [GitHub](https://github.com/livebench/LiveBench)
- No 2026 study was found that measures residual contamination or saturation on LiveBench itself. — (search finding; see Gaps)

**IRT and small-sample efficient evaluation**
- tinyBenchmarks (Polo et al., arXiv:2402.14992, ICML 2024) shows that about 100 curated examples suffice to estimate MMLU performance accurately, using IRT. — [arXiv 2402.14992](https://arxiv.org/html/2402.14992v2)
- Anchor Points (Vivek et al., arXiv:2309.08638) selects a small set of representative items. — [arXiv 2309.08638](https://arxiv.org/abs/2309.08638) **[prior knowledge, not re-verified this session]**
- LEGO-IRT (arXiv:2510.04051, Oct 2025) is a unified framework for data-efficient evaluation. It supports binary and continuous metrics and multiple benchmarks jointly. — [arXiv 2510.04051](https://arxiv.org/html/2510.04051v1)
- ATLAS (arXiv:2511.04689, revised Feb 2026) does computerized adaptive testing, with Fisher-information-guided item selection, for LLMs. — [arXiv 2511.04689](https://arxiv.org/html/2511.04689v2)
- A May 2026 preprint (arXiv:2605.07046) notes that conventional IRT fitting is "computationally expensive and numerically unstable," which limits large-scale use. — [arXiv 2605.07046](https://arxiv.org/html/2605.07046)
- PSN-IRT (AAAI 2026, "Lost in Benchmarks?", arXiv:2505.15055) uses IRT to diagnose item quality (four item properties plus model ability). — [arXiv 2505.15055](https://arxiv.org/html/2505.15055v2); [AAAI](https://ojs.aaai.org/index.php/AAAI/article/view/40814/44775)
- A 2026 Stanford course project applies Rasch/2PL models to six benchmarks from the "Fantastic-Bugs" dataset. It treats the number of items needed for reliable rankings as an open question. — [Stanford PDF](https://aimslab.stanford.edu/cs321m-projects/Ramprasath2026.pdf)

**LLM-as-judge reliability**
- arXiv:2606.19544 (June 2026) evaluated 21 judge models. High test-retest agreement can coexist with severe position bias. Verbosity bias is much smaller than 2023-era reports suggested. The paper proposes a "Minimum Viable Validation Protocol." — [arXiv 2606.19544](https://arxiv.org/abs/2606.19544) (abstract/snippet only)
- arXiv:2605.06939 (May 2026) finds that bias-corrected estimators are not automatically trustworthy. Their reliability depends on judge quality (Youden's J) and on whether the calibration assumptions hold. Model comparisons fail when judge calibration differs across the compared models. — [arXiv 2605.06939](https://arxiv.org/html/2605.06939v1) (snippet)
- "How to Correctly Report LLM-as-a-Judge Evaluations" (arXiv:2511.21140, v3 Feb 2026) shows that existing correction methods become biased when test and calibration distributions differ. It proposes a sensitivity/specificity-based estimator that stays unbiased under shift. — [arXiv 2511.21140](https://arxiv.org/html/2511.21140v3)
- An AACL 2026 paper frames position bias as an accuracy ceiling that capability alone cannot exceed. It decomposes reliability loss into noise and position-bias components. — [ACL Anthology 2026.aacl-long.207](https://aclanthology.org/2026.aacl-long.207.pdf)
- "When the Judge Changes, So Does the Measurement" (arXiv:2607.08535) finds that swapping judges changes the measured construct and recommends judge audit trails. — [arXiv 2607.08535](https://arxiv.org/pdf/2607.08535)
- A March 2026 SoK covers LLM-as-judge security (adversarial manipulation of judges). — [arXiv 2603.29403](https://arxiv.org/html/2603.29403v1)
- A survey with a meta-evaluation benchmark is available as arXiv:2411.15594 (v6). — [arXiv 2411.15594](https://arxiv.org/pdf/2411.15594v6)
- Background: Prediction-Powered Inference (Angelopoulos et al., arXiv:2301.09633) is the general framework for combining a small human-labeled set with many model-labeled items while keeping valid CIs. — [arXiv 2301.09633](https://arxiv.org/abs/2301.09633) **[prior knowledge, not re-verified this session]**

**Agent capability: METR time horizons**
- The original METR paper, "Measuring AI Ability to Complete Long Tasks" (March 2025, arXiv:2503.14499), found that the 50% task-completion time horizon doubled about every 7 months over 2019–2025. — [METR blog](https://metr.org/blog/2025-03-19-measuring-ai-ability-to-complete-long-tasks/); arXiv ID [2503.14499](https://arxiv.org/abs/2503.14499) **[ID from prior knowledge]**
- Time Horizon 1.1 (Jan 29, 2026) uses more tasks and new infrastructure and drops flawed tasks. It indicates faster progress since 2024. — [METR TH1.1](https://metr.org/blog/2026-01-29-time-horizon-1-1/)
- A LessWrong post characterizes the post-2024 trend as about 10x per year. This is a secondary interpretation, not METR's own figure. — [LessWrong](https://www.lesswrong.com/posts/EYb2K9acKfyG2bome/metr-time-horizons-now-10x-year)
- METR's Jan 22, 2026 limitations note gives Claude Opus 4.5 a 50% horizon of about 4h49m with a 95% CI of 1h49m to 20h25m. The interval spans more than 10x. — [METR limitations note](https://metr.org/notes/2026-01-22-time-horizon-limitations/) (via search snippet)
- METR's Mar 20, 2026 note says results grow more sensitive to modelling choices as the suite saturates. A regularization fix lowered some recent models' 50% horizons by up to 20%. — [METR modelling-assumptions note](https://metr.org/notes/2026-03-20-impact-of-modelling-assumptions-on-time-horizon-results/)
- Commentary claims Opus 4.6's 50% horizon exceeds what the TH1.1 suite can measure accurately. — [LessWrong "You're gonna need a bigger boat"](https://www.lesswrong.com/posts/3SywPAjGQWCtQFafb/you-re-gonna-need-a-bigger-boat-benchmark-metr) (commentary)
- METR progress reports exist for Feb 2026 and June 2026. The June report mentions a first "Frontier Risk Report." — [Feb 2026 report](https://metr.org/february-2026-progress-report.pdf); [June 2026 report](https://metr.org/june-2026-progress-report.pdf)

### Inferences: concrete open problems tractable for a solo researcher

**E1. Error bars for saturated and small agentic benchmarks.**
- *Why open:* CLT intervals break down on small n and near-ceiling accuracy (Bowyer et al.). Saturation is widespread (29/60). METR's own CI spans more than 10x.
- *Solo work:* Build a calibrated-coverage benchmark of eval-uncertainty methods: naive CLT, clustered CR1, cluster bootstrap, Bayesian hierarchical Beta-Binomial, and IRT-based posterior. Resample public item-level logs (HELM, Open LLM Leaderboard, lm-eval-harness outputs) and measure empirical coverage of the nominal 95% CIs, model-ranking stability, and false-discovery rates for "model A > model B" claims. Then extend to the METR logistic time-horizon fit: a hierarchical Bayesian version with task-family random effects.
- *Compute:* CPU only.
- *Success metric:* empirical coverage close to nominal, lower ranking flip rate, and adoption as a PR to lm-eval-harness or Inspect.

**E2. Multiple-comparison control on leaderboards.**
- *Why open:* Leaderboards implicitly run thousands of pairwise comparisons. Rankings are reported without FDR control or rank CIs. I found no 2025–2026 paper establishing a standard; see Gaps.
- *Solo work:* Compute simultaneous rank confidence sets (e.g., Holm-corrected paired tests, or bootstrap rank intervals) for existing leaderboards. Report how many "SOTA" claims survive.
- *Compute:* CPU.
- *Success metric:* fraction of published SOTA claims not supported at FDR 0.05.

**E3. A saturation early-warning metric.**
- *Why open:* The saturation paper finds age and scale predict saturation, but no prospective, item-level signal is standard.
- *Solo work:* Use IRT discrimination and difficulty drift over model generations to forecast when a benchmark loses discriminative power. Validate on held-out benchmarks.
- *Compute:* CPU.
- *Success metric:* prediction lead time and accuracy on held-out benchmarks.

**E4. Stable, cheap IRT and adaptive testing across model generations.**
- *Why open:* IRT fitting is unstable and expensive (arXiv 2605.07046). tinyBenchmarks-style anchor sets may go stale as new model families shift the ability distribution.
- *Solo work:* Measure how fast tinyBenchmarks, anchor-point and ATLAS estimates degrade on models released after the item set was calibrated. Propose online or recalibrating item banks.
- *Compute:* CPU, plus inference on a few hundred items per model via API or small local models.
- *Success metric:* MAE of estimated versus full-benchmark accuracy on post-calibration models, at a fixed budget of around 100 items.

**E5. LLM-judge measurement-error correction under shift.**
- *Why open:* Corrections assume a fixed judge confusion matrix (arXiv 2511.21140, 2605.06939). These corrections fail when judge calibration differs across compared models, which is exactly the leaderboard setting.
- *Solo work:* Model-conditional judge calibration, i.e., PPI-style estimators with a per-model confusion matrix. Pair this with a small human-label budget allocation strategy (active labeling).
- *Compute:* CPU, plus modest API.
- *Success metric:* bias and CI coverage against full human labels on public datasets (MT-Bench human judgments, Chatbot Arena samples).

**E6. Validity of agent time-horizon metrics beyond suite saturation.**
- *Why open:* TH1.1 can no longer resolve frontier horizons (commentary plus METR notes). The human-baseline times and the logistic-fit assumptions drive the results.
- *Solo work:* Reanalyse METR's released data (if public) for sensitivity to task-family weighting, to the human-time measurement error model, and to the choice between the 50% and 80% horizon. Or build small "messy" long-horizon task families using open-source models as subjects.
- *Compute:* CPU for reanalysis. Agent runs are API-cost-bound.

**E7. Evaluating probabilistic outputs.**
- *Why open:* Most LLM evals score argmax correctness and ignore calibration. Proper scoring rules (log/Brier) and their decompositions are rarely reported. arXiv 2608.28482 shows that different strictly proper rules used as RL rewards yield different forecasters. — [arXiv 2608.28482](https://arxiv.org/pdf/2608.28482)
- *Solo work:* Add a Murphy decomposition of the Brier score (reliability, resolution, uncertainty) and CRPS/log-score reporting to standard harnesses. Study which proper rule gives the most statistically efficient model ranking per item.
- *Compute:* CPU.

### Gaps
- metr.org/time-horizons could not be fetched. Current (Nov 2026) top-model horizons and any TH2.0 suite plans are unverified.
- I found no 2025–2026 paper specifically on multiple-comparison or FDR control for LLM leaderboards. This may be a real gap, or a search miss.
- I found no 2026 measurement of residual contamination in LiveBench.
- The LLM-judge papers were read only at the abstract/snippet level.

---

## 2. Real-world future events (sports, elections, markets) as contamination-proof benchmarks; demand for a sports benchmark with betting-market baselines

### Takeaway
Live-event forecasting benchmarks are now an established genre: ForecastBench, Prophet Arena (ICLR 2026), Foresight Arena, and a burst of FIFA World Cup 2026 papers. A sports benchmark with closing-line baselines already has several 2026 prototypes. However, these are single-tournament and small-n, and the methodology is contested. Whether AI has reached "superforecaster parity" is disputed, mostly over baselines, question mix and statistical power. The gap a solo researcher could fill is a **continuous, multi-sport, statistically rigorous** benchmark with a de-vigged closing-line baseline, paired tests (Diebold–Mariano-style on Brier/log-loss differences), and pre-registered analysis. There is evident demand, but "the market is very hard to beat" is the expected headline, so the contribution is rigor and longevity rather than a new idea.

### Cited Findings
- ForecastBench (arXiv:2409.19839, v5) is a dynamic benchmark of AI forecasting on questions unresolved at submission time. — [arXiv 2409.19839](https://arxiv.org/html/2409.19839v5)
- FRI says AI "has likely reached parity with superforecasters" on ForecastBench. 17 submissions rank above superforecasters on the preliminary leaderboard, and an AI ranked above superforecasters on market questions for the first time. FRI concedes the 95% CIs overlap substantially ("more consistent with parity than outperformance") and plans a fresh superforecaster round in fall 2026. — [FRI Substack](https://forecastingresearch.substack.com/p/ai-models-have-likely-reached-parity)
- Good Judgment disputes the claim: superforecasters still lead overall and are "almost 50% more accurate" on market questions. It argues the preliminary board uses only dataset questions, where AI has a structural advantage, and that the superforecaster baseline is a frozen July 2024 snapshot. — [Good Judgment "Not So Fast"](https://goodjudgment.substack.com/p/not-so-fast); [What ForecastBench Doesn't Measure](https://goodjudgment.substack.com/p/what-forecastbench-doesnt-measure?open=false)
- A non-peer-reviewed EA Forum analysis of ForecastBench market questions (as of 11 Aug 2026) found that 442 of 527 forecasters scored worse than the market price they were shown, and none demonstrably beat it. — [EA Forum "The Copier Line"](https://forum.effectivealtruism.org/posts/8YuG9EuhyYqaNBt7F/the-copier-line-what-forecastbench-s-market-scores-actually)
- A synthesis of 11 analyses concludes that parity claims haven't been evaluated rigorously enough to rule out leakage or noise. — [LessWrong "AI forecasting in 2026"](https://www.lesswrong.com/posts/a82q6yd8zKpYk56cF/ai-forecasting-in-2026-what-11-analyses-say)
- Prophet Arena ("LLM-as-a-Prophet", arXiv:2510.17638, ICLR 2026, UChicago) draws live events (elections, sports, economics) from Kalshi and Polymarket and curates them by trade volume and liquidity. It scores accuracy, calibration and economic value. Reported bottlenecks include inaccurate event recall, misunderstanding of data sources, and slower information aggregation than markets. Press framing ("AI matches prediction markets") is stronger than the paper's. — [arXiv 2510.17638](https://arxiv.org/html/2510.17638v2); [ICLR 2026](https://iclr.cc/virtual/2026/10017463); [Decrypt](https://decrypt.co/336298/ai-now-matches-prediction-markets-in-forecasting-real-events-study-finds)
- Lu, "Evaluating LLMs on Real-World Forecasting Against Expert Forecasters" (arXiv:2507.04562 v3), scored 464 Metaculus questions. Frontier models ostensibly beat the human crowd but significantly underperform experts. — [arXiv 2507.04562](https://arxiv.org/html/2507.04562v3)
- Foresight Arena (arXiv:2605.00420) is an on-chain benchmark that evaluates agents under a proper scoring rule and separates predictive skill from trading performance. — [arXiv 2605.00420](https://arxiv.org/pdf/2605.00420)
- Sports-specific 2026 work (from search snippets; full texts not fetched):
  - **LLM-SoccerArena (arXiv:2607.24573)** uses de-margined closing odds as a baseline. The best model (Gemini) had Brier 0.497 versus 0.498 for closing odds, which is not significant. No individual model stays positive after the prespecified multiplicity correction. — [arXiv 2607.24573](https://arxiv.org/pdf/2607.24573)
  - **WC2026-Agents (arXiv:2607.17765)** covers 104 FIFA World Cup 2026 matches, four LLM agents plus the bookmaker (1X2 odds) as a fifth competitor. It scores calibration and realized betting profit and is contamination-free because all matches came after the models' cutoffs. — [arXiv 2607.17765](https://arxiv.org/pdf/2607.17765); [HF papers](https://huggingface.co/papers/2607.17765)
  - **AI World Cup 2026 (arXiv:2608.03416)** covers end-to-end tournament prediction. It notes that betting markets remain hard for LLMs to beat consistently, and lists external baselines as needed future work. — [arXiv 2608.03416](https://arxiv.org/pdf/2608.03416)
  - **WorldCupArena (arXiv:2607.18084)** offers fine-grained evaluation of LMs and deep-research agents on football forecasting. — [HF papers](https://huggingface.co/papers/2607.18084.md)
  - **KellyBench (arXiv:2604.27865)** is a sequential bankroll-betting environment on 2023/24 EPL market odds. It tests profit, not just accuracy. — [arXiv 2604.27865](https://arxiv.org/html/2604.27865)
  - A UConn honors thesis covers 19,739 NCAA basketball games, using stats plus LLM summaries with fixed-stake and Kelly simulations against bookmaker odds. — [UConn](https://digitalcommons.lib.uconn.edu/srhonors_theses/1091)
- "How Proper Scoring Rules Shape LLM Forecasting" (arXiv:2608.28482) compares log, Brier, spherical and two beta scores as GRPO rewards. Rules with the same population-level incentive yield different learned forecasters. — [arXiv 2608.28482](https://arxiv.org/pdf/2608.28482)
- Background: the Diebold–Mariano test (Diebold & Mariano 1995, *JBES*) is the standard test for equal predictive accuracy of two forecast series under a loss differential, with HAC variance for serially correlated losses. — **[prior knowledge, not re-verified this session]**. No LLM-forecasting paper using DM tests surfaced in search.

### Inferences: concrete open problems
**F1. A continuous multi-sport benchmark with a closing-line baseline (direct answer: yes, demand exists, but it must differentiate).**
- *State of play:* Existing efforts are mostly one-tournament (World Cup 2026, 104 matches) or historical (KellyBench, EPL 23/24). At n≈100, the Brier difference between a model and the closing line (0.497 vs 0.498) is far below detectable size.
- *Differentiators:* (a) a rolling, multi-league, multi-sport feed giving thousands of events per year; (b) de-vigged closing *and* opening lines, measuring whether models add information beyond the line available at forecast time; (c) pre-registered paired tests (DM/HAC on per-event log-loss differences, clustered by matchday or league); (d) power analysis reporting the minimum detectable Brier edge; (e) strict timestamping to enforce information sets; (f) both "forecast-only" and "line-conditioned" tracks, the latter asking whether a model can improve on the market price.
- *Compute:* CPU plus API calls. Odds data costs money or relies on scraping, and licensing is a real constraint.
- *Success metric:* adoption by labs or leaderboards, and published detectable-edge bounds.

**F2. Scoring forecasters before resolution.** Long-horizon questions (elections, policy) take months or years to resolve. Open approaches to evaluate rigorously include market-proxy scoring (scoring against later market prices), consistency/coherence checks (Prophet Arena reports a consistency benchmark), and recursive or intermediate-resolution questions. Their validity against eventual outcomes is not established.
- *Solo work:* Using resolved historical questions, measure how well "score against the market price at t+k" predicts the final proper score. This is CPU work.

**F3. Statistical power and multiplicity in forecasting leaderboards.** The ForecastBench parity dispute largely turns on overlapping CIs, question-type mix and frozen human baselines.
- *Solo work:* Reanalyse public ForecastBench data with question-clustered paired tests and question-type stratification. Report the n needed to separate the top entries.

**F4. Leakage auditing for "live" benchmarks.** Retrieval-enabled agents can find post-cutoff results, and search engines leak future-dated content. Sports have clean timestamps, which makes them a good testbed for leakage-detection methods (e.g., accuracy jumps when the information-set cutoff is moved).

### Gaps
- Full texts of the sports papers (2607.24573, 2607.17765, 2608.03416) were not accessible (arxiv.org blocked). Sample sizes, test choice and exact baselines come from snippets.
- I found no evidence of a *persistent, multi-sport, live* LLM sports leaderboard with a closing-line baseline as of Nov 2026. Prophet Arena includes sports via Kalshi but is multi-domain.
- Demand evidence is indirect: five or more independent 2026 papers built similar setups ad hoc, and none provides persistent infrastructure. I found no explicit survey of demand.

---

## 3. Efficient training/inference with modest compute: speedruns, optimizers, quantization, MoE, speculative decoding, KV cache, test-time compute, distillation; accessible leaderboards and current records

### Takeaway
The NanoGPT speedrun (124M model to 3.28 FineWeb validation loss on 8xH100) has fallen from 45 min (May 2024) to **0.665 min (record #92, Aug 30, 2026)**. It remains the most visible efficiency leaderboard, but each record needs 8xH100 timing. Ablations at reduced scale on one GPU are feasible, and official record attempts are cheap (minutes on rented 8xH100). OpenAI's Parameter Golf (16MB artifact, under 10 min on 8xH100, Mar–Apr 2026) drew 1,000+ participants. Optimizer claims have been sharply deflated: against tuned AdamW, matrix optimizers give at most about 1.4x at 0.1B and about 1.1x at 1.2B. Fully-FP4 training stability is still openly unsolved. These make good solo targets.

### Cited Findings

**NanoGPT speedrun (KellerJordan/modded-nanogpt)** — all from the [GitHub README](https://github.com/KellerJordan/modded-nanogpt)
- Rules: reach ≤3.28 validation cross-entropy on FineWeb using 8x H100 (official timing on PrimeIntellect 8xH100). The 3.28 target follows Karpathy's llm.c GPT-2 (124M) replication.
- Track 1 (124M) milestones: llm.c baseline 45 min (05/28/24); Muon introduced, 24.9 min (10/04/24); 5.03 min with FlexAttention (11/19/24); 2.992 min (01/16/25); FlashAttention-3, 2.731 min (09/03/25); NorMuon, 2.345 min (10/24/25); batch size schedule, 2.203 min (11/29/25); paired head attention, 1.820 min (01/07/26); bigram hash embedding, 1.655 min (01/19/26).
- Recent records: #84, 1.320 min, FP8 MLP up-projection (05/21/26); #87, 1.256 min (06/11/26); #89, 1.23 min, FP8 MLP down-projection with delayed scaling (07/17/26); #90, 1.13 min, 96-dim QK, fused Triton, more FP8 (08/03/26); #91, 1.126 min, masking logits for infeasible token continuations during validation (08/06/26).
- **Current record #92: 0.665 min (08/30/26), by @DevenPzak.** Ingredients: sampled softmax, full FP8 MLP forward and backward, an 84.6M-row hashed n-gram table, an "ANVIL optimizer stack," depth reduction, and full CUDA-graph capture. This is a ~41% cut in one record, and it leans heavily on a large hashed n-gram lookup table. Whether that changes the spirit of the 124M track is a natural question; I did not find discussion of it.
- Track 2 (GPT-2 Medium, 350M, target 2.92): llm.c baseline 5.8 h; 29.3 min (01/18/25); Snoo optimizer, 23.28 min (09/16/25); Smear-MTP, 22.99 min (11/02/25); latest 17.35 min (12/31/25, "bulk transfer of short-track features").
- Current technique stack: Muon with Polar Express orthogonalization, NorMuon and cautious weight decay; FP8 in the head, MLP and QKV; FA3, sliding windows, QK-norm, paired-head attention, learnable XSA; RoPE, ReLU², value embeddings, MUDD skip connections, bigram hash embeddings; CUDA graphs, sparse embedding updates.

**Parameter Golf (OpenAI, 2026)**
- Rules: train the best LM whose artifact (code bytes plus compressed model bytes) fits in 16MB and trains in under 10 min on 8xH100. Scored in bits per byte on FineWeb validation. Ran March 18 to April 30, 2026. — [GitHub mirror](https://github.com/sekinal/parameter-golf)
- Early leaderboard (~Mar 19, 2026): best 1.2014 bpb against a naive baseline of 1.2244. Final standings were not retrieved. — [GitHub mirror](https://github.com/sekinal/parameter-golf)
- OpenAI's retrospective (May 12, 2026) reports 1,000+ participants and 2,000+ submissions, and that the "vast majority" of submitters used coding agents. — [OpenAI "What Parameter Golf taught us"](https://openai.com/index/what-parameter-golf-taught-us/) (search snippet; page fetch blocked)
- "Parameter Golf: What Really Works?" (arXiv:2607.01517, July 2026) analyses the submissions. — [arXiv 2607.01517](https://arxiv.org/abs/2607.01517) (not read)

**Optimizers**
- Wen, Hall, Ma & Liang, "Fantastic Pretraining Optimizers and Where to Find Them" (arXiv:2509.02046, ICLR 2026), benchmarks optimizers against tuned AdamW:
  - Speedup never exceeds about 1.4x, and many claimed 2x gains come from weak baselines.
  - Speedup falls with scale, from about 1.3–1.4x at 0.1B to about 1.1x at 1.2B.
  - Matrix optimizers (Muon, SOAP, etc.) give about 1.3x under 520M.
  - Muon is best at 1x Chinchilla, SOAP overtakes it at 8x or more data, and at 16x Chinchilla Muon was beaten by Nesterov AdamW in one run set.
  - Missetting a single hyperparameter (LR at 2x optimal) erases the gains.
  - — [arXiv 2509.02046](https://arxiv.org/html/2509.02046v2); [ICLR 2026](https://iclr.cc/virtual/2026/poster/10011758); [Marin W&B report](https://wandb.ai/marin-community/marin/reports/Fantastic-Optimizers-and-Where-to-Find-Them--VmlldzoxMjgzMzQ2NQ)
- Background: SOAP (arXiv:2409.11321) and Moonshot's "Muon is Scalable for LLM Training" (arXiv:2502.16982). — [SOAP](https://arxiv.org/abs/2409.11321); [Muon scalable](https://arxiv.org/abs/2502.16982) **[prior knowledge, not re-verified this session]**
- Speedrun-born optimizer variants (NorMuon, Polar Express, Snoo, cautious WD, "ANVIL") have mostly not been validated at more than 1B scale against tuned baselines. — [modded-nanogpt README](https://github.com/KellerJordan/modded-nanogpt) (record list), combined with the Wen et al. findings

**Low-precision / FP4 training**
- NVIDIA's NVFP4 report (arXiv:2509.25149) trained a 12B model on 10T tokens with NVFP4, matching an FP8 baseline in loss and downstream accuracy. — [arXiv 2509.25149](https://arxiv.org/html/2509.25149v2)
- Quartet (arXiv:2505.14669) argues that native FP4 training can be compute-optimal. — [arXiv 2505.14669](https://arxiv.org/html/2505.14669v4)
- "FP4 All the Way" (arXiv:2505.19115) does fully quantized training, with stochastic rounding in backward/update and round-to-nearest in forward for stability. — [arXiv 2505.19115](https://arxiv.org/html/2505.19115v2)
- TetraJet-v2 (arXiv:2510.27527) names weight oscillation and outliers as the key obstacles. It states that "fully FP4 training methods are yet not comparable to high-precision training… the challenging optimization problems… remain unsolved." — [arXiv 2510.27527](https://arxiv.org/html/2510.27527v1)
- Full-Stack FP4 (arXiv:2607.04422, July 2026) notes that most NVFP4 work quantizes only linear layers, leaving optimizer states, optimizer arithmetic and attention underexplored. It reports a 1.47% loss gap versus BF16 for 3B/64B tokens. — [arXiv 2607.04422](https://arxiv.org/abs/2607.04422)
- A 2026 study on native MXFP4 hardware (Llama-3.1-8B, arXiv:2605.09825) asks whether weight-gradient quantization drives most of the degradation, and compares setups with and without Hadamard transforms. — [arXiv 2605.09825](https://arxiv.org/html/2605.09825v2)
- A HiFloat4 paper (arXiv:2604.08826) states that stable end-to-end FP4 training with minimal degradation "remains an open challenge." — [arXiv 2604.08826](https://arxiv.org/html/2604.08826)

**Speculative decoding, KV cache, MoE, distillation, test-time compute**
- Not searched in this session because of the tool budget. Anchor references from background: EAGLE-3 (arXiv:2503.01840) for speculative decoding; KIVI (arXiv:2402.02750) and SnapKV (arXiv:2404.14469) for KV-cache quantization and eviction. — [EAGLE-3](https://arxiv.org/abs/2503.01840); [KIVI](https://arxiv.org/abs/2402.02750); [SnapKV](https://arxiv.org/abs/2404.14469) **[prior knowledge, not re-verified this session; 2026 SOTA unknown]**

### Inferences: concrete open problems
**T1. Do speedrun tricks transfer?**
- *Why open:* About 90 records' worth of techniques were tuned on a single 124M/3.28 setup. Wen et al. show optimizer gains shrink with scale and tuning.
- *Solo work:* A systematic "transfer audit." Take the top 10–15 speedrun innovations (bigram/n-gram hash embeddings, paired-head attention, MUDD, XSA, NorMuon, batch-size schedules, prefix-token auxiliary loss) and ablate each against a tuned baseline at 2–3 scales and token budgets on one GPU, using smaller models plus scaling-law extrapolation.
- *Compute:* one 24–80GB GPU, weeks.
- *Success metric:* speedup multiplier versus scale and token-to-parameter ratio, with seeds and CIs. Seed variance in speedrun records is itself a statistical-rigor question; record claims usually rest on a handful of runs.

**T2. Statistical validity of speedrun records.**
- *Why open:* Records are often less than 1% gains (e.g., 1.266 to 1.256 min). The validation-loss target is a threshold on a noisy quantity, and validation-time tricks (#91's masking of infeasible continuations) blur train-efficiency versus eval-engineering.
- *Solo work:* Estimate run-to-run variance of time-to-target and propose a significance protocol. This links directly to evaluation rigor (Section 1).
- *Compute:* rented 8xH100 for a few hours, or one GPU at reduced scale.

**T3. n-gram/hash-table memory versus parameters.**
- *Why open:* Record #92's 84.6M-row hashed n-gram table, and the earlier bigram hash embedding, suggest cheap lookup memory substitutes for compute at small scale.
- *Solo work:* Characterize scaling, the point where the benefit vanishes, and parameter-count accounting (it relates to Parameter Golf's artifact-size rule).
- *Compute:* one GPU.

**T4. Optimizer state and attention in FP4.**
- *Why open:* Explicitly flagged as underexplored (arXiv 2607.04422). Oscillation and outliers remain unsolved (TetraJet-v2).
- *Solo work:* Small-scale (≤350M) studies of FP4 optimizer states for Muon/SOAP (matrix optimizers have different state statistics than Adam), using simulated quantization on any GPU. Native NVFP4/MXFP4 requires Blackwell-class hardware, so most solo work would use fake-quant simulation, and kernel-speed claims would not be testable.
- *Success metric:* loss gap versus BF16 at matched tokens, and divergence rate.

**T5. Optimizer benchmarking protocol.**
- *Why open:* Wen et al. show baseline tuning dominates.
- *Solo work:* Hyperparameter-transfer laws (LR and WD scaling) for Muon and its successors at 1–16x Chinchilla, at 30M–300M scale. Also test whether newer variants (NorMuon, Snoo, Polar Express) beat tuned AdamW at 8–16x data, where Muon reportedly loses.
- *Compute:* one GPU, months.

**T6. Accessible leaderboards for solo contributors** (as of Nov 2026, from sources above):
- modded-nanogpt Track 1: 0.665 min, record #92, Aug 30, 2026.
- Track 2 (Medium): 17.35 min, Dec 31, 2025. This is the least-attacked track, and an opportunity for porting the 2026 short-track features.
- Parameter Golf: closed April 30, 2026. Its analysis paper is a source of unexplored ideas.
- Other speedruns (e.g., CIFAR-10 airbench, NanoGPT CPU variants): not checked this session.

### Gaps
- Speculative decoding, KV-cache compression, small-scale MoE, distillation and test-time-compute efficiency: no 2026 searches were done. Current SOTA and open problems there are not covered by verified sources in these notes. The report writer should treat these as uncovered or rely on other researchers' notes.
- Parameter Golf final winning score and techniques were not retrieved (openai.com blocked; arXiv 2607.01517 not read).
- Whether record #92 (n-gram table, sampled softmax) was contested under track rules is unknown.
- Wen et al.'s "2.1 follow-up" was mentioned in search but not checked.

---

## 4. What 2025–2026 position papers, workshop calls and prominent researchers list as top open problems in evaluation and efficiency

### Takeaway
Recurring themes in the 2025–2026 position literature include:
- Error bars and small-sample uncertainty (Bowyer et al., ICML 2025 position).
- Saturation and the loss of discriminative power (EvalEval, ICML 2026).
- Governance and proctoring of benchmarks (PeerBench).
- Item-level data as default infrastructure (arXiv 2604.03244).
- Judge validity and audit trails.
- Measurement validity of agent time horizons as suites saturate (METR's own notes).

On efficiency, the stated open problems are fair optimizer baselines and scaling of speedups (Wen et al., ICLR 2026) and stable fully-FP4 training (TetraJet-v2, HiFloat4, Full-Stack FP4).

### Cited Findings
- Position: CLT error bars are inadequate for small benchmarks; use Bayesian methods (ICML 2025 spotlight). — [arXiv 2503.01747](https://pdf.arxiv.org/abs/2503.01747)
- Saturation is widespread (29/60), and private test sets are not a remedy (EvalEval, ICML 2026). — [arXiv 2602.16763](https://arxiv.org/html/2602.16763); [slides](https://icml.cc/media/icml-2026/Slides/63311.pdf)
- "Benchmarking is Broken": calls for community-governed proctored evaluation (PeerBench). — [arXiv 2510.07575](https://arxiv.org/html/2510.07575v2)
- Item-level benchmark data should be standard infrastructure. — [arXiv 2604.03244](https://arxiv.org/html/2604.03244v1)
- Judge evaluations need validation protocols and audit trails. — [arXiv 2606.19544](https://arxiv.org/abs/2606.19544); [arXiv 2607.08535](https://arxiv.org/pdf/2607.08535)
- METR's own limitations and modelling-assumptions notes (Jan and Mar 2026) flag suite saturation and analysis sensitivity as open problems. — [limitations](https://metr.org/notes/2026-01-22-time-horizon-limitations/); [modelling assumptions](https://metr.org/notes/2026-03-20-impact-of-modelling-assumptions-on-time-horizon-results/)
- Forecasting evaluation: the parity debate shows open problems in baselines (frozen human panels), question-type mix and leakage. — [FRI](https://forecastingresearch.substack.com/p/ai-models-have-likely-reached-parity); [Good Judgment](https://goodjudgment.substack.com/p/not-so-fast); [LessWrong synthesis](https://www.lesswrong.com/posts/a82q6yd8zKpYk56cF/ai-forecasting-in-2026-what-11-analyses-say)
- Efficiency: optimizer speedups are overstated relative to tuned baselines and shrink with scale. — [arXiv 2509.02046](https://arxiv.org/html/2509.02046v2)
- Efficiency: fully-FP4 training optimization "remains unsolved." — [TetraJet-v2](https://arxiv.org/html/2510.27527v1); [HiFloat4](https://arxiv.org/html/2604.08826); [Full-Stack FP4](https://arxiv.org/abs/2607.04422)

### Inferences
- The common thread between the two halves is **measurement**. Speedrun records, optimizer claims and FP4 "parity" claims suffer from the same small-n, weak-baseline and multiplicity problems as LLM benchmarks. A solo researcher with a statistics background has an unusually good angle on the efficiency side by contributing significance protocols for records and optimizer comparisons (T1, T2, T5).
- The highest-leverage, lowest-compute solo projects, ranked:
  1. Coverage-calibrated uncertainty methods for small or saturated benchmarks (E1/E2; CPU).
  2. A rigorous multi-sport, closing-line forecasting benchmark with pre-registered DM-style paired tests and power analysis (F1; CPU plus data cost).
  3. Model-conditional LLM-judge correction with valid CIs (E5; CPU plus small API).
  4. Transfer and variance audit of speedrun techniques (T1/T2; one GPU).
  5. FP4 optimizer-state study under fake quantization (T4; one GPU).
- Things that may already be resolved or are outdated:
  - The old 7-month METR doubling time has been superseded by TH1.1.
  - "Muon gives 2x" is superseded by Wen et al.'s ≤1.4x.
  - "Verbosity bias is a major judge flaw" is reportedly much smaller in 2026 judges (arXiv 2606.19544).
  - NVFP4 at 12B/10T matching FP8 suggests "FP4 training is impossible" is outdated. The remaining open problem is *fully* FP4 (optimizer, attention, all GEMMs) at BF16 parity.

### Gaps
- I did not retrieve 2026 workshop calls for papers (NeurIPS 2026 Datasets & Benchmarks, ICML 2026 eval workshops, Efficient-ML workshops) or Epoch AI, Anthropic or Google eval blog posts listing open problems. That would need further searching.
- I found no explicit open-problems list from prominent individual researchers (e.g., a 2026 "open problems in evals" essay). This is a search gap, not evidence of absence.
