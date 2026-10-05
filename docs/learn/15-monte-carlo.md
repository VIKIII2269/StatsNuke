# 15 · Monte Carlo: the player simulator

> **Goal:** understand how StatsNuke turns a dozen component models into a full points distribution for every player, by playing each match thousands of times. You will learn Monte Carlo error, vectorised simulation, exact-inclusion lineup sampling, common random numbers, and how the simulator is validated.

## 1. Why simulate at all

Could we just add up expected values? Expected points *are* a sum of expected components (linearity, Module 02). But almost every FPL rule is a non-linear function of several random events:

- **Clean sheet:** 60+ minutes *and* 0 conceded *while on the pitch*. That depends jointly on the lineup, substitution time and goal times.
- **Bonus:** a ranking among *all* players in the match.
- **Thresholds:** defensive actions ≥ 10, saves in multiples of 3.
- **Correlation:** a team goal gives one player a goal and another an assist, and costs the opposing defenders their clean sheet.

Also, the optimiser and the risk measures want **whole distributions**: P(haul), covariance between teammates. There is no closed form for any of this, so we sample.

## 2. Monte Carlo error

Simulate S independent matches and average:

$$\hat{\mathbb E}[\text{pts}] = \frac1S\sum_{s=1}^S \text{pts}_s, \qquad \text{SE} = \frac{s}{\sqrt S}$$

$$\hat P(\text{pts}\ge k) = \frac{\#\{\text{pts}_s \ge k\}}{S},\qquad \text{SE} = \sqrt{\frac{\hat p(1-\hat p)}{S}}$$

ARCHITECTURE §8.3 works the numbers. With points sd s ≈ 3 and S = 10,000, the SE of the mean is about **0.03 points**. For a tail event with p = 0.05, the SE is about **0.002**. Both are far below model error (MSE ≈ 3.6). The walk-forward runs use 2,000 simulations, an SE of about 0.07, and the emulator grid uses 10⁵ because its errors feed into every inversion.

**The 1/√S law:** halving the error costs 4× the simulations. Spend simulations where the error matters.

## 3. The algorithm, one fixture at a time

`simulate_fixture` (`src/fplh/sim/simulator.py:246`). The docstring at the top of the file lists the steps:

1. **Team process** (G6, Module 12): goal and red-card minutes for each side in every simulation. Team goal totals are exactly the process's.
2. **Starting XI:** exactly 1 GK + 10 outfield players, with inclusion probabilities equal to π^S (§4 below).
3. **Exits:** each starter plays 60+ with probability π^60. The exit minute is drawn from the empirical distribution of real starters' exits (`TimingModel`, line 116). Exits beyond the substitution limit stay on.
4. **Entrants:** as many bench players as there are exits, sampled ∝ π^B, entering at the exit minutes.
5. **Red cards:** the team process's reds send off an on-pitch outfield player ∝ yellow rate, unreplaced.
6. **Goals:** each team goal is an own goal, a penalty or open play, allocated to the players on the pitch at that minute (Module 14).
7. **Penalty misses** at the league rate; the opposing keeper saves a share.
8. **Other events:** yellows, saves, defensive actions, then BPS and bonus across both teams.
9. **Points** from the official rules engine (`score_arrays`, Module 06).

Everything is an array of shape **(S, P)**, simulations × players. There are no loops over simulations, so 10 fixtures × 5,000 simulations with every component take **3.8 s**.

**Goals conceded while on the pitch** (lines 397–406) is a neat vectorised trick. Take the cumulative opponent goals by minute, then read it at the player's exit minute minus his entry minute, using `take_along_axis`.

## 4. Picking an XI with exact probabilities: systematic sampling

We need, for every simulation, a set of exactly 10 outfield starters such that **each player's inclusion probability equals his π^S**. Independent coin flips would give the right probabilities but random team sizes (9, 12, …). `systematic_sample` (line 158) does both:

1. **Scale** the π^S so they sum to 10 with each ≤ 1 (`_capped`, line 172: rescale, cap anything above 1, rescale the rest, repeat).
2. **Shuffle** the players into a random order (that randomises *joint* selections).
3. Lay their weights end to end on a line from 0 to 10, and drop a comb of 10 teeth spaced 1 apart, starting at a random u ∈ [0, 1).
4. A player is picked if a tooth lands in his segment. In code: `floor(c − u) − floor(c − w − u) ≥ 1`, with c the cumulative weight.

Each segment of length w ≤ 1 catches a tooth with probability exactly w, and exactly 10 teeth land in total. The goalkeeper stratum does the same with 1 tooth. Property tests check that **inclusion frequencies match π^S** and that there are always exactly 11 starters with one GK.

## 5. Common random numbers (CRN)

To compare two decisions (captain A or B? model X or Y?), you want the *difference* to be precise, not each estimate. If both are simulated with **the same random draws**, the shared noise cancels:

$$\text{Var}(\hat a - \hat b) = \text{Var}\,\hat a + \text{Var}\,\hat b - 2\,\text{Cov}(\hat a, \hat b)$$

CRN makes the covariance large and positive. That is the same idea as the paired comparisons of Module 07.

The repo uses it in three places:

- each fixture's generator is seeded from `(seed, crc32(fixture_uid))` (line 257), so any two runs see identical matches;
- the team simulator shares uniforms across fixtures;
- the emulator uses the same seed at every grid point.

Replays are therefore **bit-for-bit identical**, which the reproducibility tests check.

## 6. Coherence, enforced by tests

Property tests (Hypothesis, `tests/property/test_simulator_properties.py`) assert, for random inputs:

- player goals + opponents' own goals = team goals, in every simulation;
- exactly 11 starters, one of them a GK; substitutes within the limit; never more than 11 on the pitch;
- minutes within [0, 90]; assists ≤ goals;
- at most one yellow and none with a red; saves only by keepers who played; no bonus for non-players; at least 6 bonus points per match;
- points = the sum of components; identical replays;
- simulated starts and 60+ rates match their targets; the team-goal distribution matches the team-only simulator (χ² test).

## 7. Outputs

`summarise` (line 476) and `models/player_sim.py` produce, per player and fixture:

- expected points;
- the **points pmf** over −4…25;
- P(60+), P(play), P(points ≥ 2, 6, 10, 15);
- expected goals, assists, saves and bonus, and clean-sheet probability.

Double gameweeks simulate each fixture separately and **sum** per player (Module 06's property: rows score independently).

## 8. Validation: the Phase 3 exit gate

Walk-forward over 2022/23–2024/25, on 80,973 player-fixtures:

| Model | MSE | MAE | Spearman, played | Top-10 precision |
|---|---|---|---|---|
| **Simulator** | **3.633** | **0.970** | **0.379** | **0.441** |
| OpenFPL replica | 3.661 | 0.996 | 0.371 | 0.429 |
| A0 | 4.015 | 1.001 | 0.331 | 0.410 |
| Last 5 | 4.354 | 1.050 | 0.283 | 0.384 |

Simulator − replica: −0.029, CI [−0.048, −0.011]. **The gate passes.** It is narrow, 0.8 % of MSE, and 2024/25 alone is a tie. The plan says so.

**Calibration guardrails:** P(60+) predicted 0.2825 vs observed 0.2819; P(haul ≥ 10) 0.0167 vs 0.0164; CRPS 0.627. A simulator can be accurate on average and badly miscalibrated in the tails; this one is not.

**Error attribution** (`evaluate/attribution.py`) reruns the simulator with **actual outcomes forced in** (`ForcedSide`, line 96):

1. ŷ⁰ = the forecast;
2. ŷ¹ = the forecast with actual minutes imposed;
3. ŷ² = also with actual goal events imposed.

Then y − ŷ⁰ = (ŷ¹ − ŷ⁰) + (ŷ² − ŷ¹) + (y − ŷ²) splits the error *exactly* into **minutes**, **goal events** and **the rest**. That is where the 23 % / 73 % / 4 % of Module 13 comes from.

## Check yourself

1. Points sd 3.2, S = 2,000. What is the SE of expected points? And with S = 8,000?
   <details><summary>Answer</summary>3.2/√2000 = 0.072. With S = 8,000 it is 0.036: four times the simulations for half the error.</details>
2. Why not pick starters with independent Bernoulli(π^S) draws?
   <details><summary>Answer</summary>The team size would be random (sometimes 9, sometimes 12 starters), which breaks the rules and every downstream event. Systematic sampling keeps the exact marginal probabilities *and* a fixed count of 1 GK + 10 outfield.</details>
3. Two captain options are evaluated on the same 1,000 simulated matches vs on different ones. Which gives a more precise estimate of the difference, and why?
   <details><summary>Answer</summary>The same matches (common random numbers). The shared match randomness cancels in the difference, because the covariance term is large and positive.</details>

## Practical

```bash
uv run python docs/learn/exercises/ex15_monte_carlo.py
```

You will compute Monte Carlo standard errors, implement systematic sampling and capped scaling (checked against the repo's), measure the variance reduction from common random numbers, and run the repo's `simulate_fixture` to verify coherence yourself.

Quiz: `uv run python docs/learn/quiz.py take 15`
