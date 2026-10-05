# 08 · Betting markets and de-vig (M2)

> **Goal:** turn bookmaker odds into honest probabilities and then into scoring rates. Markets aggregate the opinions and money of thousands of people. De-vigged closing prices are among the best public match forecasts, so they are both a **benchmark** and an **input** for StatsNuke. (Paper only: nothing here bets.)

## 1. Odds and implied probability

**Decimal odds** o mean that a 1-unit stake returns o units in total if it wins. A fair price for an outcome with probability p would be o = 1/p, so the **implied probability** is π = 1/o.

A real 1X2 market: home 2.10, draw 3.40, away 3.60.

| Outcome | Odds | π = 1/o |
|---|---|---|
| Home | 2.10 | 0.4762 |
| Draw | 3.40 | 0.2941 |
| Away | 3.60 | 0.2778 |
| **Sum S** | | **1.0481** |

The π sum to more than 1. The excess, S − 1 = 4.8 %, is the **overround** or **vig**: the bookmaker's margin. To get probabilities you must remove it. That is **de-vig**.

## 2. Three ways to remove the margin

`src/fplh/models/market.py:24-57` implements three methods.

### Multiplicative (proportional)

$$p_k = \pi_k / S$$

Scale everything down equally. For the market above: (0.4543, 0.2806, 0.2650). It is simple, but it assumes the margin is spread in proportion to probability.

### Power

$$p_k = \pi_k^{\,c}, \quad\text{with } c \text{ solving } \sum_k \pi_k^c = 1$$

Because every π < 1, raising to a power c > 1 shrinks small numbers *proportionally more*. The method takes more margin off **longshots**. It is solved with `brentq`, a 1-D root find (Module 03). The result is (0.4602, 0.2780, 0.2618).

### Shin

Shin (1993) models a fraction z of bettors as *insiders*. Bookmakers protect themselves by shading longshot prices more:

$$p_k = \frac{\sqrt{z^2 + 4(1-z)\,\pi_k^2/S} - z}{2(1-z)},\qquad z \text{ chosen so that } \sum_k p_k = 1$$

For our market z = 0.024 and p = (0.4587, 0.2787, 0.2626).

### The favourite–longshot bias

Empirically, **longshots are overpriced**: implied probabilities overstate them more than favourites. Power and Shin correct for this, and multiplicative does not. The effect is clearer on a lopsided market (1.25 / 6.0 / 12.0):

| Method | Favourite | Draw | Longshot |
|---|---|---|---|
| Multiplicative | 0.7619 | 0.1587 | 0.0794 |
| Power | 0.7863 | 0.1450 | 0.0687 |
| Shin | 0.7778 | 0.1520 | 0.0701 |

## 3. Choosing the method by data

You don't pick a de-vig method by taste. `devig_calibration` (`market.py:96`) computes the **log loss of each method's probabilities against actual results** on historical closing prices, and the best one becomes the default (`configs/models/market.yaml`).

| Book (before 2022/23) | Matches | Multiplicative | Power | Shin |
|---|---|---|---|---|
| Pinnacle | 3,800 | 0.953245 | **0.953187** | 0.953239 |
| Market average | 1,140 | **0.968636** | 0.969353 | 0.969162 |

Power wins on Pinnacle, the sharpest book, and is the default. Notice how small the differences are: **< 1e-4 nats** on Pinnacle's thin margins. Choosing correctly matters little when the margin is small, and the plan says so honestly.

## 4. Which prices may be used

- **Pre-match prices observable at the deadline** may be used as forecast inputs.
- **Closing prices** (at kickoff) include late team news and late money. They are the best benchmark, but they are **never a pre-deadline feature** (`configs/sources.yaml`).

The A0 table shows the gap: on the same matches, deadline prices score RPS ≈ 0.204 and closing prices ≈ 0.203 in 2022/23. Information keeps arriving until kickoff.

**Real data hazard.** Pinnacle's closing odds stopped in July 2025, when its API closed. They are complete to 2024/25, cover 55 % of 2025/26, and are absent in 2026/27. The benchmark switches to de-vigged **market-average closing** prices from then on. Always check data availability before designing a benchmark.

## 5. From probabilities to scoring rates (market inversion)

The simulator needs **expected goals** (λ_home, λ_away), not 1X2 probabilities. So we **invert**: find the rates whose scoreline model reproduces the market.

Under independent Poisson goals (G0), any (λ_h, λ_a) gives a grid P(i, j) = Pois(i; λ_h) Pois(j; λ_a), which yields:

- P(home) = Σ_{i>j} P(i, j);
- P(draw) = Σ_i P(i, i);
- P(over 2.5) = Σ_{i+j>2} P(i, j).

`poisson_markets` computes these, and `invert_poisson` (`market.py:121`) minimises the squared distance to the market:

$$\min_{\lambda_h,\lambda_a}\ (P_H - p_H)^2 + (P_D - p_D)^2 + (P_A - p_A)^2 + (P_{O2.5} - p_{O2.5})^2$$

over log λ, with L-BFGS-B.

**Why include Over/Under 2.5?** 1X2 mostly pins down the *difference* in strength, not the *total* goals. Two lowly-scoring evenly matched teams and two high-scoring evenly matched teams have similar 1X2. With our market's 1X2 alone, inversion gives (1.32, 0.93). Adding P(over 2.5) = 0.52 gives (1.56, 1.16): same balance, more goals. The total-goals market identifies the level.

The fit is no longer perfect, though. At (1.56, 1.16), independent Poisson gives P(draw) = 0.251 against the market's 0.281. Independent Poisson **under-predicts draws** for a given goal total. That empirical regularity motivates Dixon–Coles and the in-match process (Modules 09 and 12).

The dependent goal models G1–G3 (Module 09) use the same idea through `GoalModel.invert`. With in-match dynamics (Module 12) the grid has no formula, so an **emulator** replaces it.

## 6. Why markets are hard to beat

A market price reflects team news, lineups, injuries, form, weather and the beliefs of professional bettors. The Phase 2 results:

- **M1 alone:** RPS 0.197.
- **Market (deadline prices):** 0.194.
- **Fused:** 0.194, a tie.

M1 learns from goals and xG. The market already knows most of that and more. A model's job is to *add information at the margin*, which is why fusion (Module 11) weights the market at about 83 %.

## 7. The paper ledger (preview)

Comparing your probability p̂ with a price o gives a paper expected value EV = p̂ o − 1. The real test of skill is **closing-line value**: did you get a better price than the de-vigged closing price? Module 17 covers it. The answer for StatsNuke is no edge (CLV −1.2 %), and the repo reports that plainly.

## Check yourself

1. Odds 1.80 / 3.80 / 4.50. What is the overround? What are the multiplicative probabilities?
   <details><summary>Answer</summary>π = (0.5556, 0.2632, 0.2222), S = 1.0409, so the overround is 4.1 %. p = π/S = (0.5337, 0.2528, 0.2135).</details>
2. Why might power de-vig beat multiplicative?
   <details><summary>Answer</summary>Because bookmakers shade longshots more (the favourite–longshot bias). Power takes proportionally more margin from small probabilities, which matches how the margin is really distributed.</details>
3. Why can't closing odds be a feature for a deadline forecast?
   <details><summary>Answer</summary>They are observed at kickoff, after the deadline, so they are outside 𝓘(D). They are a benchmark only.</details>

## Practical

```bash
uv run python docs/learn/exercises/ex08_devig.py
```

You will implement all three de-vig methods (with `brentq`), compare them against `fplh.models.market`, measure the favourite–longshot shading, and invert a market to Poisson rates with and without the totals price.

Quiz: `uv run python docs/learn/quiz.py take 08`
