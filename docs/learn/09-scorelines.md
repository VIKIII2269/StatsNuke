# 09 · Scoreline models G0–G3

> **Goal:** model the *joint* distribution of home and away goals, the scoreline grid. Every match market (1X2, totals, BTTS) and every clean-sheet probability comes from it. You will learn four classical models, how each one bends the simplest one, and the humbling result that on real data none of them beat it.

## 1. The scoreline grid

A grid P(i, j) gives the probability that the home team scores i and the away team j, for i, j ∈ 0..10 (`MAX_GOALS`). Rows are home goals and columns away goals. All the match markets are sums over regions of it (`markets`, `src/fplh/models/goal_benchmarks.py:104`):

```python
"home": np.tril(grid, -1).sum()     # i > j: below the diagonal
"draw": np.trace(grid)              # i = j: the diagonal
"away": np.triu(grid, 1).sum()      # i < j: above the diagonal
"over": grid[total > 2.5].sum()     # i + j ≥ 3
"btts": grid[1:, 1:].sum()          # both score
```

For FPL, a home clean sheet means the away team scores 0, which is the first *column*: P(home clean sheet) = Σᵢ P(i, 0).

## 2. G0: independent Poisson

$$P(i,j) = \text{Pois}(i;\lambda_h)\,\text{Pois}(j;\lambda_a)$$

The two teams' goals are independent, each Poisson with its own mean. It is an outer product of two pmfs (`g0_grid`, line 38).

With λ_h = 1.5 and λ_a = 1.1: P(1-1) = 0.1226, P(draw) = 0.258, P(over 2.5) = 0.482.

It has one big documented flaw: **real draws, especially 0-0 and 1-1, happen more often** than it predicts.

## 3. G1: Dixon–Coles (1997)

Dixon and Coles multiply the four low-score cells by a correction τ with one parameter ρ:

| Cell | τ |
|---|---|
| 0-0 | 1 − λ_h λ_a ρ |
| 0-1 | 1 + λ_h ρ |
| 1-0 | 1 + λ_a ρ |
| 1-1 | 1 − ρ |
| otherwise | 1 |

With ρ < 0, 0-0 and 1-1 go **up** and 1-0 and 0-1 go **down**. Example with λ = (1.5, 1.1):

| Cell | G0 | G1, ρ = −0.045 (fitted on real data) | G1, ρ = −0.15 |
|---|---|---|---|
| 0-0 | 0.0743 | 0.0798 | 0.0927 |
| 0-1 | 0.0817 | 0.0762 | 0.0633 |
| 1-0 | 0.1114 | 0.1059 | 0.0930 |
| 1-1 | 0.1226 | 0.1281 | 0.1409 |
| 2-1 | 0.0919 | 0.0919 | 0.0919 |

Two neat facts, which you can check from the formulas:

- The four adjustments **cancel exactly in total mass**: Pois(0)Pois(0)·λ_hλ_aρ = Pois(0)Pois(1)·λ_hρ, and so on. The grid still sums to 1, and every cell where i + j ≥ 3 is untouched, so **P(over 2.5) does not change**.
- Only the draw/no-draw split moves. ρ is a pure "draw dependence" knob.

In code (`g1_grid`, line 42), the τ factors are multiplied in place, the grid is clipped at a tiny positive value, and it is renormalised.

## 4. G2: bivariate Poisson with diagonal inflation

A shared component creates positive correlation:

$$X = W_1 + W_3,\quad Y = W_2 + W_3,\qquad W_r \sim \text{Pois}(\lambda_r)$$

so Cov(X, Y) = λ₃. A common "open game" factor raises both teams' goals together. Then the diagonal is inflated: with probability π the score is a draw (k, k), with k ~ Pois(θ):

$$P = (1-\pi)\,\text{BP}(x,y) + \pi\, q(x)\,\mathbb 1[x=y]$$

The code holds the marginal means at (λ_h, λ_a) by setting λ₁ = λ_h − λ₃ and λ₂ = λ_a − λ₃ (`g2_grid`, line 72). The bivariate pmf sums over the shared count k in log space with `gammaln`, to avoid overflow (`_bivariate_poisson`, line 51).

## 5. G3: Conway–Maxwell–Poisson marginals

$$P(Y=y) = \frac{\lambda^y}{(y!)^\nu\,Z(\lambda,\nu)}$$

The extra parameter ν bends the dispersion:

- ν = 1: Poisson.
- ν > 1: **underdispersed** (variance < mean).
- ν < 1: overdispersed.

Example: with ν = 1.3 and a mean held at 1.5, the variance is 1.27, not 1.5. The code finds λ by `brentq`, so that the marginal mean equals the target (`_com_rate_for_mean`, line 90). Means stay comparable across models; only the *shape* changes.

## 6. Fitting and inverting, the same way for all four

`GoalModel` (line 116) wraps any grid function:

- `fit(lh, la, hg, ag)`: maximise Σ log P(observed score) over the dependence parameters (ρ, or λ₃/π/θ, or ν) with L-BFGS-B, given per-match rates from another model (M1 or the market).
- `invert(p1x2, p_over)`: find (λ_h, λ_a) whose grid reproduces market prices, as in Module 08 but under this model.
- `log_prob`: vectorised for G0/G1 (closed form), cell lookup for the others.

This uniform interface is why the emulator (Module 12) can "plug into market inversion and fusion unchanged". It just provides another `grid_fn`.

## 7. The real-data verdict

Walk-forward over 2022/23–2024/25 with M1 rates (`IMPLEMENTATION_PLAN.md` §3.2–3.3):

| Model | RPS | Scoreline log loss | Fitted dependence |
|---|---|---|---|
| M1 + G0 | 0.19717 | 2.99325 | — |
| M1 + G1 | 0.19711 | 2.99407 | ρ = −0.045 |
| M1 + G2 | 0.19716 | 2.99359 | λ₃ → 0 |
| M1 + G3 | 0.19715 | 2.99284 | ν = 1.018 |

**Every CI against G0 includes zero.** The fitted parameters all sit near the Poisson case. Once team strength is accounted for, EPL scorelines are Poisson to within what three seasons can resolve.

So **G0 stays the pre-match default**. G1 is kept for market inversion and fusion, because ρ is the only non-trivial dependence parameter that survives fitting.

This is a central lesson of the course. Elaborate models are *hypotheses*; the ablation ladder (ARCHITECTURE §11.7) tests each step, and most steps fail. Reporting that honestly is the job.

## Check yourself

1. Which grid region gives P(away team keeps a clean sheet)?
   <details><summary>Answer</summary>The home team scores 0: the first row, Σⱼ P(0, j).</details>
2. With ρ < 0 in Dixon–Coles, which cells rise and which fall?
   <details><summary>Answer</summary>0-0 and 1-1 rise; 1-0 and 0-1 fall. Draws become more likely.</details>
3. Why doesn't Dixon–Coles change P(over 2.5)?
   <details><summary>Answer</summary>It only alters cells with i + j ≤ 2, and those adjustments cancel in total mass. The over region (i + j ≥ 3) is untouched and the grid still sums to 1.</details>
4. G3 fitted ν = 1.018. What does that say about goal dispersion?
   <details><summary>Answer</summary>Almost exactly Poisson, very slightly underdispersed. Conditional on team strength there is no meaningful overdispersion.</details>

## Practical

```bash
uv run python docs/learn/exercises/ex09_scorelines.py
```

You will build G0 and G1 grids from formulas, compute markets from a grid, prove numerically that Dixon–Coles preserves total mass and over 2.5, and fit ρ by maximum likelihood on simulated matches.

Quiz: `uv run python docs/learn/quiz.py take 09`
