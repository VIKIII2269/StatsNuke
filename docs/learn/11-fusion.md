# 11 · Fusion: combining the market and the model (M3)

> **Goal:** learn how to combine two forecasts so the result beats both, or at least never loses to the better one. You will see a log-linear pool with a learned weight, why the weight depends on time-to-kickoff, and an important real-data story about regularisation and honest reporting.

## 1. The problem

At each deadline StatsNuke has two estimates of every fixture's expected goals:

- **λ^mkt** from the de-vigged market, inverted to rates (Module 08), when a price exists;
- **λ^mod** from M1 (Module 10).

The market is usually better (RPS 0.194 vs 0.197), but M1 is not useless. It has a different error structure, and for gameweeks 2+ there is often **no price at all**. So we want a principled blend.

## 2. The log-linear pool

`src/fplh/models/fusion.py`:

$$\log\bar\lambda_k = w\,\log\lambda^{\text{mkt}}_k + (1-w)\,\log\lambda^{\text{mod}}_k + b_k$$

$$w = \sigma\big(\alpha_0 + \alpha_1\log(1+\tau) + \alpha_2\,\text{liq}\big)$$

- τ is the number of hours between the price snapshot and kickoff, so older prices can get less weight.
- liq is a liquidity proxy, log(1 + number of books) − 20·(overround − 0.05) (`liquidity`, line 118). More books and a thinner margin suggest a more informative price.
- σ is the logistic function, which keeps w in (0, 1) whatever the α's are.
- b_k is a small per-side bias correction (home, away).
- **No market price → w = 0**, and the model is used alone (line 52).

**Why average logs (geometric) and not rates (arithmetic)?** Rates are multiplicative quantities: a team is "20 % better". On the log scale, a weighted average is a weighted average of *multiplicative effects*, and the result is automatically positive. It also matches how both inputs were built: log-linear models in M1, and log-rates in the inversion. With λ^mkt = 1.6, λ^mod = 1.3 and w = 0.829:

- geometric: 1.6^0.829 · 1.3^0.171 = **1.544**;
- arithmetic: 0.829 · 1.6 + 0.171 · 1.3 = 1.549.

They are close here. They diverge when the inputs disagree a lot.

## 3. Fitting: scoreline NLL with ridge

The five parameters (α₀, α₁, α₂, b_H, b_A) minimise the **scoreline-grid negative log-likelihood** of observed results under a goal model, G1 in practice, plus a ridge penalty on everything except α₀ (`Fusion.fit`, line 75):

$$\min\ -\frac1n\sum_m \log P_{\text{G1}}\big(\text{score}_m \mid \bar\lambda_H,\bar\lambda_A\big) + r\,(\alpha_1^2+\alpha_2^2+b_H^2+b_A^2)$$

The data are the **deadline-time** snapshots of the five seasons before evaluation. Training on closing prices would teach the fusion to trust a quality of price it will never see at a deadline.

## 4. The story: a fit that looked fine and wasn't

From `IMPLEMENTATION_PLAN.md` §3.3, reported in full because it is a model of honest practice:

1. **First fit:** three seasons, fixed ridge 0.01. It put all its weight on the market (α₀ = 11.6, so w ≈ 1), *plus* a home bias of −0.06. The result was slightly **worse** than market-only (RPS +0.0008).
2. **Diagnosis:** the market's home-goal log bias swings from −0.13 to +0.04 between seasons, and was −0.10 in the behind-closed-doors 2020/21. A lightly regularised b_H was memorising one or two seasons' noise.
3. **Fix:** train on five seasons, and choose the ridge by **leave-one-season-out cross-validation** within them (`fit_fusion_cv`, line 100). No evaluated season is involved:

   | Ridge | 0.01 | 0.1 | **1.0** | 10 |
   |---|---|---|---|---|
   | Held-out-season NLL | 2.87886 | 2.87872 | **2.87846** | 2.87865 |

4. **Result:** bias −0.016, w = σ(1.58 − …) ≈ **0.83** at every deadline. That gives M1 about 17 % weight.
5. **Reporting:** the change was made *after* seeing the first result, so the plan reports **both runs**. Changing a method after seeing the test, without saying so, is how people fool themselves.

Note that α₁ ≈ −0.0002 and α₂ ≈ −0.00005 were fitted near zero. The time-to-kickoff and liquidity terms carried no signal in this data, and the ridge correctly shrank them away.

## 5. Results

| Comparison | RPS difference (95 % CI) | Scoreline log-loss difference |
|---|---|---|
| Fused − market only | +0.00012 [−0.00046, +0.00071] | −0.0022 [−0.0049, 0.0004] |
| Fused − M1 + G1 | −0.0020 [−0.0046, 0.0008] | −0.0072 |

The exit gate (fused ≥ market) was **met as non-inferiority**: fused is not worse, and is best on scoreline log loss, but not *significantly* better on anything. The market already prices most of what goals and xG know. Phase 3 adds what the market may *not* price as well: minutes, lineups and player-level xG.

## 6. Why fuse at all, then?

- **Where there is no market**, h > 1, the fused rate *is* M1 (w = 0). You still need a good model.
- **A never-worse guarantee.** A learned weight can drift toward whichever input is better as conditions change.
- **Separation of concerns.** The market supplies the level; the model supplies structure, such as the full scoreline distribution and later lineup effects.

## Check yourself

1. w = σ(1.58). What is w? What weight does M1 get?
   <details><summary>Answer</summary>σ(1.58) = 1/(1 + e^(−1.58)) = 0.829, so M1 gets 0.171.</details>
2. Why is b_H dangerous without enough regularisation?
   <details><summary>Answer</summary>The market's home bias varies by about ±0.1 between seasons. A free b_H fits the training seasons' particular bias, which does not persist, and so worsens out-of-sample forecasts.</details>
3. Why train fusion on deadline-time prices rather than closing prices?
   <details><summary>Answer</summary>At prediction time only deadline prices exist. Fitting on closing prices would overstate the market's informativeness and give the model too little weight. That is train/serve skew.</details>

## Practical

```bash
uv run python docs/learn/exercises/ex11_fusion.py
```

You will implement the pooled rate and the weight, verify them against `fplh.models.fusion.Fusion`, and fit a fusion on synthetic data where the "market" is noisier than the "model", watching the learned weight flip.

Quiz: `uv run python docs/learn/quiz.py take 11`
