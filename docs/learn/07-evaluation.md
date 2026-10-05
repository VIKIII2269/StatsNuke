# 07 · Evaluating forecasts honestly

> **Goal:** know how to say "model A is better than model B" and mean it. That takes the right **score**, the right **protocol** (walk-forward) and the right **uncertainty** (block bootstrap and Diebold–Mariano). This module is why the repo can say "Phase 4 failed" with confidence.

## 1. Scores for probabilistic forecasts

A forecast here is a *distribution*: P(home, draw, away), a scoreline grid, a points pmf. We need a number that rewards forecasts that are both **sharp** (confident) and **calibrated** (right as often as they claim).

A scoring rule is **proper** if you minimise your expected score by reporting your *true* belief. Improper rules reward shading, such as always predicting 0 or 1. Every rule below is proper.

### Log loss (negative log-likelihood)

$$\text{LL} = -\frac1n\sum_i \log p_i(y_i)$$

Only the probability given to what actually happened counts. Saying 1 % for something that happens costs −log 0.01 = 4.6 nats. Saying 0 % costs **infinity**, which is why the code clips at 1e-15 (`metrics.py:19`).

*Example:* a forecast (0.5, 0.3, 0.2) for (H, D, A), and the match is drawn. The loss is −log 0.3 = 1.204.

### Brier score

For a binary event, Brier = mean of (p − y)². It is bounded in [0, 1] and less harsh on confident misses than log loss. M4's minutes stages use it (`metrics.py:49`).

### Ranked probability score (RPS): ordered outcomes

Home win, draw, away win are **ordered**: a draw is "closer" to a home win than an away win is. RPS compares *cumulative* distributions:

$$\text{RPS} = \frac{1}{r-1}\sum_{k=1}^{r-1}\Big(\sum_{j\le k}(p_j - o_j)\Big)^2$$

*Example:* forecast (0.5, 0.3, 0.2), home win, so o = (1, 0, 0).

- Cumulative forecast: (0.5, 0.8). Cumulative outcome: (1, 1).
- Differences: (−0.5, −0.2).
- RPS = (0.25 + 0.04)/2 = **0.145**.

Had it been an away win, o = (0, 0, 1): differences (0.5, 0.8), RPS = (0.25 + 0.64)/2 = 0.445. Missing "by more" costs more. The code is `rps`, `metrics.py:22`. RPS is the headline match metric in Phase 2: market 0.19399 vs fused 0.19419.

### Scoreline-grid log loss

The log loss of the exact score, from the joint grid. Scores beyond 6 share the leftover mass (`scoreline_grid_log_loss`, `metrics.py:32`). It is the sharpest test of a goal model, because it checks the whole joint distribution, not just 1X2.

### CRPS for counts

For a pmf over 0..K, CRPS = Σₖ (F(k) − 1[y ≤ k])², the RPS generalised to many ordered outcomes (`crps_discrete`, `metrics.py:75`). The simulator's points pmf scores CRPS 0.627.

### Point forecasts: MSE and MAE

For *expected points*, mean squared error is proper for the mean. The exit-gate numbers (simulator 3.633 vs replica 3.661) are MSEs. MAE rewards medians instead.

### Ranking metrics

FPL decisions depend on *ordering* players, so the repo also reports **Spearman correlation within position** (`spearman_within`, `metrics.py:95`) and **top-10 precision**. A model can have a decent MSE and still rank poorly.

## 2. Calibration

**Calibrated:** among all the times you said 30 %, the event happened about 30 % of the time.

**Expected calibration error (ECE):** bin the predictions into 10 equal-width bins. For each bin, take |mean prediction − observed frequency|, weight it by the bin's share of predictions, and sum (`ece`, `metrics.py:54`).

| Bin | Share of forecasts | Mean predicted | Observed | Contribution |
|---|---|---|---|---|
| 0.0–0.1 | 0.6 | 0.04 | 0.05 | 0.6 × 0.01 = 0.006 |
| 0.8–0.9 | 0.4 | 0.85 | 0.80 | 0.4 × 0.05 = 0.020 |
| **ECE** | | | | **0.026** |

The minutes model reaches ECE 0.0015 on P(start); the simulator's P(60+) reaches 0.006. **Calibration is necessary but not sufficient.** Predicting the base rate for everyone is perfectly calibrated and useless.

**PIT for counts.** For a calibrated forecast distribution F, the value F(y) is uniform on [0, 1]. For discrete y, add noise: u = F(y−1) + v·[F(y) − F(y−1)] with v ~ U(0, 1) (`randomised_pit`, `metrics.py:66`). A histogram of u that isn't flat reveals bias (skew) or wrong spread (a U or hump shape).

## 3. Walk-forward evaluation

Random k-fold cross-validation **breaks time**: it trains on 2024 to predict 2023. Football has trends (substitution rules, tactics, data definitions), so that leaks. StatsNuke evaluates **walk-forward** (`evaluate/walk_forward.py:55`):

```text
for each deadline D in time order:
    info  = InformationSet.at(D)                    # only the past
    spine = players × fixtures in the next h rounds
    pred  = predictor.predict(info, spine)          # may refit on info
    assert info.max_observed_at <= D                 # proof of no leakage
```

Season roles:

- **Training history:** 2014/15 onward.
- **Tuning / gate seasons:** 2022/23–2024/25. Models are compared here.
- **Holdout:** 2025/26, **untouched**. It is the final exam, used once. The only documented exception is M7, which has no other data.
- **Live:** 2026/27 as it happens.

Expensive models refit every 4 deadlines, not every one, to save compute. Runs are cached by a hash of predictor version, data and config (Module 18).

## 4. Is the difference real? Paired comparisons

Every model is scored **on the same rows**, and we study the **difference** d = loss_A − loss_B per row. Pairing removes the shared difficulty: a chaotic gameweek is hard for both models.

### Why blocks

Rows in the same gameweek are **correlated**: same fixtures, same weather, same surprise results. Treating 80,973 player-fixtures as independent would make the confidence intervals far too narrow. So the repo:

1. averages d **within each gameweek**, giving 110 block means (`per_block_diff`, `bootstrap.py:33`);
2. **bootstraps** by resampling 110 blocks *with replacement*, 2,000 times, and recomputing the mean each time;
3. reads the 2.5 % and 97.5 % quantiles as a 95 % CI (`compare`, `bootstrap.py:72`).

If the CI excludes 0, the difference is significant at about the 5 % level.

### Diebold–Mariano test

DM tests H₀: E[d] = 0 with a t-statistic on the block differences:

$$\text{DM} = \frac{\bar d}{\sqrt{\widehat{\text{LRV}}/n}}$$

LRV is the **long-run variance**, estimated by Newey–West. It adds weighted autocovariances to the variance, in case consecutive gameweeks' differences are correlated (`newey_west_lrv`, `bootstrap.py:43`).

The repo also applies the **Harvey–Leybourne–Newbold correction** and Student-t critical values with n − 1 degrees of freedom (`diebold_mariano`, `bootstrap.py:54`). Why? A finding from real data: the textbook Normal DM rejected **8.6 %** of A-vs-A comparisons at season length (38 gameweeks) when it should reject 5 %. The corrected version rejects about 5 %. A test that "finds" differences between identical models 8.6 % of the time is dishonest.

## 5. Non-inferiority, superiority and power

- **Superiority:** the CI lies entirely below 0. A is better.
- **Non-inferiority:** the CI includes 0 but its upper end is small. A is *not worse*. The Phase 2 gate was met this way: fused − market RPS = +0.00012, CI [−0.00046, +0.00071].
- **Power:** can the test detect the effect at all? In the Phase 4 replay, gameweek scores have sd ≈ 15. With n = 110 gameweeks, the standard error of a mean difference is roughly 15/√110 ≈ 1.4 points. Detecting a true effect with 80 % power at the 5 % level needs it to be about (1.96 + 0.84) × 1.4 ≈ **4 points per gameweek**. The simulator's real edge (+2.3) is about half that, so the gate *could not* pass with three seasons, even if the edge is real.

**Pre-registration.** The gates were fixed *before* seeing results. Where a change was made after seeing a result (the fusion ridge, Module 11), the plan reports **both** runs. That is how you avoid fooling yourself.

## Check yourself

1. Forecast (0.6, 0.25, 0.15), result: away win. What are the RPS and the log loss?
   <details><summary>Answer</summary>Cumulative forecast (0.6, 0.85) vs outcome (0, 0), so the differences are (0.6, 0.85) and RPS = (0.36 + 0.7225)/2 = 0.541. Log loss = −ln 0.15 = 1.897.</details>
2. Why resample gameweeks instead of rows?
   <details><summary>Answer</summary>Losses within a gameweek are correlated. Row resampling treats them as independent and gives CIs that are too narrow, declaring differences "significant" too often.</details>
3. A model predicts every player starts with probability 0.30 (the base rate). Is it calibrated? Is it useful?
   <details><summary>Answer</summary>Calibrated overall (ECE ≈ 0 in its one bin), but useless: no sharpness, it cannot tell starters from benchwarmers. Its Brier score is far worse than M4's.</details>

## Practical

```bash
uv run python docs/learn/exercises/ex07_evaluation.py
```

You will implement log loss, RPS and ECE against `fplh.evaluate.metrics`, build a gameweek-block bootstrap CI that matches `compare`, and show by simulation that row-resampling CIs are too narrow when losses are correlated.

Quiz: `uv run python docs/learn/quiz.py take 07`
