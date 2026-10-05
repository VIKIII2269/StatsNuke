# 13 · Minutes with gradient-boosted trees (M4)

> **Goal:** model the single biggest source of FPL forecast error: **whether, and how long, a player plays.** You will learn decision trees, gradient boosting (XGBoost), monotone constraints and isotonic calibration, and why the problem is split into stages.

## 1. Why minutes matter most

Error attribution (`IMPLEMENTATION_PLAN.md` §4.6) splits the simulator's forecast error:

- **Minutes, about 23 %.** Knowing who plays and for how long removes 0.81 of the 3.56 MSE.
- **Goal events, about 73 %.** This is mostly the irreducible luck of scoring.
- **The rest, about 4 %.**

Minutes is the largest *reducible* part. Ablation A4, which replaces M4 with naive "share of the last 3 matches", costs **+0.200 MSE**, seven times the simulator's entire margin over the OpenFPL replica. A benched premium forward scores 0, whatever his xG.

## 2. Stages: decompose, then multiply

Rather than one model for "minutes", M4 fits three conditional probabilities (`src/fplh/models/minutes.py:1-8`):

| Stage | Probability | Trained on |
|---|---|---|
| start | π^S = P(start) | all player-fixtures |
| full | π^60 = P(60+ min given start) | rows where the player started |
| sub | π^B = P(appears given not started) | rows where he didn't start |

So P(appears) = π^S + (1 − π^S)π^B, and P(starts and plays 60+) = π^S · π^60. Each stage is a simpler question, with its own cleanly labelled data. In the simulator, starters come from π^S, exits from π^60 and an empirical exit-minute distribution, and entrants from π^B (Module 15).

## 3. Decision trees

A tree splits the data with yes/no questions, such as "started ≥ 3 of the last 5?" or "chance_of_playing < 50?". Each leaf predicts a value. Training picks, at each node, the split that most reduces the loss. Trees capture **interactions and thresholds** naturally, for example "a regular starter *and* fully fit", with no feature engineering.

A single deep tree overfits; a shallow one underfits. The fix is many small trees.

## 4. Gradient boosting

Boosting builds an additive model of small trees, each correcting the previous ones:

$$F_M(x) = \sum_{m=1}^M \eta\, f_m(x),\qquad p = \sigma(F_M(x))$$

With the log-loss objective (`"objective": "binary:logistic"`), each new tree is fit to the **gradient and Hessian** of the loss with respect to the current log-odds:

$$g_i = p_i - y_i,\qquad h_i = p_i(1-p_i)$$

XGBoost uses both (second-order). The optimal value of a leaf containing rows I is a Newton step:

$$w^* = -\frac{\sum_{i\in I} g_i}{\sum_{i\in I} h_i + \lambda}$$

(λ is L2 regularisation on leaf values). Compare Module 03: boosting is Newton's method in "function space".

The repo's settings (`PARAMS`, `minutes.py:44`):

| Setting | Value | Role |
|---|---|---|
| eta (learning rate) | 0.05 | Shrinks each tree's contribution, so the many trees average out noise |
| max_depth | 5 | Interactions up to 5 features deep |
| min_child_weight | 20 | A leaf needs Σh ≥ 20, so leaves can't fit a handful of rows |
| subsample / colsample_bytree | 0.8 | Random rows and columns per tree reduce overfitting |
| rounds | 250 | Number of trees |

**Missing values:** XGBoost learns a default direction for missing values at each split. `chance_of_playing` is **missing, never imputed**, before the repo's own snapshot captures began (spec gap 1). The trees learn "missing" as its own case, instead of inventing a fake value.

## 5. Monotone constraints

Some relationships *must* go one way. A player who started more of his recent matches should never get a *lower* start probability, all else equal. Unconstrained trees can produce such inversions in sparse regions, which produces silly, unstable forecasts. `_train` (line 100) passes:

```python
cons = tuple(1 if c in MONOTONE_UP else 0 for c in x.columns)  # +1: non-decreasing
params = {**PARAMS, "monotone_constraints": "(" + ",".join(map(str, cons)) + ")"}
```

for `h_started_1/3/5/10` and `chance_of_playing`. This is domain knowledge encoded as a hard constraint. It costs almost nothing in accuracy and buys robustness.

## 6. Calibration with isotonic regression

Boosted probabilities often rank well but are miscalibrated, for example slightly overconfident at the extremes. **Isotonic regression** fits the best *non-decreasing* map from raw score to probability, by the pool-adjacent-violators algorithm (PAV):

1. Sort rows by raw prediction.
2. Walk through the labels. Whenever a block's mean is higher than the next block's, **merge** them and use the merged mean.
3. The result is a step function, interpolated linearly between knots (`Isotonic`, line 61).

**Honest calibration** needs held-out data. `fit_stage` (line 106) works like this:

1. Train a model on the chronologically **first 80 %** of rows.
2. Predict the **last 20 %** and fit the isotonic map there.
3. Train the final model on **all** rows and apply that map.

Fitting the calibrator on the training rows would learn the model's *in-sample* optimism. If fewer than 50 calibration rows exist, the map is the identity.

The results (walk-forward, 2022/23–2024/25):

| Stage | Brier | ECE | Mean predicted | Observed |
|---|---|---|---|---|
| Start | 0.0805 | 0.0015 | 0.301 | 0.301 |
| 60+ given start | 0.0615 | 0.0039 | 0.929 | 0.933 |
| Appearance given not started | 0.0866 | 0.0029 | 0.157 | 0.156 |

Against naive last-3 shares (A4), the start Brier score is −0.0241, CI [−0.0258, −0.0225].

## 7. Real-data lessons

- **Substitution eras.** In 2022/23 the EPL moved from 3 to 5 substitutes. Trees can't extrapolate to a value of a feature they've never seen, and the features carry no era. So π^B gets a **logit shift per substitution limit**: the δ such that mean σ(logit π^B + δ) equals the observed appearance rate on that era's own bench rows (`_logit_shift`, line 175, solved with `brentq`). Measuring it on the era's own rows controls for squad sizes. A rate-ratio version over-predicted 2022/23 by 0.03.
- **No per-team normalisation.** It seems natural to rescale each team's π^S to sum to 11. But the deadline squad list includes departed and long-term injured players, and scaling over it biased every probability. The repo removed it.
- **The unit bug** (Module 01): days since the last appearance were computed from raw timestamps in mixed µs/ns units, and walk-forward P(start) collapsed to 0.14. It was caught *because* the walk-forward mean prediction was checked against the observed rate.

## Check yourself

1. π^S = 0.7, π^60 = 0.85, π^B = 0.4. What are P(appears) and P(starts and plays 60+)?
   <details><summary>Answer</summary>P(appears) = 0.7 + 0.3 × 0.4 = 0.82. P(starts and plays 60+) = 0.7 × 0.85 = 0.595.</details>
2. For logistic boosting, what are g and h for a row with p = 0.8 and y = 0?
   <details><summary>Answer</summary>g = p − y = 0.8 and h = p(1 − p) = 0.16.</details>
3. Why fit the isotonic calibrator on the last 20 % using a model trained on the first 80 %?
   <details><summary>Answer</summary>Calibration must be learned on predictions the model didn't train on. In-sample predictions are overconfident, so a map learned on them would be wrong for new data. Chronological order mimics live use.</details>

## Practical

```bash
uv run python docs/learn/exercises/ex13_minutes.py
```

You will implement PAV isotonic regression (checked against SciPy), the boosting gradient, Hessian and leaf value, the substitution-era logit shift (against the repo's), and train the repo's `fit_stage` on synthetic data to verify calibration and the monotone constraint.

Quiz: `uv run python docs/learn/quiz.py take 13`
