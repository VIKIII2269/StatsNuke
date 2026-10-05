# 10 · Dynamic team strength (M1)

> **Goal:** understand the mathematical core of the team level. Each team's attack and defence are hidden quantities that drift over time. The filter keeps a Gaussian *belief* about all of them, predicts each match from that belief, and updates it after the result. You will meet state-space models, the Kalman filter, the Laplace approximation, and a glimpse of MCMC.

## 1. The model

For a match with home team i and away team j:

$$\log\mu_H = c + \eta + a_i - d_j,\qquad \log\mu_A = c + a_j - d_i$$

| Symbol | Meaning | Fitted value |
|---|---|---|
| c | League scoring level (log scale) | starts at 0.30 |
| η | Home advantage | 0.199, i.e. e^0.199 = 1.22, so home sides score about 22 % more |
| a_i | Attack of team i (higher = scores more) | |
| d_i | Defence of team i (higher = concedes less) | |

Goals are Poisson: Y_H ~ Pois(μ_H) and Y_A ~ Pois(μ_A).

**Identifiability.** Adding 0.1 to every attack *and* subtracting 0.1 from c gives identical predictions. So does adding 0.1 to every attack and every defence. The parameters are not identifiable. The fix is **sum-to-zero**: Σ a_i = Σ d_i = 0 over active teams. `_center` (`src/fplh/models/team_strength.py:120`) projects the state onto that constraint after every update.

## 2. Ratings move: a state-space model

Form changes. A **state-space model** has two parts:

- **Transition:** how the hidden state evolves between observations.
- **Observation:** how the data depend on the state.

The transition here is a **mean-reverting random walk**. After Δ days:

$$a \leftarrow \phi^{\Delta/365}\, a + \varepsilon,\qquad \varepsilon\sim\mathcal N\!\Big(0,\ \sigma_a^2\,\frac{\Delta}{365}\Big)$$

φ < 1 pulls ratings toward 0 (the average); σ² adds uncertainty proportional to elapsed time. In code (`propagate`, line 161):

```python
decay = p.phi_year ** (days * DAY)  # DAY = 1/365
self.m = f * self.m  # mean shrinks
self.P = (f[:, None] * self.P * f[None, :]) + np.diag(q)  # variance: shrink, then add q
```

**Forecasting h rounds ahead** chains this. The variance keeps growing, so a fixture six weeks away is forecast with *wider* uncertainty, exactly the ARCHITECTURE §7.2 horizon formula. Because of the log-normal mean (Module 02), the expected goals use exp(η̂ + s²/2) (`expected_goals`, line 198).

## 3. The Kalman filter in one dimension

Before the real thing, the linear-Gaussian case. Your belief about a quantity is N(m, v). You observe y = θ + noise, with noise ~ N(0, r):

$$K = \frac{v}{v + r},\qquad m' = m + K\,(y - m),\qquad v' = (1-K)\,v$$

K is the **gain**: how much you trust the new observation relative to your prior.

*Example:* m = 0, v = 0.04 (rating sd 0.2), r = 0.36 (a noisy match), y = 0.5. Then K = 0.04/0.40 = 0.1, m' = 0.05 and v' = 0.036. One match moves the rating 10 % of the way and shrinks uncertainty a little.

In many dimensions, with state mean **m**, covariance **P** and observation matrix **H**: S = HPHᵀ + R, K = PHᵀS⁻¹, **m** ← **m** + K(y − H**m**), and **P** ← **P** − KSKᵀ. Covariance lets one match update *other* teams too: if we learn that A's attack minus B's defence is high, everything correlated with those adjusts.

## 4. The real update: Laplace + exact conditioning

Goals are Poisson and xG is Gamma, not Gaussian, so the plain Kalman update doesn't apply. M1's update (`update`, `team_strength.py:205`) does two steps.

**Step 1. Laplace approximation in the 2-D match space.** The match depends on the state only through η = Hθ + (η, 0), the two log-rates. Its prior is N(η₀, S) with S = HPHᵀ. The log-posterior of η is

$$\underbrace{-\tfrac12(\eta-\eta_0)^\top S^{-1}(\eta-\eta_0)}_{\text{prior}} + \underbrace{\sum_k \big(y_k\eta_k - e^{\eta_k}\big)}_{\text{Poisson goals}} + \underbrace{\omega\sum_k \kappa\big(-\eta_k - x_k e^{-\eta_k}\big)}_{\text{Gamma xG, weighted by }\omega}$$

Differentiate (Module 03). The gradient is

$$g = (y - \mu) + \omega\kappa\Big(\frac{x}{\mu} - 1\Big) - S^{-1}(\eta - \eta_0)$$

and the negative curvature is

$$S^{-1} + \text{diag}\Big(\mu + \omega\kappa\,\frac{x}{\mu}\Big)$$

Newton iterations (lines 244–256) find the mode η*. The curvature at the mode gives the approximate posterior covariance V = (S⁻¹ + W)⁻¹. That is the **Laplace approximation**: a Gaussian centred at the mode, with the posterior's curvature.

**Step 2. Push it back to all teams exactly.** θ and η are jointly Gaussian under the prior, so conditioning θ on η is a linear-Gaussian formula. With G = PHᵀS⁻¹ (line 260):

```python
self.m = self.m + gain @ (eta - eta0)
self.P = self.P - gain @ (s - v) @ gain.T
```

- If V = S (we learned nothing), P is unchanged.
- If V → 0 (we learned η exactly), this is the full Kalman reduction.

Why the xG term looks like that: Gamma(κ, κ/μ) has log density κ log(κ/μ) − (κ/μ)x + …, which in η = log μ is −κη − κx e^{−η}. **ω = 0.12** means xG's likelihood counts about an eighth as much as goals. Even so, it significantly improves forecasts (§7).

## 5. Season transitions and promoted teams

Every summer (`start_season`, line 132):

- **Returning teams** have their ratings multiplied by `season_regress` and get extra variance `season_var` for squad churn. The fit put season_regress at 1: no summer regression is wanted.
- **Promoted teams** get a prior from their *Championship* ratings, translated to EPL scale: a = promoted_a + slope · a^E1. The fitted attack slope is 0.98, so Championship attack strength carries over almost one-for-one, from an offset of −0.62. A dominant promoted side is predicted better than a typical one. The championship ratings themselves come from a static Poisson fit (`static_ratings`, line 396).

## 6. Fitting the hyper-parameters

There are 14 hyper-parameters (φ, σ's, ω, κ, η, promoted priors…). Each is fitted by **maximising the one-step-ahead predictive log-likelihood**: every match is scored *before* the filter sees it (lines 225–227). The optimiser is Nelder–Mead in a logistic-transformed space, with restarts (`fit_hyperparameters`, line 342; Module 03). Results on 2014/15–2021/22, per match:

| Parameters | Predictive log-lik |
|---|---|
| Defaults | −2.90762 |
| Goals only (ω = 0) | −2.90591 |
| Without the E1 promoted prior | −2.90289 |
| **Fitted** | **−2.89881** |

Interesting fitted values: φ ≈ 1 (no within-season mean reversion), and σ_a sits at its **lower bound** 0.02, so attack ratings barely drift within a season, while σ_d = 0.10 lets defences drift more.

## 7. Results and what they teach

Walk-forward 2022/23–2024/25 (Module 07 protocol):

- **xG helps (ablation A2):** goals + xG vs goals only, RPS −0.0041, CI [−0.0071, −0.0018], DM p = 0.003. A real, significant gain.
- **Tuning barely matters at deadlines (A3):** tuned vs default RPS +0.0013, CI [−0.0014, 0.0048]. The fitted parameters win on the training objective, but not significantly on the evaluation seasons.
- **M1 alone vs market:** 0.197 vs 0.194. The market wins. M1's job is to add at the margin (Module 11).

## 8. The full-Bayes reference: NUTS

`team_strength_nuts.py` writes the same model in **NumPyro** and samples the full posterior with **NUTS**:

- **MCMC** (Markov chain Monte Carlo) draws samples whose distribution converges to the posterior. You get uncertainty for everything at once, not just a Gaussian approximation.
- **HMC / NUTS** uses gradients to make long, efficient moves. NUTS ("No-U-Turn Sampler") tunes the trajectory length automatically.
- **Non-centred parameterisation** (line 43): ratings are written as a₀ + cumsum(σ_a · z) with z ~ N(0, 1), instead of sampling a directly. When σ_a is small, the "funnel" geometry of the centred version traps samplers; this fixes it.

Full NUTS is too slow to refit at every deadline: a filter pass takes about 2.4 s against minutes for MCMC. So it serves as a **reference**. A test checks that the filter's ratings agree with a NUTS fit on the same data. A fast approximation plus a slow exact check is a pattern worth copying.

## 9. An engineering footnote

One filter pass over 3,040 matches took **76 s with 4 BLAS threads and 2.4 s with 1** (`run_filter`, line 327). Thousands of tiny matrix operations mean thread overhead dominates the work.

## Check yourself

1. In the 1-D Kalman example, what happens to the gain K if the observation noise r → 0? If r → ∞?
   <details><summary>Answer</summary>r → 0 gives K → 1: trust the data fully, m' = y. r → ∞ gives K → 0: ignore the data, m' = m.</details>
2. Why is sum-to-zero needed?
   <details><summary>Answer</summary>Without it, shifting all attacks up and the league level c down by the same amount leaves every prediction unchanged. The parameters would be unidentifiable and drift freely.</details>
3. With η = 0.2, c = 0.3, a_home = 0.25, d_away = −0.1, what is the home expected goal rate, ignoring uncertainty?
   <details><summary>Answer</summary>exp(0.3 + 0.2 + 0.25 − (−0.1)) = exp(0.85) = 2.34.</details>
4. What does it mean that σ_a sits at its lower bound?
   <details><summary>Answer</summary>The predictive likelihood prefers attack ratings that barely change within a season. Week-to-week attacking form is mostly noise, and it is better to average over many matches.</details>

## Practical

```bash
uv run python docs/learn/exercises/ex10_team_strength.py
```

You will code the 1-D Kalman update, propagation with mean reversion, and a 1-D Laplace update for a Poisson observation, checked against brute-force numerical integration. Then you will run the repo's `TeamStrengthFilter` on a simulated league and check that it recovers the true ordering.

Quiz: `uv run python docs/learn/quiz.py take 10`
