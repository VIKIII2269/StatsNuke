# 03 · Likelihood and optimisation

> **Goal:** understand what "fit" means in every `fit()` in the repo. A model turns parameters into probabilities for the data. Fitting finds the parameters that make the observed data most probable, usually with a penalty that keeps them sensible. Then a numerical optimiser does the searching.

## 1. Models are probability machines

A **model** says: "given parameters θ, data y has probability p(y | θ)". For example:

- **M1 (team strength):** given attack and defence ratings and home advantage, home goals ~ Poisson(e^(c + η + a_home − d_away)).
- **G1 (Dixon–Coles):** given ρ, the probability of each scoreline.
- **M4 (minutes):** given tree weights, P(start) for each player.

## 2. Likelihood

Fix the observed data y₁…yₙ and treat p as a function of θ. That is the **likelihood**. For independent observations it is a product:

$$L(\theta) = \prod_{i=1}^n p(y_i\mid\theta), \qquad \ell(\theta) = \log L(\theta) = \sum_{i=1}^n \log p(y_i\mid\theta)$$

The log turns a product of tiny numbers into a sum of manageable ones (Module 02, §6). The **maximum likelihood estimate (MLE)** is the θ that maximises ℓ.

### Worked example: a team's scoring rate

A team scores 2, 0, 1, 3, 1 in five matches. Model: goals ~ Poisson(λ).

$$\ell(\lambda) = \sum_i \big(y_i\log\lambda - \lambda - \log y_i!\big) = 7\log\lambda - 5\lambda - \text{const}$$

To find the top of the curve, set the slope to zero:

$$\frac{d\ell}{d\lambda} = \frac{7}{\lambda} - 5 = 0 \;\Rightarrow\; \hat\lambda = \frac{7}{5} = 1.4$$

For a Poisson, the MLE is the sample mean. That is reassuring, but the general method matters more, because most models have no closed form.

## 3. Derivatives: the minimum you need

A derivative is a **slope**: how fast f changes as x changes.

| f(x) | f′(x) |
|---|---|
| c (constant) | 0 |
| xⁿ | n xⁿ⁻¹ |
| eˣ | eˣ |
| log x | 1/x |
| g(h(x)) | g′(h(x)) · h′(x) (the **chain rule**) |

With many parameters, the **gradient** ∇ℓ is the vector of partial derivatives (one slope per parameter). The **Hessian** is the matrix of second derivatives, the *curvature*. At a maximum the gradient is zero and the curvature is negative.

**The Poisson regression gradient** (used all over the repo). With η = log μ and the model y ~ Poisson(e^η):

$$\ell = y\eta - e^{\eta} - \log y!,\qquad \frac{\partial\ell}{\partial\eta} = y - e^{\eta} = y - \mu,\qquad \frac{\partial^2\ell}{\partial\eta^2} = -\mu$$

The gradient is simply **observed minus expected**. If η = xᵀβ (a linear model on the log scale, a **Poisson GLM**), the chain rule gives ∂ℓ/∂β = Σᵢ (yᵢ − μᵢ) xᵢ.

Look at `src/fplh/models/goal_process.py:251`:

```python
resid = y - w * lam  # dℓ/dη
grad = [np.bincount(c.gbin, weights=resid, minlength=N_G)]
```

The derivative for each time-bin parameter is the sum of (observed − expected) over the cells in that bin. `np.bincount` with weights is a vectorised "sum per group". (The extra `w` comes from integrating out frailty, Module 12.)

## 4. Optimisers: how the computer climbs

Optimisers *minimise* by convention, so code minimises the **negative** log-likelihood (NLL).

| Method | Uses | Where in repo | When |
|---|---|---|---|
| **Gradient descent** | gradient | (conceptual) | Step downhill: θ ← θ − α∇f |
| **Newton's method** | gradient + Hessian | M1 update, `team_strength.py:244`; emulator inversion, `sim/emulator.py:99` | Few parameters with cheap curvature; converges in a handful of steps |
| **L-BFGS-B** | gradient (estimates curvature) + bounds | `market.invert_poisson`, `GoalModel.fit`, goal process `_Fit.run` | Medium or large smooth problems, with box bounds |
| **Nelder–Mead** | function values only | `Fusion.fit`, M1 `fit_hyperparameters` | Small, noisy or non-smooth objectives with no gradient |
| **Brent's method** (`brentq`) | 1-D sign change | power/Shin de-vig, `_logit_shift`, COM-Poisson rate | Solve f(x) = 0 for one unknown |

**Newton's method** in one dimension:

$$\theta_{\text{new}} = \theta - \frac{f'(\theta)}{f''(\theta)}$$

It fits a parabola at the current point and jumps to the parabola's peak. On the Poisson example in the log scale, η ← η + (Σy − n e^η)/(n e^η), which converges to log 1.4 in about five steps. M1's per-match update (`team_strength.py:244-256`) is exactly this in two dimensions, plus a prior term:

```python
grad = y - mu  # Poisson part
hess = s_inv + np.diag(w)  # prior curvature + data curvature
step = np.linalg.solve(hess, grad - s_inv @ (eta - eta0))
eta = eta + step
```

**Why provide the gradient?** With `jac=True`, the objective returns (value, gradient) and L-BFGS-B uses them directly. Without it, SciPy estimates each partial derivative by finite differences, which costs one extra evaluation per parameter per step and is less accurate.

## 5. Keeping parameters legal: reparameterisation

Rates must be positive and probabilities must lie in (0, 1). Rather than constrain the optimiser, **optimise a transformed variable that can be anything**:

- `invert_poisson` optimises x = log λ and uses λ = eˣ, which is always positive (`market.py:128`).
- `fit_hyperparameters` maps an unbounded z to the interval [lo, hi] by `lo + (hi − lo)·σ(z)` (`team_strength.py:366`).
- The frailty shape a > 0 is optimised as log a (`goal_process.py:205`).

## 6. Overfitting and regularisation

More parameters always fit the *training* data better, but can fit **noise**, which then fails on new data. That is **overfitting**. Defences used here:

- **Ridge (L2) penalty:** minimise NLL + λ‖β‖². It pulls parameters toward 0 unless the data insist. Fusion uses it (`fusion.py:80`), and so do the game-state effects (`goal_process.py:263`).
- **Smoothness penalty:** the time profile log g is penalised by `rw · Σ (log g_{b+1} − log g_b)²`, so neighbouring 5-minute bins cannot jump around (`goal_process.py:262-263`).
- **Shrinkage priors** (Module 04): the Bayesian version of the same idea.
- **Choosing λ by cross-validation:** `fit_fusion_cv` (`fusion.py:100`) tries ridge 0.01, 0.1, 1 and 10. For each, it fits on all seasons but one and scores the held-out one, then picks the best. The real result: an unshrunk fit put a −0.06 home bias on the market and did *worse* than the market. With CV-chosen ridge 1.0, the bias shrank to −0.016.

A regularised MLE is the same as a **MAP estimate** under a prior. A ridge penalty λβ² is a Normal prior on β (Module 04).

## 7. Which objective? Match it to the decision

| Objective | Equivalent to | Used for |
|---|---|---|
| Squared error | Gaussian likelihood | MSE of expected points; least-squares BPS weights |
| Log loss / NLL | Bernoulli or categorical likelihood | Minutes classifiers, 1X2 |
| Scoreline-grid NLL | Joint pmf likelihood | Fusion, G-ladder |
| **One-step-ahead predictive likelihood** | Honest out-of-sample fit | M1 hyper-parameters |

M1's hyper-parameters (`fit_hyperparameters`, `team_strength.py:342`) maximise the log-likelihood of **each match predicted before it was seen** (`team_strength.py:225-227`). That is walk-forward evaluation folded into the objective, so the hyper-parameters cannot reward hindsight.

## 8. Why the repo fits things this way

- **Closed form when possible.** The Gamma–Poisson posterior (Module 04) needs no optimiser at all, so it is fast and stable.
- **Analytic gradients for big likelihoods.** The goal process fits about 30 parameters over hundreds of thousands of minute-cells in 0.4–2 s because its gradient is exact.
- **Derivative-free for awkward objectives.** M1's objective runs the whole filter over 3,000 matches, and its gradient would be hard to derive, so Nelder–Mead with restarts is used. Real finding: one 400-iteration run stopped early at −2.90002, while two warm-started chains with restarts agree to 1e-4 at −2.89881.

## Check yourself

1. Goals 0, 1, 1, 2: what is the Poisson MLE of λ? And the NLL at that λ, ignoring constants?
   <details><summary>Answer</summary>λ̂ = 4/4 = 1. The NLL up to constants is −(Σy·log λ − nλ) = −(4·0 − 4) = 4.</details>
2. What is ∂ℓ/∂η for a Poisson observation y with log-mean η?
   <details><summary>Answer</summary>y − e^η = observed − expected.</details>
3. Why does `invert_poisson` optimise log λ instead of λ?
   <details><summary>Answer</summary>So λ = e^x stays positive whatever x the optimiser tries. It also makes the problem better scaled: multiplicative changes become additive.</details>
4. What does a ridge penalty correspond to in Bayesian terms?
   <details><summary>Answer</summary>A Normal prior centred at 0 on the penalised parameters. The penalised optimum is the posterior mode (MAP).</details>

## Practical

```bash
uv run python docs/learn/exercises/ex03_likelihood.py
```

You will write a Poisson log-likelihood, find the MLE by Newton's method, fit a small Poisson GLM (home advantage) with an analytic gradient, and see ridge shrink coefficients.

Quiz: `uv run python docs/learn/quiz.py take 03`
