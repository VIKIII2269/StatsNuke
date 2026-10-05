# 04 · Bayes and shrinkage

> **Goal:** understand the most important statistical idea in the player models. Small samples lie. A player with 2 goals in his only 90 minutes is not a 2-goals-per-game striker. Shrinkage pulls noisy individual estimates toward a sensible group average, by exactly as much as the data deserve. The Bayesian derivation tells you *how much*.

## 1. Bayes' rule

For a hypothesis H and data D:

$$P(H \mid D) = \frac{P(D \mid H)\,P(H)}{P(D)} \quad\Longleftrightarrow\quad \text{posterior} \propto \text{likelihood} \times \text{prior}$$

- The **prior** P(H) is what you believed before seeing D.
- The **likelihood** P(D | H) is how probable the data are if H is true.
- The **posterior** P(H | D) is the updated belief.

**Worked example.** 10 % of forwards are "elite", with a true shot rate of 4.0 per 90. The rest are "regular", at 2.0 per 90. A new forward takes 5 shots in one 90-minute match. Is he elite?

- Under elite: P(5 shots) = e⁻⁴ 4⁵/5! = 0.1563.
- Under regular: P(5 shots) = e⁻² 2⁵/5! = 0.0361.
- Posterior: P(elite | 5) = 0.1563 × 0.1 / (0.1563 × 0.1 + 0.0361 × 0.9) = 0.01563 / 0.04812 = **0.325**.

One great match moves the probability from 10 % to 32 %. It is far from certain, because one match is little evidence. That is shrinkage in miniature.

## 2. Continuous rates: the Gamma–Poisson model

In practice a player's true rate r is a continuous unknown. StatsNuke uses:

- **Prior:** r ~ Gamma(α, α/μ), which has mean μ (the player's group average) and variance μ²/α.
- **Likelihood:** in matches with exposures uᵢ (90-minute units), shots Sᵢ ~ Poisson(r uᵢ).

Multiply prior by likelihood, keeping only the terms with r:

$$\underbrace{r^{\alpha-1}e^{-(\alpha/\mu) r}}_{\text{prior}}\;\times\;\underbrace{\prod_i r^{S_i}e^{-r u_i}}_{\text{likelihood}} \;=\; r^{\alpha + S - 1}\,e^{-(\alpha/\mu + u)\,r}$$

with S = Σ Sᵢ and u = Σ uᵢ. That is again a Gamma shape. The prior is **conjugate**: the posterior is in the same family, so no optimiser or sampling is needed:

$$r \mid \text{data} \sim \text{Gamma}\Big(\alpha + S,\; \frac{\alpha}{\mu} + u\Big), \qquad \mathbb E[r\mid \text{data}] = \frac{\alpha + S}{\alpha/\mu + u}$$

### Read it as a weighted average

Rewrite the posterior mean as

$$\mathbb E[r\mid\text{data}] = \underbrace{w}_{\text{data weight}}\cdot\frac{S}{u} + (1-w)\cdot\mu, \qquad w = \frac{u}{u + \alpha/\mu}$$

- **α/μ is the prior's "pseudo-exposure"**: the prior counts as if you had already watched α/μ extra 90s at the group rate.
- With little data (u ≪ α/μ), w ≈ 0 and the estimate stays near the group mean.
- With lots of data, w → 1 and the estimate approaches the player's own rate.

**Worked example.** The group mean is μ = 2.0 shots/90 and α = 6, so the pseudo-exposure is 3 matches.

| Player | Shots S | 90s u | Raw S/u | Posterior mean (6+S)/(3+u) | Data weight w |
|---|---|---|---|---|---|
| New signing | 5 | 1 | 5.00 | 11/4 = **2.75** | 0.25 |
| Rotation player | 20 | 6 | 3.33 | 26/9 = **2.89** | 0.67 |
| Regular starter | 100 | 30 | 3.33 | 106/33 = **3.21** | 0.91 |

The two players with the same raw rate (3.33) get different estimates, because the starter has proved it over more minutes.

## 3. How strong should the prior be? Method of moments

α decides how much the group pulls. The repo **estimates it from the data** (this is called *empirical Bayes*) in `src/fplh/models/shrinkage.py:38`:

```python
mu = d["c"].sum() / max(d["e"].sum(), 1e-9)  # group mean rate
big = d[d["e"] >= 5]  # players with ≥ 5 90s
rates = big["c"] / big["e"]
between = rates.var() - (mu / big["e"]).mean()  # observed spread − Poisson noise
alpha = float(np.clip(mu**2 / max(between, 1e-6 * mu**2), 0.5, 200.0))
...
out[d.index] = (alpha + d["c"]) / (alpha / mu + d["e"])
```

The logic:

1. The variance of the observed rates S/u across players has two sources: real differences between players, and Poisson noise. For a player with exposure u the noise variance is ≈ μ/u.
2. So **between-player variance ≈ observed variance − mean noise variance.**
3. The Gamma prior's variance is μ²/α, so **α = μ² / between**.
4. Guard rails: α is clipped to [0.5, 200]. With fewer than 5 well-observed players in a group, a default of α = 5μ is used, which is 5 matches of pseudo-exposure.

If players in a group truly differ a lot, "between" is large, α is small and the prior is weak. If they are nearly identical, α is large and everyone is pulled hard to the mean.

## 4. Groups: who to shrink toward

Shrinking a centre-back toward the league average would be silly. `player_groups` (`shrinkage.py:64`) assigns each player to **FPL position × Understat role** (CB, FB, DM, CM, AM, W, FW…), and every component model shrinks within those groups. This is **partial pooling**:

- *No pooling:* every player alone. Too noisy.
- *Complete pooling:* everyone gets the group mean. Ignores real skill.
- *Partial pooling:* the posterior weighted average, decided by how much data each player has.

## 5. Old data counts less: time decay

A player's form two seasons ago matters less than last month's. Each row gets a weight

$$w_m = 2^{-\text{age}_m / h_{1/2}}$$

with half-life h = 365 days (`shrinkage.decay`, `shrinkage.py:58`). A match a year old counts half, and two years old a quarter. Weighted counts Σ w S and weighted exposures Σ w u go straight into the Gamma–Poisson formula. Decay also *shrinks the effective sample*, so it interacts with shrinkage: old evidence fades and the prior regains influence.

## 6. Pseudo-count shrinkage for ratios

The attack model (`src/fplh/models/attack.py`) shrinks two more quantities with the same idea in its simplest form, adding pseudo-observations at the target:

- **Shot quality** (xG per shot), with 20 pseudo-shots at the group's quality:

  $$\bar q_p = \frac{\text{xG}_p + 20\,\bar q_{g}}{\text{shots}_p + 20}$$

- **Finishing** (goals per xG), with 30 pseudo-xG at **1.0**, meaning "finishes exactly as xG says":

  $$f_p = \frac{G_p + 30}{\text{xG}_p + 30}$$

  Finishing skill is real but weak, and needs years of data to detect, so most players stay close to f = 1. A player with 12 goals from 8 xG gets f = 42/38 = 1.105, not 1.5.

The player's goal rate is then shot rate × quality × finishing.

## 7. The evidence: shrinkage works

Ablation A5 (`IMPLEMENTATION_PLAN.md` §4.4) compares non-penalty goals per appearance, by Poisson log loss (lower is better):

| Players | Shrunk | Raw decayed per-90 | Difference (95 % CI) |
|---|---|---|---|
| All (34,296) | 0.2692 | 0.3488 | −0.081 [−0.092, −0.070] |
| Under 10 decayed 90s of history (8,551) | 0.2197 | 0.4494 | **−0.231** [−0.270, −0.191] |

The gain is largest exactly where the data are thinnest. In the full simulator, raw rates cost +0.016 MSE (CI excluding 0).

## 8. Bayes elsewhere in the repo

- **M1** (Module 10) is Bayesian updating with Gaussian priors. Each match turns the prior on ratings into a posterior, which becomes the next prior.
- **M10 bonus** shrinks each player's residual BPS with n₀ = σ²/τ² pseudo-matches. It is the same formula with a Normal model.
- **M8 saves** shrinks the keeper multiplier g_k toward 1. Its spread ends up about ±2 %, because the data cannot tell keepers apart much once the opponent is known.
- **Uncertainty:** the posterior is a distribution, not just a mean. ARCHITECTURE §7.12 plans to propagate it (epistemic uncertainty), but the A8 ablation that needs posterior draws is not built yet.

## Check yourself

1. With μ = 0.3 goals/90 and α = 3, what is the pseudo-exposure? What is the posterior mean for a player with 2 goals in 2 matches?
   <details><summary>Answer</summary>α/μ = 10 matches of pseudo-exposure. The posterior mean is (3 + 2)/(10 + 2) = 0.417, against a raw rate of 1.0.</details>
2. A group's observed per-player rate variance is 0.50 and the mean Poisson noise term μ/u is 0.30, with μ = 2. What α does the method of moments give?
   <details><summary>Answer</summary>between = 0.50 − 0.30 = 0.20, so α = μ²/between = 4/0.2 = 20.</details>
3. What does the finishing estimate become for a player with 0 goals from 5 xG?
   <details><summary>Answer</summary>(0 + 30)/(5 + 30) = 0.857. Bad luck in a small sample only dents the estimate.</details>

## Practical

```bash
uv run python docs/learn/exercises/ex04_shrinkage.py
```

You will derive the posterior mean, implement the method-of-moments α, reproduce `fplh.models.shrinkage.gamma_poisson`, and measure on simulated players how much shrinkage beats raw rates.

Quiz: `uv run python docs/learn/quiz.py take 04`
