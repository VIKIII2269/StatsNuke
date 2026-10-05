# 02 · Probability from zero

> **Goal:** be fluent with every distribution that appears in StatsNuke. Know what each one *models*, its mean and variance, and how it is parameterised in the code. Almost every model in the repo is "pick the right distribution, then estimate its parameters".

## 1. Random variables and distributions

A **random variable** is a quantity whose value is uncertain until it happens: the number of goals Arsenal score on Saturday, whether Saka starts, the minute of the first red card.

Its **distribution** says how likely each value is.

- For **discrete** variables (counts, yes/no) it is a **probability mass function** (pmf): P(X = k) for each k. The values are ≥ 0 and sum to 1.
- For **continuous** variables (xG, time) it is a **density** f(x). Probabilities are areas under it: P(a < X < b) = ∫ₐᵇ f(x) dx.
- The **cumulative distribution function** (CDF) F(x) = P(X ≤ x) works for both.

## 2. Expectation and variance

The **expected value** is the probability-weighted average:

$$\mathbb E[X] = \sum_k k\, P(X=k) \qquad\text{(discrete)}$$

**Variance** measures spread: $\text{Var}(X) = \mathbb E[(X - \mathbb E X)^2] = \mathbb E[X^2] - (\mathbb E X)^2$. The standard deviation is its square root.

Three rules you will use constantly:

1. **Linearity:** E[aX + bY] = a E[X] + b E[Y], *always*, even when X and Y are dependent. Expected FPL points are therefore the sum of expected points from each component. The MILP relies on this: a squad's expected points are a *sum* of player expected points (Module 16).
2. **Independent variances add:** Var(X + Y) = Var X + Var Y if X and Y are independent. Otherwise you add 2 Cov(X, Y). Teammates' points are positively correlated (a team goal can give one player a goal and another an assist), which is why the simulator keeps whole samples, not just means.
3. **Functions don't commute with E:** in general E[g(X)] ≠ g(E[X]). FPL is full of thresholds (clean sheet only if 0 conceded, defensive points only if ≥ 10 actions), so you need whole distributions, not just means.

## 3. The distributions in this repo

### Bernoulli and Binomial: yes/no events

A **Bernoulli(p)** variable is 1 with probability p and 0 otherwise: does the player start, does a shot go in. Its mean is p and its variance p(1 − p).

**Binomial(n, p)** counts successes in n independent tries:

$$P(X=k) = \binom{n}{k} p^k (1-p)^{n-k},\quad \mathbb E X = np,\quad \text{Var }X = np(1-p)$$

### Poisson: counts of rare events in a window

Goals in a match, shots in 90 minutes and saves are counts of events that each have a small chance in each small moment. Split 90 minutes into n tiny slices, each with scoring chance λ/n. As n → ∞ the Binomial(n, λ/n) becomes **Poisson(λ)**:

$$P(X = k) = \frac{e^{-\lambda}\lambda^k}{k!},\qquad \mathbb E X = \text{Var }X = \lambda$$

Worked example: a team expected to score λ = 1.5.

| Goals k | P(X = k) |
|---|---|
| 0 | e^(−1.5) = 0.2231 |
| 1 | 1.5 · e^(−1.5) = 0.3347 |
| 2 | 1.5²/2 · e^(−1.5) = 0.2510 |
| 3 | 1.5³/6 · e^(−1.5) = 0.1255 |

So the **opponent's clean-sheet chance** against this team is P(X = 0) = **22.3 %**, the probability that a defender playing the full match earns clean-sheet points.

The key property is **mean = variance**. Data with variance above the mean is *overdispersed*; below is *underdispersed*. Module 12 shows EPL goals, once team strength is known, are close to Poisson or even slightly *under*dispersed. That is why the simple G0 model is so hard to beat.

### Exponential: waiting times and hazards

If events arrive as a Poisson process with rate λ per minute, the **waiting time** to the next one is **Exponential(λ)**, with P(T > t) = e^(−λt). Its key property is that it is **memoryless**: having waited 20 minutes tells you nothing about the next 5.

The **hazard** h(t) is the instantaneous event rate at time t, given nothing has happened yet. A constant hazard gives an exponential wait. A *time-varying* hazard, such as scoring rates that rise through a match or red-card risk that grows when you are losing, gives a richer process. That is the G4–G6 model (Module 12). Over a short slot of length Δ, P(event) ≈ 1 − e^(−hΔ) ≈ hΔ.

### Normal: sums of many small effects

**Normal(μ, σ²)** has the bell-shaped density $f(x) = \frac{1}{\sqrt{2\pi}\sigma} e^{-(x-\mu)^2/2\sigma^2}$. Sums of many independent effects tend to a Normal (the central limit theorem). That is why bootstrap means and Diebold–Mariano statistics are Normal-ish (Module 07), and why team ratings drift by Normal steps (Module 10).

### Log-normal: positive quantities with multiplicative noise

If log Y ~ Normal(m, s²), then Y is **log-normal**, with

$$\mathbb E[Y] = e^{m + s^2/2}$$

Note the **+s²/2**: the mean of Y is *bigger* than e^m. M1 predicts log goal rates η with uncertainty s², so the expected goal rate is exp(η + s²/2) (`src/fplh/models/team_strength.py:203`). Rating uncertainty therefore *raises* expected goals slightly. That is the Jensen effect from rule 3 above.

### Gamma: positive rates and xG

**Gamma(shape α, rate β)** is a flexible distribution on positive numbers:

$$f(x) = \frac{\beta^\alpha}{\Gamma(\alpha)} x^{\alpha-1}e^{-\beta x},\qquad \mathbb E X = \frac{\alpha}{\beta},\qquad \text{Var }X = \frac{\alpha}{\beta^2}$$

The repo uses it in two ways:

- **As a likelihood for xG** in M1: X ~ Gamma(κ, κ/μ). Plugging in gives mean κ/(κ/μ) = μ and variance κ/(κ/μ)² = μ²/κ. So xG is centred on the true rate μ, and κ controls how noisy it is. The fitted κ is about 16 (`configs/models/team_strength.yaml`).
- **As a prior for rates** (Module 04): a player's true shot rate is unknown, so it gets a Gamma prior.

### Beta: probabilities themselves

**Beta(a, b)** lives on [0, 1] with mean a/(a + b). It is the natural prior for a probability such as penalty conversion. Each success adds to a and each failure adds to b.

### Negative binomial: overdispersed counts

Suppose a player's true defensive-action rate varies from match to match: a Gamma "frailty" that is sometimes busy, sometimes quiet. Given the rate the count is Poisson, but averaged over the rate the count is **negative binomial (NegBin)** with mean μ and size k:

$$\mathbb E C = \mu,\qquad \text{Var }C = \mu + \frac{\mu^2}{k}$$

As k → ∞ the extra variance vanishes and you are back to Poisson. M7 (defence) and M8 (saves) use NegBin, because their counts *are* overdispersed (fitted k ≈ 7.5 for defenders, 14 for saves).

In SciPy, `nbinom(n, p)` uses n = k and p = k/(k + μ). You will convert between the two forms in the practical.

## 4. Threshold events and the CDF

FPL rewards thresholds. A defender earns defensive-contribution points if clearances + blocks + interceptions + tackles **≥ 10**:

$$P(C \ge 10) = 1 - P(C \le 9) = 1 - F(9)$$

Because the reward depends on crossing a line, **variance matters as much as the mean.** Take two defenders who both average 7.5 actions. The one with higher variance (smaller k) crosses 10 more often. This is exactly why M7 models the full NegBin, not just the mean.

## 5. Logit and sigmoid: probabilities on an unbounded scale

A probability is stuck in [0, 1], but linear models produce any real number. The bridge is:

$$\text{logit}(p) = \log\frac{p}{1-p}\in(-\infty,\infty), \qquad \sigma(x) = \frac{1}{1+e^{-x}}\in(0,1)$$

These are inverses. Logistic regression, XGBoost's `binary:logistic`, fusion weights (`models/fusion.py:36`) and the substitution-era shift of M4 (`models/minutes.py:175`) all work on the logit scale. There, adding δ moves every probability in the same direction without leaving [0, 1].

Worked example: p = 0.2 has logit log(0.25) = −1.386. Add δ = 0.5 to get −0.886, and σ(−0.886) = 0.292.

## 6. Logs, because likelihoods multiply

Probabilities of independent events multiply, and products of thousands of small numbers underflow to 0 on a computer. Logs turn products into sums:

$$\log(ab) = \log a + \log b,\qquad \log(a^k) = k\log a,\qquad \log e^x = x$$

So the repo works with log probabilities everywhere: `poisson.logpmf`, `gammaln` (the log of the Gamma function, log k! = gammaln(k + 1)), and `log_frailty_shape`. When you see `np.log(np.clip(p, 1e-15, 1))` (`metrics.py:19`), the clip stops log(0) = −∞ from destroying an average.

## 7. Conditional probability (a first look)

P(A | B) is the probability of A *given* that B happened: P(A | B) = P(A and B) / P(B). The minutes model is built from conditionals:

- π^S = P(start);
- π^60 = P(60+ minutes | start);
- π^B = P(appears | not started).

So P(starts and plays 60+) = π^S · π^60, and P(appears) = π^S + (1 − π^S) π^B. That is the **law of total probability**, splitting on whether the player started.

## Check yourself

1. A team's expected goals are 2.0. What is P(they score 0)? What is P(their opponent keeps a clean sheet)?
   <details><summary>Answer</summary>Both are e^(−2) = 0.1353. A clean sheet for the opponent means this team scores 0.</details>
2. Gamma(κ, κ/μ) with κ = 16 and μ = 1.5: what are the mean and the standard deviation?
   <details><summary>Answer</summary>The mean is μ = 1.5. The variance is μ²/κ = 2.25/16 = 0.1406, so the sd is 0.375.</details>
3. π^S = 0.8 and π^60 = 0.9. What is P(60+ minutes)?
   <details><summary>Answer</summary>0.8 × 0.9 = 0.72, ignoring the rare substitute who comes on very early and still plays 60+ (the simulator allows that, because it draws real substitution minutes).</details>
4. Why does the M1 filter compute expected goals as exp(η + s²/2) rather than exp(η)?
   <details><summary>Answer</summary>η is uncertain, distributed Normal(η̂, s²), so the rate is log-normal, and the mean of a log-normal is e^(m + s²/2). Ignoring s² would under-predict goals for teams whose ratings are uncertain.</details>

## Practical

```bash
uv run python docs/learn/exercises/ex02_probability.py
```

You will implement the Poisson pmf from its formula, show Binomial → Poisson, convert NegBin (mean, size) to SciPy's form, compute threshold probabilities, and invert logit and sigmoid.

Quiz: `uv run python docs/learn/quiz.py take 02`
