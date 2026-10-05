# 12 · The in-match goal process (G4–G6) and the emulator

> **Goal:** model *when* goals happen inside a match, not just how many. This gives the simulator real game dynamics: trailing teams pushing, red cards, stoppage time. It also gives you three of the most useful tools in applied statistics: hazard models, the piecewise-exponential ↔ Poisson-GLM equivalence, and emulators for expensive simulators.

## 1. Why a process, when G0 already works

Module 09 showed that pre-match scorelines are Poisson enough. But FPL needs more than final scores:

- **Who is on the pitch when a goal happens** decides who gets goals, assists and clean sheets. A defender subbed off at 60' keeps his clean sheet if the goal came at 70'.
- **Red cards** change both teams' scoring rates for the rest of the match.
- **Game state** changes behaviour: leaders sit back and trailers push.

So the simulator needs goals with *minutes attached*. ARCHITECTURE §7.5 lists the regularities a process can capture: excess draws, trailing teams scoring more late on, and red-card effects.

## 2. Intensity: goals per minute

A **hazard** or **intensity** λ(t) is an instantaneous rate: expected events per minute at time t. Over a short slot of length Δ, P(goal) ≈ λ(t)Δ. The repo's model (`src/fplh/models/goal_process.py`, docstring lines 3–17) for side k in minute t:

$$\lambda_k(t) = \frac{\mu_k}{90}\; g(t)\; \exp\!\Big(\beta\big(\Delta_k(t), b(t)\big) + \beta_{\text{own}}R_k(t) + \beta_{\text{opp}}R_{-k}(t)\Big)\;\varepsilon$$

| Piece | Meaning |
|---|---|
| μ_k/90 | Pre-match expected goals, spread evenly over 90 minutes (M1 when fitting, the emulator's nominal rate when simulating) |
| g(t) | Time profile: 18 five-minute bins plus a stoppage bin (log scale, smoothed by a random-walk penalty) |
| β(Δ, b) | Game state: goal difference Δ ∈ {−2..2} (Δ = 0 is the reference) × time bucket (0–30, 30–60, 60–75, 75+) |
| R | Red cards shown to each side |
| ε | Match "frailty" shared by both sides: an open or closed game |

The state Δ_k(t) must be the score **before** minute t. In `build_cells` that is `np.cumsum(goals, axis=2) - goals` (line 130, Module 01).

## 3. Piecewise exponential = Poisson GLM

How do you fit an intensity? The key fact:

> If the hazard is constant within each small cell (match m, side k, slot b) with exposure E and count N, the likelihood of the event times is proportional to
> $$\prod_{\text{cells}} \text{Pois}\big(N \mid E\cdot\lambda\big)$$
> So you can fit it as a **Poisson regression on cell counts**, with log E as an offset.

Each match becomes up to 2 × 105 rows (one per side per minute slot, including possible stoppage). The linear predictor of each row is

$$\log\lambda = \underbrace{\log E + \log(\mu/90)}_{\text{offset (known)}} + \log g_{\text{bin}} + \beta_{\Delta,b} + \beta_{\text{own}}R + \beta_{\text{opp}}R'$$

That is `_Fit.eta` (line 208). The gradient for each parameter is "observed − expected" summed over the cells that use it, via `np.bincount` (lines 251–259, Module 03). This is why fits take 0.4–2 s on hundreds of thousands of cells.

**Second-half stoppage** is handled by exposure. A stoppage slot 90 + u is played only in matches that reach it, with survival probability S(u), estimated from shot counts in those slots relative to minutes 80–89 (`estimate_stoppage`, line 73). Its exposure is S(u), not 1.

## 4. Frailty that integrates out

If both teams' intensities share a factor ε ~ Gamma(a, a) (mean 1, variance 1/a), then given ε the match's goals are Poisson, and integrating ε out gives a closed form. With N_m goals in match m and Λ_m = Σ λ over its cells:

$$\log p(\text{match } m) = \sum y\,\eta \;+\; \log\Gamma(N_m+a) - \log\Gamma(a) + a\log a - (N_m+a)\log(a+\Lambda_m)$$

That is lines 229–234. The gradient picks up a per-match weight w = (N_m + a)/(a + Λ_m) (line 235), which is the **posterior mean of ε given the match's goals**. A match with more goals than expected is inferred to have been "open", and every cell in it is reweighted. The spec asked for a log-normal frailty; the code uses a Gamma because it integrates exactly. Same role, closed-form likelihood.

**Dispersion.** Frailty makes goals overdispersed, with Var(N) = μ + μ²/a. Game-state damping (leaders slowing down) makes them *under*dispersed. The data decide. Fitted on 2015/16–2021/22: **frailty variance ≈ 0**, and Pearson φ̂ = 0.77–1.11 per season, below 1 in 5 of 7 seasons. EPL goals, once strength is known, are not overdispersed. That echoes G3's ν = 1.018 from a completely different angle.

## 5. The ladder G4 → G5 → G6

| Level | Adds | Why |
|---|---|---|
| G4 | Time profile g + game-state β | Basic dynamics |
| G5 | + red cards (own/opponent effects), frailty, a red-card hazard | Reds change everything after them |
| G6 | Game-state β estimated from **non-penalty shots** (~10× the events of goals) + a shrunk per-state conversion offset | Goals are rare, so state effects from goals alone are noisy; shots respond to game state in the same direction |

Fitted values:

- Your own red multiplies your scoring rate by e^−0.55 ≈ **0.58**; an opponent's red by e^0.60 ≈ **1.82**.
- The red-card hazard rises through the match.
- Game-state effects are small: trailing by one after 75', a team scores about 10–15 % more.
- The time profile rises through the match, with a bump at minutes 45–49, where first-half stoppage time is recorded.

**Evaluation** (scoreline log loss, 1,140 fixtures): G0 3.0007, G4 3.0009, G5 3.0017, G6 3.0010. **No step is significant.** Pre-match, the process adds nothing over G0, so G0 stays the pre-match default. But the *simulator* needs in-match dynamics for player events, so it uses the highest level not significantly worse than its predecessor: **G6**. That is a sensible rule for picking a model when the metric can't distinguish them.

## 6. Simulating the process

`src/fplh/sim/team.py` steps through 105 one-minute slots for **all fixtures × simulations at once** (shape (F, S)):

```python
for t in range(SLOTS):
    for k in (0, 1):
        delta = goals[k] - goals[1 - k]  # state before this minute
        eta = intensity.log_lambda(log_nom[:, k], t, delta, reds[k], reds[1 - k])
        lam = np.exp(eta) * eps[None, :] * active  # active: match still running
        new_goals.append(_poisson(lam, u_goal[k][None, :]))  # inverse-CDF Poisson draw
```

Two details:

- `_poisson` (line 52) draws Poisson counts by inverting the CDF with a uniform u. λ per minute is tiny, so counts above 4 are negligible.
- **Common random numbers:** uniforms are drawn per slot for the *simulation axis only* and shared across fixtures (docstring, lines 5–7). Neighbouring input rates then give smoothly varying outputs, which the emulator needs.

## 7. The emulator: making a slow simulator invertible

To use market prices (Module 08) with G6, you need the inverse map: "which nominal rates give mean goals (1.5, 1.1)?" Nominal ≠ mean, because game state, the time profile and stoppage all bend the totals. Simulating inside an optimiser loop would be far too slow. So `src/fplh/sim/emulator.py` builds an **emulator**, a cheap smooth surrogate:

1. **Grid:** a 20 × 20 log-spaced grid of nominal (home, away) rates. Run 10⁵ team-only simulations at each point, with the same seed at every point (common random numbers).
2. **Smooth:** fit a bicubic spline (`RectBivariateSpline`) per output: mean goals, and each scoreline cell. The smoothing `s` is set to the Monte Carlo variance Σ z(1 − z)/n (line 83), so the spline smooths out noise and nothing more.
3. **Invert:** 2-D **Newton's method** on the spline, using its analytic derivatives (lines 99–108), to find the nominal rates whose mean goals hit the target.
4. **Plug in:** `as_goal_model()` wraps it as a `GoalModel`, so inversion and fusion use it unchanged (Module 09).
5. **Cache:** emulators are stored in gold, keyed by a hash of the parameters (`params_sha`, line 37), and rebuilt only when the goal process changes.

The validation numbers:

| Check | Result | Target |
|---|---|---|
| Emulator vs 2·10⁵-draw direct simulation | max abs error 0.0050, mean 0.0013 | ≤ 0.005 |
| Draw rate, observed − predicted | +0.010, CI [−0.013, 0.032] | CI covers 0 |
| Scoreline cells up to 4–4 inside their CIs | 25 / 25 | ≥ 90 % |
| In-play next-goal ECE (home / away / none) | 0.015 / **0.024** / 0.014 | ≤ 0.02 |

The away next-goal miss is **reported, investigated and left open**:

- A perfectly calibrated forecaster would score ≤ 0.018 at the 95th percentile, so the miss is real, not noise.
- The training residuals show no missing term.
- The conclusion is a shift in the tuning seasons: away teams score relatively more late on.

An honest "known issue" is worth more than a hidden one.

## Check yourself

1. A side's base intensity is 1.35/90 per minute and it trails by one after 75', with β = 0.14. What is its intensity in that state?
   <details><summary>Answer</summary>0.015 × e^0.14 ≈ 0.0173 goals per minute, about 15 % more.</details>
2. Why does fitting a piecewise-constant hazard reduce to a Poisson GLM with an offset?
   <details><summary>Answer</summary>With constant hazard λ in a cell of exposure E, the likelihood of the event times in it is λ^N e^(−λE). As a function of the parameters that is the Poisson likelihood of N with mean λE. Summing over cells gives a Poisson regression where log E enters with coefficient 1 (an offset).</details>
3. Why use an emulator rather than simulating inside the market-inversion optimiser?
   <details><summary>Answer</summary>Each simulation is noisy and slow. An optimiser needs many smooth, cheap evaluations (and derivatives). The emulator precomputes a grid once with common random numbers and gives a smooth, differentiable surrogate.</details>

## Practical

```bash
uv run python docs/learn/exercises/ex12_goal_process.py
```

You will compute match-level probabilities from piecewise hazards, estimate a "trailing" effect with the closed-form Poisson-GLM MLE on simulated matches, check the Gamma-frailty marginal likelihood against numerical integration, and use the repo's emulator to hit target mean goals.

Quiz: `uv run python docs/learn/quiz.py take 12`
