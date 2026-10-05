# 14 · Component models M5–M10

> **Goal:** see how each FPL scoring event gets its own small, well-targeted model: goals, assists, defensive actions, saves, cards and bonus. They all reuse the tools you already have (Gamma–Poisson shrinkage, NegBin thresholds, least squares). Each is judged by what it adds to the simulator, and each beats its baseline.

**The design principle** (ARCHITECTURE §7.1): *each component models one real quantity, with its own likelihood and metric; components meet only inside the simulator.* A small model is easy to validate, and when the simulator errs, the attribution tells you which component to fix.

## 1. M5/M6 attack: who scores and assists

`src/fplh/models/attack.py` builds, per player, from decayed Understat data (half-life 365 days):

| Quantity | Model | Prior / shrinkage |
|---|---|---|
| Shot rate r_p (non-penalty shots/90) | Gamma–Poisson | toward position × role group, α by method of moments (Module 04) |
| Shot quality q̄_p (xG per shot) | pseudo-counts | 20 pseudo-shots at the group's quality |
| Finishing f_p (goals per xG) | pseudo-counts | 30 pseudo-xG at **1.0** |
| Assist rate a_p (xA/90) | Gamma–Poisson | toward the group |
| Penalty weight | decayed penalty attempts | — |

The **goal rate** is r_p · q̄_p · f_p. That is three noisy quantities, each shrunk separately, then multiplied.

League constants (fitted at each deadline): penalties are 7.4 % of goals, own goals 3.3 %, penalty conversion 0.79, 0.021 missed penalties per side-match, and 88 % of goals have an FPL assist.

### Allocation keeps the totals coherent

The *team* process (Module 12) decides **when** a team scores. Players only *split* that goal:

$$P(\text{scorer} = p) = \frac{r_p\,\mathbb 1[p \text{ on pitch}]}{\sum_q r_q\,\mathbb 1[q \text{ on pitch}]}$$

So player goals always add up to team goals (property-tested). Separately estimated player goal models would not guarantee that. This is principle P3, coherence. Each goal is first classified (`simulator.py:333-335`):

- an **own goal** (3.3 %), credited to an opponent on the pitch, weighted toward defenders;
- a **penalty** (7.4 %), scored by the on-pitch taker with the most attempts;
- **open play**: scorer ∝ goal rate, then an assist with the league probability, assister ∝ assist rate among teammates.

**Evidence (A5):** shrunk vs raw goal rates, log loss 0.269 vs 0.349 overall and 0.220 vs 0.449 for players with little history. In the simulator, raw rates cost +0.016 MSE.

## 2. M7 defensive actions: a threshold problem

From 2025/26, DEF score 2 points for ≥ 10 clearances/blocks/interceptions + tackles, and MID/FWD for ≥ 12 including recoveries (Module 06). `src/fplh/models/defence.py`:

- **Per-action rates:** decayed Gamma–Poisson per 90, shrunk to position × role.
- **Opponent multiplier,** shrunk (±5 %): possession-heavy opponents force more defending.
- **Dispersion:** a Gamma frailty per player-match, **shared** by the actions (a busy match is busy for all of them). The group total is then NegBin with size k by position, fitted by method of moments: k ≈ 7.5 (DEF) and 9.5 (MID).
- **Definition drift:** FPL's counting changed between 2018/19 and 2025/26. Tackles per 90 doubled (×2.06) and recoveries fell to about 0.73×. Where an (action, position) rate differs by more than 10 % between eras, the old counts are rescaled by the ratio. *The definitions changed, not the football*: DEF CBI+T is about 7.5/90 in both eras.

The FPL outcome is a threshold: P(C ≥ T) = 1 − F_NB(T − 1) (Module 02). On threshold Brier vs the position-mean model, DEF in 2025/26 scored 0.1297 vs 0.1482, a difference of −0.019, CI [−0.022, −0.015].

*Data caveat:* these counts exist only for 2016/17–2018/19 and from 2025/26, so M7 is the **one documented exception** to the untouched 2025/26 holdout. It is evaluated within-season, walk-forward.

## 3. M8 saves

`src/fplh/models/gk.py`:

$$\text{saves} \sim \text{NegBin}\Big(\text{mean} = \frac{\text{minutes}}{90}\cdot g_k\cdot(a + b\,\mu_{\text{opp}}),\ \text{size} \approx 14\Big)$$

μ_opp is the *opponent's* pre-match expected goals (M1, point in time). The fit gives a ≈ 1.17 and b ≈ 1.24: more dangerous opponents force more saves.

Two findings:

- **Given μ_opp, the match's realised goals add nothing** (coefficient 0.04). So saves are drawn *independently* of simulated goals, which is simpler and just as good.
- **The keeper multiplier g_k barely matters.** Shrunk toward 1, its spread is about ±2 %. Log loss of save points: model 0.9683, opponent-only 0.9701, CI includes 0. The opponent carries the model.

Penalty saves: FPL counts saved penalties as missed, and about 75 % of missed penalties are saved, so the opposing keeper on the pitch gets that share.

## 4. M9 cards

`src/fplh/models/cards.py`: a decayed Gamma–Poisson **yellow-card rate per 90**, shrunk to position × role. In the simulator, P(yellow) = 1 − exp(−rate · minutes/90), at most one, and none with a red. A red card is given to a player ∝ yellow rate. Yellow vs position-rate Brier: 0.1121 vs 0.1127, a small but significant gain (CI [−0.0009, −0.0003]).

What it leaves out, deliberately:

- **No game-state term,** because yellow-card minutes are not in the data.
- **No referee term,** because the referee is announced too late for the deadline. That would be leakage.

## 5. M10 bonus

Bonus goes to the top 3 BPS in a match (Module 06), and BPS depends on *everyone's* events. `src/fplh/models/bonus.py`:

$$\text{BPS}_p = \sum_c w_c\,x_c(\text{events}) + \mu_g + \delta_p + \sigma_g\,\varepsilon$$

- x_c are the counts of events the simulator generates (minutes bands, goals by position, assists, clean sheets, saves, cards…).
- **Effective weights** w_c come from least squares of official BPS on x over the last two seasons. They differ from the official table because a goal *also* brings correlated actions the table scores separately, like shots on target. Example: a forward's goal is about 23 effective vs 24 in the table.
- μ_g and σ_g are the residual mean and spread by position × (60+ minutes).
- δ_p is the player's own shrunk mean residual, for passing and other actions not simulated. The shrinkage uses n₀ = σ²/τ² pseudo-matches (Module 04).

Then the **official allocation** runs over both sides' players who played (`assign_bonus_array`, eligible = minutes > 0).

| Target | Model | Official table, no residual | Position rate |
|---|---|---|---|
| P(bonus > 0), Brier | **0.0356** | 0.0695 | 0.0939 |
| E[bonus], squared error | **0.1428** | 0.2476 | 0.4464 |

## 6. How components are judged

`fplh evaluate components` scores each model **given the player's actual minutes**, and bonus given the match's actual events. So each one is judged on *what it adds to the simulator*, not on how well it guesses minutes, which is M4's job. Separation of concerns, in the evaluation too.

## Check yourself

1. Shot rate 3.0/90, xG per shot 0.12, finishing 1.05. What is the goal rate per 90?
   <details><summary>Answer</summary>3.0 × 0.12 × 1.05 = 0.378 non-penalty goals per 90.</details>
2. On the pitch: A (rate 0.40), B (0.20), C (0.05), plus eight players at 0.01. A team goal from open play: P(A scores)?
   <details><summary>Answer</summary>0.40 / (0.40 + 0.20 + 0.05 + 0.08) = 0.40/0.73 = 0.548.</details>
3. Why does the saves model use the opponent's *pre-match* expected goals and ignore the simulated goals?
   <details><summary>Answer</summary>Given μ_opp, the realised goals carried no extra information about saves (coefficient 0.04). Drawing saves independently is simpler, and μ_opp is point-in-time, so there is no leakage.</details>
4. Why are effective BPS weights different from the official table?
   <details><summary>Answer</summary>Events the simulator generates (a goal) come with correlated actions it doesn't generate (shots on target, key passes). Least squares absorbs their typical BPS into the effective weight.</details>

## Practical

```bash
uv run python docs/learn/exercises/ex14_components.py
```

You will combine attack components, allocate goals to the players on the pitch (checking coherence by simulation), compute defensive-threshold and save-point expectations with the repo's NegBin, and derive yellow-card probabilities.

Quiz: `uv run python docs/learn/quiz.py take 14`
