# 17 · The season replay and the paper ledger

> **Goal:** judge forecasts by the decisions they lead to, the only thing that finally matters. You will learn why a season replay is the honest end-to-end test, how expected value, Kelly staking and closing-line value work, and how to read a *failed* gate.

## 1. Why replay a season?

Better MSE doesn't automatically mean more FPL points. The optimiser might exploit a forecaster's *errors*: an overconfident model makes it chase noise and take hits. The decision-level benchmark (ARCHITECTURE §11.3) **runs every forecaster through the same optimiser** and scores real points. `src/fplh/evaluate/replay.py`:

1. Every strategy starts at GW2 with a free £100m squad. GW1 has no deadline spine, so it is excluded for all.
2. At each deadline, build E[p, g] for the next 5 gameweeks from that strategy's forecasts (`expected_points`, line 117):
   - **native**: the simulator's own horizon-5 forecasts;
   - **repeat**: a horizon-1 forecaster's next-week per-fixture value, repeated for every later fixture of the player's club. Doubles count twice and blanks zero.
3. Solve `plan_week` (Module 16) with the season's rules and **that deadline's prices**: H = 5, δ = 0.9, β = 0.1, fixed *a priori*.
4. Apply transfers with the FPL sell-price formula. Score the week on **actual** points with official auto-subs (`rules/team.py`), minus 4 per hit.
5. Carry the bank, free transfers and chips forward. A free-hit squad reverts afterwards.

Prices and fixtures are public at the deadline. Actual points are used only to score the week *after* the decision, so the replay obeys the information-set rule too.

## 2. Results: the simulator wins, and the gate fails

| Strategy | 22/23 | 23/24 | 24/25 | Total | Hits | Transfers |
|---|---|---|---|---|---|---|
| **Simulator (horizon 5)** | 2,272 | **2,384** | **2,300** | **6,956** | **39** | 250 |
| Simulator (repeat) | **2,276** | 2,304 | 2,271 | 6,851 | 130 | 350 |
| OpenFPL replica (repeat) | 2,185 | 2,250 | 2,264 | 6,699 | 117 | 344 |
| A0 (repeat) | 2,050 | 1,958 | 2,212 | 6,220 | 100 | 312 |
| Last 5 (repeat) | 1,977 | 2,005 | 2,018 | 6,000 | 158 | 378 |

The gate compares the simulator with the strongest baseline, in points per gameweek over 110 gameweek blocks:

- **vs replica:** +2.34, 95 % CI **[−1.26, +6.05]**, DM p = 0.21. **FAIL.**
- vs A0: +6.69 [+3.11, +10.32]. vs last 5: +8.69 [+4.84, +12.66].

How to read it, using Modules 07 and 15:

- **The direction is consistent:** the simulator leads the replica in all three seasons.
- **The power is insufficient:** gameweek points have sd ≈ 15, so 110 gameweeks can detect about 4 points per gameweek, not 2. Phase 3's forecast edge was 0.8 % of MSE, so a small decision edge is exactly what to expect.
- **Forecasting ahead helps through discipline:** 39 hits vs 130, and 100 fewer transfers. The point gain over repeating (+0.95/GW, CI [−1.76, +3.51]) is not significant on its own.
- **Overconfidence costs:** A0 and last-5 expect their XI to score 85–93 but get 60. The optimiser chases their noise. Simulator and replica expect about 62 and get 65–67. **Calibration matters for decisions**, not just for metrics.

What could pass next time: better forecasts (team news, set pieces, props), more seasons (narrower CI), and a **paired replay on common squads**. That is variance reduction, like common random numbers (Module 15): today each strategy's squad drifts apart, so path dependence adds noise.

## 3. The paper ledger: comparing with bookmakers, on paper

> **Paper only.** Real-money gambling is prohibited in the operator's jurisdiction. `tests/unit/test_ledger.py` **fails the build** if any HTTP write verb or bookmaker order endpoint appears in `src/`. The HTTP client only issues GETs.

`src/fplh/delivery/ledger.py` compares the fused match probabilities p̂ at each deadline with the **best pre-match price o observable at that deadline**.

### Expected value

$$\text{EV} = \hat p\,o - 1$$

If p̂ = 0.50 and o = 2.20, then EV = 0.10: a 10 % expected return *if p̂ is right*. That "if" is the whole problem.

### Kelly staking (paper)

The Kelly fraction maximises long-run log growth of a bankroll:

$$f^* = \frac{\hat p\,o - 1}{o - 1}$$

Full Kelly is very aggressive when p̂ is uncertain, so the ledger uses **quarter Kelly** (κ = 0.25), bets only when EV > 3 %, and caps stakes at 5 % of bankroll (`paper_bets`, line 119). Example: p̂ = 0.5, o = 2.2: f* = 0.1/1.2 = 0.083, quarter Kelly = **2.1 %** of bankroll.

### Closing-line value: the real test

Profit over a few hundred bets is mostly luck. The better skill signal is **CLV**: did you get a better price than the *de-vigged closing* price, the market's final, most-informed estimate?

$$\text{CLV} = \frac{o_{\text{taken}}}{o^*_{\text{close}}} - 1,\qquad o^*_{\text{close}} = 1/p^{\text{fair}}_{\text{close}}$$

A forecaster with no edge has mean CLV ≈ 0. Consistently beating the close means your information was ahead of the market's.

### Result

On 499 paper bets over 2022/23–2024/25:

- **mean CLV −1.15 %**, CI [−2.16 %, −0.14 %], so **no edge**;
- ROI +2.3 %, which is noise at this sample size and *not* a criterion;
- consistent with Phase 2: the fused forecast ties the market.

**Research note (paper only).** A no-model strategy that compares soft-book prices against the *sharp* early price had +3.2 % CLV over nine seasons. The plan records it as a benchmark, with the caveat that real accounts get limited fast.

## 4. What this module teaches

1. **Evaluate at the level of the decision** when you can, because models can be "better" in ways decisions don't reward.
2. **Report failures plainly.** "Phase 4: the exit gate fails" sits in the README. That honesty is what makes every *passed* gate believable.
3. **Profit is a noisy metric.** CLV and calibrated probabilities are better evidence than P&L over small samples.

## Check yourself

1. p̂ = 0.40, best price 2.80. What are the EV and the quarter-Kelly stake fraction?
   <details><summary>Answer</summary>EV = 0.4 × 2.8 − 1 = 0.12. f* = 0.12/1.8 = 0.0667, so quarter Kelly is 1.67 % of bankroll.</details>
2. You took 2.10 and the de-vigged closing probability was 0.50. What is the CLV?
   <details><summary>Answer</summary>o*_close = 2.0, so CLV = 2.10/2.0 − 1 = +5 %.</details>
3. Why can the season replay rank strategies differently from MSE?
   <details><summary>Answer</summary>The optimiser reacts to forecast *errors*: overconfident forecasts cause hits and churn, and ranking and calibration matter more than average error. Decisions reward different properties than MSE does.</details>

## Practical

```bash
uv run python docs/learn/exercises/ex17_ledger.py
```

You will implement EV, quarter-Kelly with the cap, and CLV against the repo's ledger, and simulate a no-edge bettor to see that mean CLV ≈ 0 while ROI swings wildly.

Quiz: `uv run python docs/learn/quiz.py take 17`
