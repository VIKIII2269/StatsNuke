# 16 · Decisions: mixed-integer linear programming

> **Goal:** turn forecasts into the best legal FPL decisions over several gameweeks: squad, XI, captain, transfers and chips. You will learn linear and integer programming, the standard modelling tricks (implications, AND, big-M, flow, budgets), how solvers work, and how the repo makes a hard problem fast.

## 1. Why not just pick the top players?

FPL is a constrained puzzle:

- exactly 2/5/5/3 per position;
- at most 3 per club;
- a £100m budget, with sell prices below buy prices after a rise;
- a valid XI (1 GK, at least 3 DEF, 2 MID, 1 FWD), a captain in the XI;
- free transfers that bank up to a cap, hits of −4 beyond them;
- chips that can be used once per window.

Decisions interact *across weeks*: selling a player now affects the budget and squad for every later week. Greedy rules fail. An **optimiser** searches the whole space and proves the best answer.

## 2. Linear programming in five minutes

An **LP** chooses real numbers x to

$$\max\ c^\top x\quad \text{s.t.}\quad A x \le b,\ \ \ell \le x \le u$$

Every constraint is a half-space, so the feasible region is a convex polytope, and a linear objective is maximised at a **corner**. The simplex method walks from corner to corner; interior-point methods cut through the middle. Both solve problems with millions of variables quickly.

## 3. Integer programming: when choices are yes/no

"Pick player p" is a **binary** variable x_p ∈ {0, 1}. With integer variables the problem becomes a **MILP**, which is NP-hard in general. Solvers such as **HiGHS** (via `scipy.optimize.milp`, `src/fplh/optimize/milp.py:149`) use **branch and bound**:

1. Solve the **LP relaxation** (allow 0 ≤ x ≤ 1). Its value is an upper bound.
2. If a binary is fractional, say x = 0.6, **branch** into two subproblems, x = 0 and x = 1.
3. Any integer solution found is a lower bound. Prune branches whose relaxation can't beat it.
4. Stop when the **gap** between bound and best solution is small. `mip_rel_gap` is 5e-4 here, meaning within 0.05 % of optimal.

A model whose relaxation is close to the integer optimum (a "tight" formulation) solves fast. A weak one can stall, which matters in §7.

## 4. The modelling toolkit

Every FPL rule becomes linear inequalities on binaries. These patterns recur in all of operations research:

| Rule | Linear form | Code |
|---|---|---|
| Exactly n_q per position | Σ_{p∈q} x_p = n_q | line 316 |
| At most 3 per club | Σ_{p∈c} x_p ≤ 3 | line 321 |
| XI only from squad (**implication**) | y_p ≤ x_p | line 335 |
| Captain is in the XI | k_p ≤ y_p, and Σ k_p = 1 | lines 347, 351 |
| Triple-captain bonus = captain **AND** chip | kt ≤ k, kt ≤ tc (maximisation pushes kt up to min(k, tc)) | lines 349–350 |
| Squad **flow** over time | x_{p,t} = x_{p,t−1} + b_{p,t} − s_{p,t} | line 356 |
| Bank balance | bank_t = bank_{t−1} + Σ sell·s − Σ price·b ≥ 0 | line 363 |
| Hits (**big-M** switch) | h_t ≥ Σ b_t − F_t − M(w_t + f_t) | line 387 |

**AND of binaries.** z = a·b is not linear. If you *maximise* z, the constraints z ≤ a and z ≤ b suffice: z rises to 1 only when both are 1. To force it in both directions, add z ≥ a + b − 1.

**Big-M.** "Hits are waived in a wildcard week" uses a large constant M. When w = 1 the constraint becomes h ≥ (something) − M, which is always satisfied (switched off). When w = 0 it binds. M must be large enough to switch off but no larger, because a huge M weakens the LP relaxation. The code uses 15, the most transfers possible.

**Free-transfer banking**, F_{t+1} = min(cap, max(F_t − transfers, 0) + 1), has a min and a max, so it is linearised with one extra binary `over` per week (lines 397–418). Because the optimiser *wants* more free transfers, upper bounds are enough.

## 5. The objective

$$\max\ \sum_t \delta^{t-1}\Big[\sum_p E_{p,t}\,(y_{p,t} + k_{p,t} + kt_{p,t}) + \beta\sum_p E_{p,t}\,u_{p,t} + (1-\beta)\sum_p E_{p,t}\,z_{p,t} - 4h_t\Big] - \text{chip opportunity costs}$$

- **E_{p,t}** is expected points from the simulator (Module 15), summed over a player's fixtures in a double gameweek, 0 in a blank. Expectation's linearity (Module 02) is what makes a *linear* objective correct for expected team points.
- **δ = 0.9** discounts later weeks, which are more uncertain and can be re-planned.
- **β = 0.1** values bench players (u = squad but not XI) slightly, as a proxy for auto-substitution value.
- **Captain:** y + k counts the captain twice.
- A tiny **transfer penalty** (0.01) breaks ties against pointless like-for-like churn.

**Sell price** (lines 242–249). FPL gives you half of any rise, rounded down to £0.1m:

$$\sigma_p = c^{\text{buy}} + \Big\lfloor\frac{c^{\text{now}} - c^{\text{buy}}}{2}\Big\rfloor \quad\text{if risen, else } c^{\text{now}}$$

*Example:* bought at 7.5 (75), now 8.0 (80). You can sell at 75 + ⌊5/2⌋ = 77, i.e. £7.7m.

## 6. A solution pool by "no-good cuts"

The user should see near-equivalent alternatives. After each solve, the code adds a constraint that forbids *this* combination of first-week decisions (lines 445–447):

$$\sum_{j\in\text{ones}} v_j - \sum_{j\in\text{zeros}} v_j \le |\text{ones}| - 1$$

and solves again, `top_k` times.

## 7. Making it fast: pools and decomposition

- **Player pool** (`pool`, line 186): only owned players, the top 25 per position by discounted horizon points, and the cheapest few per position (so a budget-feasible squad always exists). That is about 3,700 variables at H = 5, instead of variables for all ~700 players in the game.
- **Chips decomposed** (`plan_week`, line 451). The joint chip model's LP relaxation was weak: an **85 % gap after 30 s** on a real week, against about 1 s to optimality without chips. So `plan_week` works in steps:
  1. solve without chips;
  2. value bench boost and triple captain from that plan;
  3. value the free hit and wildcard by solves with them *forced* this week;
  4. play a chip only if this week is its best in the horizon *and* its gain beats an **opportunity cost** for keeping it (wildcard 20, free hit and bench boost 15, triple captain 10).

  This trades a little optimality for a lot of speed, a common and honest engineering compromise.
- **Correctness:** tests compare against **brute force** on toy instances (`brute_force_single_week`, line 603) and have a violating input for every constraint (`tests/unit/test_milp.py`).

## 8. Limits of the expected-value objective

A linear EV objective ignores variance and correlation: two forwards from the same team are a correlated bet. ARCHITECTURE §10.2 plans a **sample average approximation** over simulated scenarios, a rank-aware objective v = E(1 − EO) for mini-leagues, and a CVaR risk term. Those are Phase 5 extensions, gated by ablations.

## Check yourself

1. Write the constraint that the vice-captain must be in the XI, with v_p binary.
   <details><summary>Answer</summary>v_p ≤ y_p for every p, and Σ_p v_p = 1. To forbid the captain also being vice: v_p + k_p ≤ 1.</details>
2. You maximise and want z = a AND b. Which constraints suffice? What if you minimise z's coefficient?
   <details><summary>Answer</summary>When maximising a positive-coefficient z, z ≤ a and z ≤ b suffice. If z could be pushed down instead (negative coefficient), you also need z ≥ a + b − 1.</details>
3. Bought at 6.0, now 6.3. Sell price?
   <details><summary>Answer</summary>60 + ⌊3/2⌋ = 61, so £6.1m.</details>
4. Why does a huge big-M slow the solver?
   <details><summary>Answer</summary>It loosens the LP relaxation. Fractional values of the switch variable can almost disable the constraint, so the bounds are weak and branch and bound explores far more nodes.</details>

## Practical

```bash
uv run python docs/learn/exercises/ex16_milp.py
```

You will solve a tiny selection MILP with `scipy.optimize.milp` and check it against brute force, verify the AND linearisation, implement the sell-price and free-transfer rules (against the repo's), and run the repo's `optimise` on a toy game.

Quiz: `uv run python docs/learn/quiz.py take 16`
