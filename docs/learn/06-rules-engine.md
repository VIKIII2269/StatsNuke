# 06 · Rules as data: the scoring engine and its tests

> **Goal:** see how a small, *exactly correct* component anchors a big probabilistic system. The rules engine turns match events into FPL points. Everything downstream (simulated points, walk-forward errors, the season replay) is only as right as this function. It is also the cleanest example of testing in the repo.

## 1. The problem

FPL points are a deterministic function of what each player did. Some example rules for 2025/26:

| Event | Points |
|---|---|
| Play 1–59 min / 60+ min | 1 / 2 |
| Goal by GK / DEF / MID / FWD | 10 / 6 / 5 / 4 |
| Assist | 3 |
| Clean sheet (60+ min, 0 conceded *while on the pitch*): GK, DEF / MID | 4 / 1 |
| Every 2 goals conceded (GK, DEF) | −1 |
| Every 3 saves | +1 |
| Penalty save / miss | +5 / −2 |
| Yellow / red / own goal | −1 / −3 / −2 |
| Defensive contribution (DEF ≥ 10 CBI+T; MID/FWD ≥ 12 CBI+T+R) | +2 (capped at 2) |
| Bonus (top 3 BPS in the match) | 3 / 2 / 1 |

The rules **change between seasons**. Defensive contribution started in 2025/26, and the penalty-save BPS weight fell from 15 to about 8. A simulator scoring 2018/19 with 2025/26 rules would be quietly wrong.

## 2. Rules as data, not code

Each season's rules live in YAML (`configs/rules/fpl_2025_26.yaml`):

```yaml
goal:              {GK: 10, DEF: 6, MID: 5, FWD: 4}
clean_sheet:       {GK: 4, DEF: 4, MID: 1, FWD: 0, min_minutes: 60}
goals_conceded:    {GK: {per: 2, points: -1}, DEF: {per: 2, points: -1}}
defensive_contribution:
  DEF:     {actions: [clearance, block, interception, tackle], threshold: 10, points: 2}
```

`rules/config.py` parses it into **pydantic models with `extra="forbid"`** (line 23). A typo like `golas:` is an error, not a silently ignored key. Any value still marked with the `VERIFY` placeholder is rejected too. The schema makes wrong configs *unloadable*.

The benefits of rules as data:

- A new season is a new YAML file, not a code change.
- The diff between two seasons is readable by a non-programmer.
- The *same* engine serves 11 seasons, the simulator and the golden tests.

## 3. The engine: one pure, vectorised function

`score_arrays(events, position, rules)` in `src/fplh/rules/engine.py:81` takes **arrays of any shape**, for example (simulations × players), and returns points per component. Read lines 96–142 and notice:

- **No loops over players.** Every rule is a `np.where` or an integer division:

  ```python
  kept_clean = (minutes >= cs.min_minutes) & (conceded == 0) & played
  gc[mask] = (conceded[mask] // rule.per) * rule.points          # −1 per 2 conceded
  "saves": (col("saves") // saves.per) * saves.points,             # +1 per 3 saves
  ```

- **A breakdown, not just a total.** It returns `appearance`, `goals`, `clean_sheet`, … separately (`COMPONENTS`, line 37). Error attribution and debugging can then point to *which* rule disagrees.
- **Pure.** There is no I/O and no global state, so it is trivial to test and safe to call from anywhere.

The simulator calls exactly this function on its (S, P) event arrays (`sim/simulator.py:468`). Simulated points therefore follow the *official* rules by construction.

## 4. Bonus: ranking with ties

Bonus goes to the top three BPS in each match, with FPL's tie rule. `src/fplh/rules/bonus.py` uses **competition ranking**: rank = 1 + (number of players with *strictly* more BPS).

| BPS | Ranks | Bonus |
|---|---|---|
| 40, 40, 30 | 1, 1, 3 | 3, 3, 1 |
| 40, 35, 35 | 1, 2, 2 | 3, 2, 2 |
| 40, 35, 30, 30 | 1, 2, 3, 3 | 3, 2, 1, 1 |

The vectorised version (`assign_bonus_array`, line 35) computes all ranks in one broadcast:

```python
rank = 1 + (b[..., None, :] > b[..., :, None]).sum(axis=-1)
```

For each player i it counts the j with b_j > b_i. Players who did not play get −∞ BPS: they never rank and never push anyone down.

## 5. Team scoring: auto-subs and captaincy

`src/fplh/rules/team.py` scores a *manager's* gameweek. Starters with 0 minutes are replaced in bench order, if the formation stays valid (1 GK, minimum DEF/MID/FWD). The captain counts ×2 (×3 with triple captain); if the captain didn't play, the vice-captain inherits it. Bench boost scores all 15 players. The season replay (Module 17) uses this to score every strategy on actual points.

## 6. Golden tests: the definitive check

How do you know the YAML is right? **Feed the official per-player event stats into the engine and demand the official `total_points`, exactly**, for every player-fixture:

$$\mathcal R_{\text{season}}(\text{official events}) \stackrel{!}{=} \text{official total\_points}$$

The result was **0 mismatches on 254,119 player-fixtures** from 2016/17 to 2026/27 (`tests/golden/`, `rules/golden.py`). Any rule error, such as a wrong clean-sheet threshold, a missed cap or a wrong tie rule, shows up as mismatching rows. CI fetches full seasons and runs them.

Golden testing also *discovers* facts:

- **GK goal points** were identified as 6 up to 2024/25 from one real case (Alisson, 2020/21). The 10 used from 2025/26 comes from the rules page and is "not identified by data" until a goalkeeper scores. The YAML says so in a comment.
- **Penalty-save BPS** fell from 15 to about 8. Regressing BPS on events (`fplh rules check-bps`) found +8.9 in 2024/25 and +7.0 in 2025/26.
- **Our clean-sheet and defensive-count derivations** agree with FPL's own fields on every row. So the simulator can *derive* them from events, and never needs to read them.

`tests/golden/known_exceptions.yaml` exists for documented upstream data errors, never for convenience.

## 7. Property-based tests (Hypothesis)

Golden tests check real cases. **Property tests** check *invariants* on thousands of random cases (`tests/property/test_rules_properties.py`):

```python
@given(frames, frames)
def test_rows_score_independently(a, b):
    """Scoring a concatenation (e.g. both DGW fixtures) = concatenating the scores."""
```

Hypothesis generates random event tables within stated ranges. When a property fails, it *shrinks* the example to a minimal counterexample. Properties in the repo include:

- row order doesn't matter;
- rows score independently, so double gameweeks are a sum;
- bonus totals are right, including ties;
- in the simulator: player goals + opponents' own goals = team goals, exactly 11 starters, substitutions ≤ limit.

| Test kind | Question it answers | Example |
|---|---|---|
| Unit | Does this function do what I meant on a case I chose? | `test_engine.py` |
| Golden | Does it match reality exactly? | 254,119 official rows |
| Property | Does it hold for *every* input in a space? | Order independence |
| Contract | Does the API still send the shape we parse? | `tests/contracts/` with saved payloads |

## 8. Why build it this way

- **Exactness at the base.** Probabilistic layers already carry uncertainty. The rules layer must carry *none*, or its error hides inside model error forever.
- **One implementation, many callers.** The CLI, golden tests, simulator and replay all score through `score_arrays`, so there is nothing to keep in sync.
- **Data finds bugs that reading the rules doesn't.** The penalty-save BPS change was found by regression, not by documentation.

## Check yourself

1. Under 2025/26 rules, a midfielder plays 75 minutes, scores once, assists once, his team concedes 0 while he is on, and he gets 0 bonus. What are his points?
   <details><summary>Answer</summary>Appearance 2 + goal 5 + assist 3 + MID clean sheet 1 = 11.</details>
2. BPS 31, 31, 31, 20. What bonus does each player get?
   <details><summary>Answer</summary>The ranks are 1, 1, 1, 4, so the bonus is 3, 3, 3, 0. Three tied for first all get 3, and nobody gets 2 or 1.</details>
3. Why does the engine return a per-component breakdown?
   <details><summary>Answer</summary>To attribute errors and debug. A golden mismatch or a forecast error can be traced to a specific rule or event, such as clean sheets, rather than just "total differs".</details>

## Practical

```bash
uv run python docs/learn/exercises/ex06_rules.py
```

You will write a tiny scorer for a few rules and check it against `score_arrays`, implement competition ranking for bonus and verify it against `assign_bonus_array` on random cases, and write a property check of your own.

Quiz: `uv run python docs/learn/quiz.py take 06`
