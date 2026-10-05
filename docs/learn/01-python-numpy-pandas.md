# 01 · Python, NumPy and pandas the way this repo uses them

> **Goal:** read any file in `src/fplh/` without getting stuck on the Python. The models are mostly *array programs*: whole simulations advance in one line of NumPy. Learn the dozen idioms below and the code opens up.

## 1. Why arrays instead of loops

The player simulator plays 10 fixtures × 5,000 simulations × about 30 players × 95 minutes. That is 142 million player-minutes. A Python `for` loop runs at roughly 10–50 million simple steps per second, so a loop over every player-minute would take seconds to minutes *per gameweek*, and walk-forward evaluation repeats it thousands of times.

NumPy runs the same arithmetic in compiled C over whole arrays at once. The plan reports that 10 fixtures × 5,000 simulations with every component take **3.8 s**. The trick is to hold the state of *all simulations at once* in an array of shape `(simulations, players)` and update it with array operations:

```python
# one minute of the match for every simulation and every player at once
on_pitch = (on <= t) & (off > t)  # bool array (S, P)
minutes += on_pitch  # adds 1 where True
```

That is the meaning of "vectorised" in the docstrings.

## 2. Arrays: shape, dtype, axis

```python
import numpy as np

x = np.array([[1, 2, 3], [4, 5, 6]])  # shape (2, 3): 2 rows, 3 columns
x.dtype  # int64
x.sum(axis=0)  # [5, 7, 9]   collapse rows    → one value per column
x.sum(axis=1)  # [6, 15]     collapse columns → one value per row
x.sum(axis=1, keepdims=True)  # [[6], [15]]  shape (2, 1): stays 2-D for broadcasting
```

Rule of thumb: **`axis=k` is the axis that disappears.** `devig_multiplicative` (`src/fplh/models/market.py:24`) uses exactly this to normalise each row of odds:

```python
implied = 1.0 / np.asarray(prices, dtype=float)
out = implied / implied.sum(axis=-1, keepdims=True)  # axis=-1: the last axis
```

`keepdims=True` keeps the summed axis as size 1, so the division broadcasts row by row.

## 3. Broadcasting: the most important idea

When two arrays of different shapes meet, NumPy compares shapes **from the right**. Each pair of dimensions must be equal, or one of them must be 1, and a 1 is stretched to match.

```text
(3, 1)  with  (1, 4)  →  (3, 4)
(5, 2, 1) with (95,)  →  (5, 2, 95)
(3,)    with  (4,)    →  error
```

The scoreline grid in `src/fplh/models/goal_benchmarks.py:38` is a broadcasting outer product:

```python
K = np.arange(11)  # 0..10 goals
grid = np.outer(poisson.pmf(K, lh), poisson.pmf(K, la))
# same as: poisson.pmf(K, lh)[:, None] * poisson.pmf(K, la)[None, :]
#          shape (11, 1) * (1, 11) → (11, 11);  grid[i, j] = P(home i) · P(away j)
```

`[:, None]` adds an axis: `None` is shorthand for `np.newaxis`. You will see it constantly, for example `k[:, None] + k[None, :]` builds the "total goals" table in `markets()`.

## 4. Boolean masks and `np.where`

```python
pos = np.array(["GK", "DEF", "MID", "DEF"])
mask = pos == "DEF"  # [False, True, False, True]
goals_pts = np.where(mask, 6, 5)  # [5, 6, 5, 6]: if-else for every element
```

`rules/engine.py:125` scores appearance points for every player in every simulation without a loop:

```python
np.where(minutes >= 60, rules.appearance.gte_60, np.where(played, rules.appearance.lt_60, 0))
```

## 5. Cumulative sums and "state before this minute"

The goal process needs the score *before* each minute. `src/fplh/models/goal_process.py:130`:

```python
before = np.cumsum(goals, axis=2) - goals  # goals strictly before each slot
```

`cumsum` gives goals *up to and including* each minute. Subtracting the minute's own goals gives the state *before* it. Off-by-one errors in time are leakage at the smallest scale, so this line matters.

## 6. Scatter-add with `np.add.at`

To count events into bins, `arr[idx] += 1` is **wrong** when an index repeats: NumPy buffers the update and each index is incremented only once. `np.add.at` is the unbuffered version:

```python
arr = np.zeros(5, dtype=int)
idx = np.array([1, 1, 3])
arr[idx] += 1  # arr = [0, 1, 0, 1, 0]  ← minute 1 counted once!
np.add.at(arr, idx, 1)  # arr = [0, 2, 0, 1, 0]  ← correct
```

`build_cells` (`goal_process.py:122`) uses `np.add.at(arr, (m, s, t), 1)` to count goals per (match, side, minute).

## 7. Ranks with argsort-of-argsort

`np.argsort(x)` returns the *indices that would sort x*. Applying argsort **twice** gives each element's **rank**:

```python
x = np.array([30, 10, 20])
np.argsort(x)  # [1, 2, 0]: the smallest is at index 1, then 2, then 0
np.argsort(np.argsort(x))  # [2, 0, 1]: x[0] has rank 2, x[1] rank 0, x[2] rank 1
```

`_lineups` in `src/fplh/sim/simulator.py:219` uses this to keep only the earliest `limit` substitutions in each simulation, for all simulations at once.

## 8. `take_along_axis` and fancy indexing

```python
p = np.array([[0.5, 0.3, 0.2], [0.1, 0.6, 0.3]])
y = np.array([0, 2])  # observed class per row
p[np.arange(len(y)), y]  # [0.5, 0.3]: P(observed) per row
```

This one line is the heart of `log_loss` (`src/fplh/evaluate/metrics.py:19`). `np.take_along_axis(a, idx, axis=1)` does the same with an index *array* per row.

## 9. Random numbers you can reproduce

```python
rng = np.random.default_rng(seed)  # a Generator, not the global np.random
rng.random((S, P))  # uniforms
rng.poisson(lam, size=S)
```

The simulator seeds one generator per fixture from `(seed, crc32(fixture_uid))` (`simulator.py:257`). The same inputs therefore give **identical** results: replays are exact, tests are deterministic, and two decisions can be compared on the *same* random draws (common random numbers, Module 15).

## 10. pandas: the table toolkit

| Idiom | Example in repo | Meaning |
|---|---|---|
| `df[df["observed_at"] <= D]` | `information_set.py:127` | Filter rows (a boolean mask on a column) |
| `df.groupby("g")["x"].mean()` | `metrics.py:156` `per_block_diff` | Split → apply → combine |
| `df.merge(other, on=keys, how="left")` | `leakage.py:56` | SQL-style join |
| `df.sort_values(...).drop_duplicates(..., keep="last")` | `shrinkage.py:70` | Latest row per key |
| `s.rank()`, `s.corr()` | `metrics.py:107` | Spearman = correlation of ranks |

**A warning learned the hard way** (`IMPLEMENTATION_PLAN.md` §4.4): a day count divided raw timestamps by nanoseconds per day, while some tables stored microseconds. Walk-forward P(start) collapsed from 0.30 to 0.14. Always convert times with `.dt.total_seconds()` (as `shrinkage.decay` does), never with raw integers.

## 11. Dataclasses, Protocols and type hints

**Dataclasses** bundle data with less boilerplate:

```python
@dataclass(frozen=True)  # frozen: immutable, hashable, safe to share
class TeamStrengthParams:
    eta: float = 0.20
    phi_year: float = 0.80

    def with_(self, **kw):
        return replace(self, **kw)  # a modified copy
```

Parameters are frozen so that nothing can change them mid-run. That matters when a manifest hashes them to identify a run.

**Protocols** are interfaces checked by type, not inheritance. `src/fplh/evaluate/walk_forward.py:26`:

```python
class Predictor(Protocol):
    name: str
    version: str

    def predict(self, info: InformationSet, spine: pd.DataFrame) -> pd.DataFrame: ...
```

Any object with those attributes can be walk-forward evaluated: the A0 baseline, the OpenFPL replica and the simulator alike. This is how one harness evaluates every model on equal terms.

**Type hints** plus `mypy --strict` (see `pyproject.toml`) catch whole classes of bugs before running. `npt.NDArray[np.float64]` says "a float array".

## 12. Why the repo looks like this

- **Pure functions where possible.** `score_arrays(events, position, rules)` has no hidden state, so the simulator, the golden tests and the CLI all share one implementation.
- **Shapes in comments.** `# (S, P)` and `# (F, 2, 1)` everywhere. When you write array code, annotate shapes; it is the cheapest debugger there is.
- **Single-threaded where it pays.** The M1 filter does thousands of *tiny* matrix operations, and BLAS threads only contend. One pass took 76 s with 4 threads and 2.4 s with 1, so `team_strength.py:327` forces one thread. Vectorise the big things and keep the small things simple.

## Check yourself

1. What is the shape of `a[:, None] * b[None, :]` if `a` has shape (11,) and `b` has shape (11,)?
   <details><summary>Answer</summary>(11, 11). The first becomes (11, 1), the second (1, 11), and broadcasting stretches both to (11, 11).</details>
2. Why does `arr[idx] += 1` undercount when `idx` has duplicates?
   <details><summary>Answer</summary>The fancy-indexed update is buffered: NumPy reads the values, adds 1, and writes them back once per position, so a repeated index is written twice with the same value. Use np.add.at.</details>
3. Why are parameter classes `frozen=True`?
   <details><summary>Answer</summary>So they cannot be mutated during a run. That keeps results reproducible, makes the objects hashable, and makes a manifest's description of the run trustworthy.</details>

## Practical

```bash
uv run python docs/learn/exercises/ex01_numpy.py
```

You will implement the outer-product grid, ranks by double argsort, scatter-add counting and a pandas latest-row-per-key, each checked against the repo's own patterns.

Then take the quiz: `uv run python docs/learn/quiz.py take 01`
