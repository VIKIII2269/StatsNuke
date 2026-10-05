"""Exercise 18: content-derived run ids, stable hashes, backoff, determinism checks.

Run: uv run python docs/learn/exercises/ex18_engineering.py
"""

from __future__ import annotations

import hashlib  # noqa: F401  (you will need it)
import json  # noqa: F401
import random
from dataclasses import asdict

import numpy as np
from _check import run, task

from fplh.evaluate.manifest import Manifest

# ------------------------------------------------------------------ demo
M = Manifest(
    predictor="simulator",
    predictor_version="7",
    deadlines=["2024-08-16T17:30:00+00:00"],
    horizon=1,
    git_sha="abc123",
    data_manifest_sha256="d" * 64,
    config_sha256="c" * 64,
    rules_config="2024/25",
    seeds={"sim": 0},
)
print("run id:", M.run_id)


# ------------------------------------------------------------------ your tasks
def run_id(fields: dict[str, object]) -> str:
    """Like Manifest.run_id: drop 'max_observed_at' and 'extra', JSON-dump with sorted keys,
    SHA-256, first 12 hex chars, prefixed by '<predictor>_'."""
    raise NotImplementedError


def stable_hash(config: dict[str, object]) -> str:
    """SHA-256 hex of a config dict that does NOT depend on key insertion order."""
    raise NotImplementedError


def backoff(
    attempt: int, base: float = 1.0, cap: float = 60.0, rng: random.Random | None = None
) -> float:
    """Wait before retry `attempt` (1-based): min(cap, base·2^(attempt−1) + U(0, base))."""
    raise NotImplementedError


def is_deterministic(predict: object, inputs: np.ndarray, repeats: int = 3) -> bool:
    """Call predict(inputs) `repeats` times; True only if every output is identical."""
    raise NotImplementedError


@task("run_id reproduces Manifest.run_id, and changes when an input changes")
def _() -> None:
    fields = asdict(M)
    assert run_id(fields) == M.run_id
    assert run_id({**fields, "max_observed_at": "2099-01-01"}) == M.run_id  # excluded field
    assert run_id({**fields, "predictor_version": "8"}) != M.run_id


@task("stable_hash ignores key order but not values")
def _() -> None:
    a = {"x": 1, "y": [1, 2], "z": {"k": 0.5}}
    b = {"z": {"k": 0.5}, "y": [1, 2], "x": 1}
    assert stable_hash(a) == stable_hash(b)
    assert stable_hash(a) != stable_hash({**a, "x": 2})


@task("backoff grows exponentially, with jitter, up to the cap")
def _() -> None:
    r = random.Random(0)
    waits = [backoff(n, rng=r) for n in range(1, 9)]
    for n, w in enumerate(waits, 1):
        lo = min(60.0, 2 ** (n - 1))
        assert lo <= w <= min(60.0, lo + 1.0), (n, w)
    assert waits[-1] == 60.0


@task("is_deterministic catches an unseeded predictor")
def _() -> None:
    x = np.arange(5.0)
    seeded = lambda v: v + np.random.default_rng(0).normal(size=v.shape)  # noqa: E731
    unseeded = lambda v: v + np.random.default_rng().normal(size=v.shape)  # noqa: E731
    assert is_deterministic(seeded, x)
    assert not is_deterministic(unseeded, x)


run(globals())
