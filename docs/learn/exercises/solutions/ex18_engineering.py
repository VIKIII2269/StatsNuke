"""Reference solutions for exercise 18."""

from __future__ import annotations

import hashlib
import json
import random

import numpy as np


def run_id(fields: dict[str, object]) -> str:
    payload = {k: v for k, v in fields.items() if k not in ("max_observed_at", "extra")}
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:12]
    return f"{fields['predictor']}_{digest}"


def stable_hash(config: dict[str, object]) -> str:
    return hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()


def backoff(
    attempt: int, base: float = 1.0, cap: float = 60.0, rng: random.Random | None = None
) -> float:
    r = rng or random.Random()
    return float(min(cap, base * 2 ** (attempt - 1) + r.uniform(0, base)))


def is_deterministic(predict: object, inputs: np.ndarray, repeats: int = 3) -> bool:
    outs = [np.asarray(predict(inputs)) for _ in range(repeats)]  # type: ignore[operator]
    return all(np.array_equal(outs[0], o) for o in outs[1:])
