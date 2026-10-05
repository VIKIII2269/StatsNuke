"""Tiny task checker shared by the exercise scripts.

Each exercise defines functions for you to implement (they start as ``raise NotImplementedError``)
and registers checks with ``@task``. ``run()`` prints one line per task:

    PASS <name>          your implementation is correct
    TODO <name>          not implemented yet
    FAIL <name>: <why>   implemented, but wrong

and exits 0 only when every task passes, which is what ``quiz.py practicals`` records.

Set ``LEARN_SOLUTIONS=1`` to run the reference solutions in ``solutions/`` instead of your code.
"""

from __future__ import annotations

import importlib.util
import os
import sys
import traceback
from collections.abc import Callable
from pathlib import Path
from types import ModuleType
from typing import Any

_TASKS: list[tuple[str, Callable[[], None]]] = []


def task(name: str) -> Callable[[Callable[[], None]], Callable[[], None]]:
    def register(fn: Callable[[], None]) -> Callable[[], None]:
        _TASKS.append((name, fn))
        return fn

    return register


def close(got: Any, want: Any, tol: float = 1e-6) -> None:
    """Assert numbers or arrays agree to ``tol`` (absolute) with a readable message."""
    import numpy as np

    g, w = np.asarray(got, dtype=float), np.asarray(want, dtype=float)
    if g.shape != w.shape:
        raise AssertionError(f"shape {g.shape} != expected {w.shape}")
    if not np.allclose(g, w, atol=tol, rtol=0):
        raise AssertionError(f"got {np.round(g, 6).tolist()}, expected {np.round(w, 6).tolist()}")


def _use_solutions(namespace: dict[str, Any]) -> None:
    script = Path(namespace["__file__"])
    path = script.parent / "solutions" / script.name
    spec = importlib.util.spec_from_file_location(f"solution_{script.stem}", path)
    if spec is None or spec.loader is None:
        raise FileNotFoundError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for name, value in vars(module).items():
        if not name.startswith("_") and name in namespace and not isinstance(value, ModuleType):
            namespace[name] = value


def run(namespace: dict[str, Any]) -> None:
    if os.environ.get("LEARN_SOLUTIONS") == "1":
        _use_solutions(namespace)
    passed = 0
    print()
    for name, fn in _TASKS:
        try:
            fn()
        except NotImplementedError:
            print(f"TODO {name}")
        except Exception as exc:  # report every other failure as a wrong answer
            where = traceback.extract_tb(exc.__traceback__)[-1]
            print(f"FAIL {name}: {type(exc).__name__}: {exc} (line {where.lineno})")
        else:
            print(f"PASS {name}")
            passed += 1
    print(f"\n{passed}/{len(_TASKS)} tasks pass")
    sys.exit(0 if passed == len(_TASKS) else 1)
