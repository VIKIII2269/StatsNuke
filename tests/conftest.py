from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from fplh.lake.storage import Lake
from fplh.rules import Rules, load_rules

TESTS = Path(__file__).parent
SAMPLES = TESTS / "contracts" / "samples"


@pytest.fixture(scope="session")
def rules_2526() -> Rules:
    return load_rules("2025/26")


@pytest.fixture(scope="session")
def rules_2627() -> Rules:
    return load_rules("2026/27")


@pytest.fixture
def lake(tmp_path: Path) -> Lake:
    return Lake(str(tmp_path / "lake"))


def load_sample(name: str) -> Any:
    return json.loads((SAMPLES / name).read_text())
