"""Run provenance (ARCHITECTURE.md §12.6).

A manifest pins everything a prediction depended on: code (git SHA), data (SHA-256 of
every silver part read), configuration, predictor version, deadlines and seeds. The
``run_id`` is derived from those inputs, so replaying the same inputs reproduces the
same run id and byte-identical outputs; nothing wall-clock enters the outputs.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from fplh.lake.parquet import sha256_of
from fplh.lake.storage import Lake
from fplh.settings import REPO_ROOT, get_settings


def git_sha(root: Path = REPO_ROOT) -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, check=True
        )
        return out.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def config_sha256(configs_dir: Path | None = None) -> str:
    base = configs_dir or get_settings().configs_dir
    h = hashlib.sha256()
    for p in sorted(base.rglob("*.yaml")):
        h.update(str(p.relative_to(base)).encode())
        h.update(p.read_bytes())
    return h.hexdigest()


def data_sha256(
    lake: Lake | None, prefixes: tuple[str, ...] = ("silver/",)
) -> tuple[str, dict[str, str]]:
    """Hash over every silver part (key + content hash)."""
    if lake is None:
        return "in-memory", {}
    files = {
        k: sha256_of(lake, k) for p in prefixes for k in lake.list(p) if k.endswith(".parquet")
    }
    h = hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()
    return h, files


@dataclass
class Manifest:
    predictor: str
    predictor_version: str
    deadlines: list[str]
    horizon: int
    git_sha: str
    data_manifest_sha256: str
    config_sha256: str
    rules_config: str
    seeds: dict[str, int] = field(default_factory=dict)
    max_observed_at: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def run_id(self) -> str:
        """Content-derived id: identical inputs → identical id."""
        payload = {k: v for k, v in asdict(self).items() if k not in ("max_observed_at", "extra")}
        digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:12]
        return f"{self.predictor}_{digest}"

    def to_json(self) -> str:
        return json.dumps({"run_id": self.run_id, **asdict(self)}, indent=2, sort_keys=True)

    @classmethod
    def from_json(cls, text: str) -> Manifest:
        data = json.loads(text)
        data.pop("run_id", None)
        return cls(**data)
