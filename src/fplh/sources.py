"""Typed access to ``configs/sources.yaml`` (endpoints, rate limits, publication lags)."""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field

from fplh.settings import get_settings


class SourceConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    base_url: str | None = None
    requests_per_second: float = Field(default=1.0, gt=0)
    max_concurrency: int = Field(default=4, ge=1)
    # Conservative lag between event time and the time the fact could first be known
    # (o_f = e_f + lag) for backfilled history; ARCHITECTURE.md §6.2.
    publication_lag_hours: float = Field(default=0.0, ge=0)
    cadence: str | None = None
    notes: str | None = None


class SourcesConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    sources: dict[str, SourceConfig]

    def __getitem__(self, name: str) -> SourceConfig:
        return self.sources[name]


def load_sources(path: Path | None = None) -> SourcesConfig:
    path = path or get_settings().configs_dir / "sources.yaml"
    with path.open() as fh:
        return SourcesConfig.model_validate(yaml.safe_load(fh))
