"""Runtime settings, read from environment variables (prefix ``FPLH_``).

Secrets (bucket credentials, API keys) are only ever read from the environment,
never from files in the repository (ARCHITECTURE.md §12.9).
"""

from __future__ import annotations

from pathlib import Path

from pydantic import SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="FPLH_", extra="ignore")

    # Any fsspec URL: a local path (default), file://..., s3://bucket/prefix (R2/B2 via
    # endpoint_url in FPLH_LAKE_STORAGE_OPTIONS or the standard AWS_* variables).
    lake_uri: str = "lake"
    configs_dir: Path = REPO_ROOT / "configs"
    cache_dir: Path = REPO_ROOT / ".cache"
    user_agent: str = "StatsNuke-fplh/0.1 (+https://github.com/vikiii2269/statsnuke)"
    http_timeout_s: float = 30.0
    odds_api_key: SecretStr | None = None  # FPLH_ODDS_API_KEY; never logged or stored
    odds_monthly_credits: int = 500

    @field_validator("odds_api_key", mode="before")
    @classmethod
    def _blank_is_unset(cls, v: object) -> object:
        """CI passes unset secrets as empty strings: treat them as not set."""
        return None if isinstance(v, str) and not v.strip() else v


def get_settings() -> Settings:
    return Settings()
