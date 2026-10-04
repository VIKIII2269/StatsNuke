from __future__ import annotations

import pytest

from fplh.settings import Settings


@pytest.mark.parametrize("value", ["", "   "])
def test_blank_odds_key_counts_as_unset(monkeypatch: pytest.MonkeyPatch, value: str) -> None:
    # GitHub Actions passes a missing secret as an empty string
    monkeypatch.setenv("FPLH_ODDS_API_KEY", value)
    assert Settings().odds_api_key is None


def test_odds_key_is_read_when_set(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FPLH_ODDS_API_KEY", "abc123")
    key = Settings().odds_api_key
    assert key is not None and key.get_secret_value() == "abc123"
