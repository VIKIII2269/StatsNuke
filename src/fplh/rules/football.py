"""Laws of the game the simulator needs (``configs/rules/football.yaml``)."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import pandas as pd
import yaml

from fplh.settings import get_settings


@dataclass(frozen=True)
class SubstitutionRules:
    eras: tuple[tuple[pd.Timestamp, int], ...]
    concussion_from: pd.Timestamp
    concussion_max: int

    def limit(self, kickoff: pd.Timestamp) -> int:
        """Permanent (non-concussion) substitutions allowed per team at ``kickoff``."""
        allowed = [n for start, n in self.eras if start <= kickoff]
        if not allowed:
            raise ValueError(f"no substitution rule before {kickoff}")
        return allowed[-1]

    def concussion_allowance(self, kickoff: pd.Timestamp) -> int:
        return self.concussion_max if kickoff >= self.concussion_from else 0


def _ts(value: str) -> pd.Timestamp:
    return pd.Timestamp(value, tz="UTC")


@lru_cache(maxsize=4)
def load_substitution_rules(path: Path | None = None) -> SubstitutionRules:
    path = path or get_settings().configs_dir / "rules" / "football.yaml"
    cfg = yaml.safe_load(path.read_text())
    eras = tuple(sorted((_ts(e["from"]), int(e["max"])) for e in cfg["substitutions"]))
    conc = cfg["concussion_substitutes"]
    return SubstitutionRules(eras, _ts(conc["from"]), int(conc["max"]))
