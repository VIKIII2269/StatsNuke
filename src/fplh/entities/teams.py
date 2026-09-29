"""Team identity: football-data names are canonical; other sources alias onto them."""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

import yaml
from rapidfuzz import process
from rapidfuzz.fuzz import token_set_ratio

from fplh.settings import get_settings

# Letters that Unicode decomposition leaves alone (Ødegaard, Højbjerg, Błaszczykowski …).
_TRANSLITERATE = str.maketrans(
    {
        "Ø": "O",
        "ø": "o",
        "Æ": "AE",
        "æ": "ae",
        "Œ": "OE",
        "œ": "oe",
        "ß": "ss",
        "Ł": "L",
        "ł": "l",
        "Đ": "D",
        "đ": "d",
        "Þ": "Th",
        "þ": "th",
        "ı": "i",
        "Ð": "D",
        "ð": "d",
    }
)


def normalise_name(name: str) -> str:
    """Accent-, punctuation- and case-insensitive form used for matching."""
    s = unicodedata.normalize("NFKD", name.translate(_TRANSLITERATE))
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = s.replace("'", "").replace("’", "")
    s = re.sub(r"[^A-Za-z0-9]+", " ", s)
    return s.strip().lower()


def slug(name: str) -> str:
    return normalise_name(name).replace(" ", "-")


class UnknownTeamError(ValueError):
    pass


@dataclass
class TeamResolver:
    aliases: dict[str, str]
    known: set[str] = field(default_factory=set)  # canonical uids

    @classmethod
    def from_config(cls, path: Path | None = None) -> TeamResolver:
        path = path or get_settings().configs_dir / "entities" / "teams.yaml"
        cfg = yaml.safe_load(path.read_text())
        aliases = {normalise_name(k): slug(v) for k, v in (cfg.get("aliases") or {}).items()}
        known = {slug(n) for n in cfg.get("canonical") or []} | set(aliases.values())
        return cls(aliases, known)

    def add_canonical(self, names: Iterable[str]) -> None:
        self.known.update(slug(n) for n in names)

    def uid(self, name: str) -> str:
        n = normalise_name(name)
        if n in self.aliases:
            return self.aliases[n]
        s = slug(name)
        if s in self.known:
            return s
        suggestions = process.extract(s, sorted(self.known), scorer=token_set_ratio, limit=3)
        raise UnknownTeamError(
            f"unknown team name {name!r}; add it to configs/entities/teams.yaml "
            f"(closest: {[x[0] for x in suggestions]})"
        )

    def uids(self, names: Iterable[str]) -> dict[str, str]:
        """Resolve many names, reporting every unknown one at once."""
        out: dict[str, str] = {}
        errors: list[str] = []
        for name in sorted(set(names)):
            try:
                out[name] = self.uid(name)
            except UnknownTeamError as exc:
                errors.append(str(exc))
        if errors:
            raise UnknownTeamError("\n".join(errors))
        return out
