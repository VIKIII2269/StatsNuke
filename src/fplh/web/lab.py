"""The experiment log (``docs/EXPERIMENTS_V2.md``) as structured sections for the site's
Lab page: each heading with its paragraphs' first line and its markdown tables."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def _cells(line: str) -> list[str]:
    return [c.strip() for c in line.strip().strip("|").split("|")]


def parse_log(text: str) -> list[dict[str, Any]]:
    """[{title, level, notes, tables: [{header, rows}]}] in document order."""
    sections: list[dict[str, Any]] = []
    cur: dict[str, Any] | None = None
    table: dict[str, Any] | None = None
    for raw in text.splitlines():
        line = raw.rstrip()
        if line.startswith("#"):
            level = len(line) - len(line.lstrip("#"))
            cur = {"title": line.lstrip("#").strip(), "level": level, "notes": [], "tables": []}
            sections.append(cur)
            table = None
            continue
        if cur is None:
            continue
        if line.startswith("|"):
            cells = _cells(line)
            if table is None:
                table = {"header": cells, "rows": []}
                cur["tables"].append(table)
            elif not all(set(c) <= set("-: ") for c in cells):
                table["rows"].append(cells)
            continue
        table = None
        if line.strip() and not line.startswith("```"):
            cur["notes"].append(line.strip())
    return [s for s in sections if s["tables"] or s["notes"]]


def load_log(path: Path) -> list[dict[str, Any]]:
    return parse_log(path.read_text()) if path.exists() else []
