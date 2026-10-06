from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from fplh.lake.storage import Lake
from fplh.web import benchmarks as bench
from fplh.web.checks import bronze_feeds
from fplh.web.export import clean, short
from fplh.web.lab import parse_log

NOW = pd.Timestamp("2026-10-06 12:00", tz="UTC")


def test_parse_log_keeps_tables_and_skips_separators() -> None:
    text = (
        "# T\n\nintro\n\n## A\n\nnote\n\n| a | b |\n|---|:-:|\n| 1 | 2 |\n\n"
        "### B\n\n| x |\n|---|\n| y |\n"
    )
    out = parse_log(text)
    assert [s["title"] for s in out] == ["T", "A", "B"]
    assert out[1]["tables"] == [{"header": ["a", "b"], "rows": [["1", "2"]]}]
    assert out[1]["notes"] == ["note"]


def test_clean_makes_strict_json() -> None:
    obj = {"a": np.float64("nan"), "b": [np.int64(3), math.inf], "c": NOW}
    out = clean(obj)
    assert out == {"a": None, "b": [3, None], "c": NOW.isoformat()}
    json.dumps(out, allow_nan=False)
    assert short("aston-villa") == "AVL" and short("somewhere") == "SOM"


def fake_fpl(total: int) -> tuple[bench.Getter, list[str]]:
    calls: list[str] = []

    def get(path: str) -> Any:
        calls.append(path)
        if path.startswith("/leagues-classic"):
            page = int(path.rsplit("=", 1)[1])
            first = (page - 1) * 50 + 1
            return {"standings": {"results": [{"entry": first + i} for i in range(50)]}}
        entry = int(path.split("/")[2])
        return {
            "current": [
                {
                    "event": gw,
                    "points": entry % 100 + gw,
                    "event_transfers_cost": 4 if gw == 2 else 0,
                }
                for gw in (1, 2)
            ]
        }

    return get, calls


def test_panels_are_chosen_once_and_fetched_per_final_gameweek(tmp_path: Path) -> None:
    lake = Lake(str(tmp_path))
    get, calls = fake_fpl(10_000)
    state = bench.update_panels(lake, "2026-27", 10_000, [1, 2], get, NOW)
    assert state["panels"]["p50"]["rank"] == 5000
    assert state["panels"]["p50"]["entries"][0] == 5000  # the entry at that rank
    assert len(state["panels"]["p99"]["entries"]) == bench.PANEL_SIZE
    n = len(calls)
    bench.update_panels(lake, "2026-27", 10_000, [1, 2], get, NOW)
    assert len(calls) == n  # nothing new to fetch
    med = bench.panel_medians(state)
    gw2 = next(r for r in med["p90"] if r["gw"] == 2)
    assert gw2["n"] == bench.PANEL_SIZE
    entries = state["panels"]["p90"]["entries"]
    assert gw2["points"] == float(np.median([e % 100 + 2 - 4 for e in entries]))  # net of hits


def test_openfpl_live_scores_both_forecasts_on_the_same_rows() -> None:
    v2 = pd.DataFrame(
        {
            "gw": 6,
            "player_uid": ["a", "b", "c"],
            "fixture_uid": "f",
            "expected_points": [2.0, 4.0, 1.0],
        }
    )
    rep = pd.DataFrame(
        {"gw": 6, "player_uid": ["a", "b"], "fixture_uid": "f", "expected_points": [3.0, 3.0]}
    )
    out = pd.DataFrame(
        {"player_uid": ["a", "b", "c"], "fixture_uid": "f", "total_points": [2, 6, 0]}
    )
    (row,) = bench.openfpl_live(v2, rep, out)
    assert row["n"] == 2
    assert row["mse_v2"] == 2.0 and row["mse_replica"] == 5.0


def test_bronze_feeds_report_age_and_status(tmp_path: Path) -> None:
    lake = Lake(str(tmp_path))
    for obs in ("2026-10-06T06:00:00Z", "2026-10-05T06:00:00Z"):
        key = f"bronze/source=fpl/endpoint=bootstrap-static/dt={obs[:10]}/obs={obs}.meta.json"
        lake.put_bytes(key, b"{}")
    lake.put_bytes(
        "bronze/source=odds_api/endpoint=odds/dt=2026-09-20/obs=2026-09-20T00:00:00Z.meta.json",
        b"{}",
    )
    feeds = {f["feed"]: f for f in bronze_feeds(lake, NOW)}
    assert feeds["fpl/bootstrap-static"]["captures"] == 2
    assert feeds["fpl/bootstrap-static"]["status"] == "warn"  # 6 h old, warn after 4 h
    assert feeds["odds_api/odds"]["status"] == "fail"
