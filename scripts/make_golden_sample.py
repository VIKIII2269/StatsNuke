"""Rebuild the offline golden samples in tests/golden/data from the cached vaastav CSVs.

Run ``uv run fplh golden fetch --season 2025/26 --season 2026/27`` first. Keeps whole
fixtures (bonus ranking needs every player in a match), choosing one fixture per rules
edge case plus both fixtures of a double-gameweek player and fixture 56, which carries
the known exact-duplicate artefact.
"""

from __future__ import annotations

import pandas as pd

from fplh.rules.engine import DEFENSIVE_COLUMNS, EVENT_COLUMNS
from fplh.rules.golden import golden_cache_path
from fplh.settings import REPO_ROOT, get_settings

OUT = REPO_ROOT / "tests" / "golden" / "data"
COLUMNS = [
    *("element", "name", "position", "team", "fixture", "GW", "kickoff_time", "total_points"),
    *EVENT_COLUMNS,
    *DEFENSIVE_COLUMNS,
    *("defensive_contribution", "bps", "clean_sheets", "starts"),
]
DUPLICATE_ARTEFACT_FIXTURE = 56


def _has_bonus_tie(g: pd.DataFrame) -> bool:
    top = g.loc[g["minutes"] > 0, "bps"].sort_values(ascending=False).head(4).tolist()
    return len(top) >= 3 and len(set(top)) < len(top)


def main() -> None:
    cache = get_settings().cache_dir
    raw = pd.read_csv(golden_cache_path(cache, "2025/26"))
    d = raw.drop_duplicates()
    cbi_t = d["clearances_blocks_interceptions"] + d["tackles"]
    outfield_mf = d["position"].isin(["MID", "FWD"])
    edge_cases = {
        "penalty save": d["penalties_saved"] > 0,
        "penalty miss": d["penalties_missed"] > 0,
        "own goal": d["own_goals"] > 0,
        "red card": d["red_cards"] > 0,
        "DEF defensive contribution": (d["position"] == "DEF") & (cbi_t >= 10),
        "MID/FWD defensive contribution": outfield_mf & (cbi_t + d["recoveries"] >= 12),
        "GK 6+ saves": (d["position"] == "GK") & (d["saves"] >= 6),
        "MID clean sheet": (d["position"] == "MID") & (d["clean_sheets"] == 1),
        "DEF conceded 4+": (d["position"] == "DEF") & (d["goals_conceded"] >= 4),
        "sub, <60 min, 0 conceded": (d["minutes"].between(1, 59)) & (d["goals_conceded"] == 0),
    }
    ties = {int(k) for k, g in d.groupby("fixture") if _has_bonus_tie(g)}
    chosen: set[int] = {DUPLICATE_ARTEFACT_FIXTURE}
    for name, mask in [*edge_cases.items(), ("bonus tie", d["fixture"].isin(ties))]:
        fixtures = sorted(set(d.loc[mask, "fixture"]))
        if not fixtures:
            raise SystemExit(f"no fixture with: {name}")
        if not chosen & set(fixtures):
            chosen.add(int(fixtures[0]))
    dgw = d[d.groupby(["element", "GW"])["fixture"].transform("nunique") > 1]
    first = dgw.sort_values(["GW", "element"]).iloc[0]
    chosen |= set(
        dgw.loc[(dgw["element"] == first["element"]) & (dgw["GW"] == first["GW"]), "fixture"]
    )

    sample = raw[raw["fixture"].isin(chosen)][COLUMNS].sort_values(["fixture", "element"])
    sample.to_csv(OUT / "2025_26_sample.csv", index=False)
    gw1 = pd.read_csv(golden_cache_path(cache, "2026/27"))[COLUMNS]
    gw1.sort_values(["fixture", "element"]).to_csv(OUT / "2026_27_gw1.csv", index=False)
    print(f"2025/26 fixtures {sorted(chosen)}: {len(sample)} rows; 2026/27: {len(gw1)} rows")


if __name__ == "__main__":
    main()
