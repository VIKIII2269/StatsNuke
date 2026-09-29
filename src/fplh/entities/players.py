"""Player identity: ``player_uid = "fpl:<code>"`` linked to Understat player ids.

FPL's ``code`` is stable across seasons, so FPL is the backbone. Understat links are
made per team-season (ARCHITECTURE.md §6.4):

1. candidates: players with minutes for the same team in the same season;
2. name score: max rapidfuzz ``token_set_ratio`` over FPL name variants (full name,
   web name, first + web) against the Understat name, all accent/case-normalised;
3. minutes agreement: season minutes for that team must agree (independent evidence);
4. one-to-one greedy assignment, strongest evidence first;
5. decision: score ≥ 92 with agreeing minutes → auto; 80–92 with near-exact minutes →
   auto (corroborated); a pair already linked in another season with agreeing minutes →
   auto; other 80–92 → review queue; < 80 → unlinked;
6. ``configs/entities/player_overrides.yaml`` applies last and always wins.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
import yaml
from rapidfuzz.fuzz import token_set_ratio

from fplh.entities.teams import normalise_name
from fplh.settings import get_settings

AUTO = 92.0
REVIEW = 80.0


def minutes_agree(a: np.ndarray, b: np.ndarray, *, exact: bool = False) -> np.ndarray:
    """Season minutes agree: within max(10, 5 %) (or max(2, 1 %) when ``exact``)."""
    big = np.maximum(a, b)
    tol = np.maximum(2.0, 0.01 * big) if exact else np.maximum(10.0, 0.05 * big)
    return np.asarray(np.abs(a - b) <= tol, dtype=bool)


def fpl_name_variants(first: str, second: str, web: str) -> list[str]:
    variants = {
        normalise_name(f"{first} {second}"),
        normalise_name(web),
        normalise_name(f"{first} {web}"),
    }
    return sorted(variants - {""})


@dataclass
class LinkResult:
    links: pd.DataFrame  # season, team, code, understat_player_id, score, method
    review: pd.DataFrame  # candidate pairs needing a human decision
    coverage: pd.DataFrame  # season, fpl_minutes, linked_minutes, share


LINK_COLUMNS = ["season", "team", "code", "understat_player_id", "score", "method"]


def load_overrides(path: Path | None = None) -> list[dict[str, Any]]:
    path = path or get_settings().configs_dir / "entities" / "player_overrides.yaml"
    if not path.exists():
        return []
    return list(yaml.safe_load(path.read_text()) or [])


def _candidates(
    fpl_minutes: pd.DataFrame, us_minutes: pd.DataFrame, fpl_players: pd.DataFrame
) -> pd.DataFrame:
    variants_by = {
        (str(s), int(c)): fpl_name_variants(str(f), str(sn), str(w))
        for s, c, f, sn, w in fpl_players[
            ["season", "code", "first_name", "second_name", "web_name"]
        ].itertuples(index=False)
    }
    rows = []
    us_groups = dict(tuple(us_minutes.groupby(["season", "team"])))
    for (season, team), fg in fpl_minutes.groupby(["season", "team"]):
        ug = us_groups.get((season, team))
        if ug is None:
            continue
        us = [
            (int(u), normalise_name(str(n)), float(m))
            for u, n, m in ug[["understat_player_id", "player_name", "minutes"]].itertuples(
                index=False
            )
        ]
        for code, minutes in fg[["code", "minutes"]].itertuples(index=False):
            variants = variants_by[(str(season), int(code))]
            for uid, uname, umin in us:
                score = max(token_set_ratio(v, uname) for v in variants)
                if score >= REVIEW:
                    rows.append((season, team, int(code), uid, float(score), float(minutes), umin))
    cand = pd.DataFrame(
        rows,
        columns=[
            "season",
            "team",
            "code",
            "understat_player_id",
            "score",
            "fpl_minutes",
            "us_minutes",
        ],
    )
    a, b = cand["fpl_minutes"].to_numpy(), cand["us_minutes"].to_numpy()
    cand["mins_ok"] = minutes_agree(a, b)
    cand["mins_exact"] = minutes_agree(a, b, exact=True) & (a > 0)
    strong = cand[(cand["score"] >= AUTO) & cand["mins_ok"]]
    known = set(zip(strong["code"], strong["understat_player_id"], strict=True))
    cand["known"] = [
        k in known for k in zip(cand["code"], cand["understat_player_id"], strict=True)
    ]
    return cand


def _method(score: float, known: bool, mins_ok: bool, mins_exact: bool) -> str | None:
    if score >= AUTO and mins_ok:
        return "name"
    if known and mins_ok:
        return "cross_season"
    if score >= REVIEW and mins_exact:
        return "name+exact_minutes"
    return None


def _apply_overrides(
    links: pd.DataFrame, fpl_minutes: pd.DataFrame, overrides: list[dict[str, Any]]
) -> pd.DataFrame:
    for o in overrides:
        code = int(o["code"])
        mask = links["code"] == code
        if "season" in o:
            mask &= links["season"] == o["season"]
        links = links[~mask]
        if o.get("understat_player_id") is None:
            continue
        fm = fpl_minutes[fpl_minutes["code"] == code]
        if "season" in o:
            fm = fm[fm["season"] == o["season"]]
        added = (
            fm[["season", "team"]]
            .drop_duplicates()
            .assign(
                code=code,
                understat_player_id=int(o["understat_player_id"]),
                score=100.0,
                method="override",
            )
        )
        links = pd.concat([links, added], ignore_index=True)
    return links


def link_players(
    fpl_minutes: pd.DataFrame,
    us_minutes: pd.DataFrame,
    fpl_players: pd.DataFrame,
    overrides: list[dict[str, Any]] | None = None,
) -> LinkResult:
    """Link per team-season.

    ``fpl_minutes``: season, team, code, minutes (summed per team-season);
    ``us_minutes``: season, team, understat_player_id, player_name, minutes;
    ``fpl_players``: season, code, first_name, second_name, web_name.
    """
    cand = _candidates(fpl_minutes, us_minutes, fpl_players)
    links: list[dict[str, Any]] = []
    review: list[dict[str, Any]] = []
    order = ["known", "score", "mins_exact"]
    for _, g in cand.sort_values(order, ascending=False).groupby(["season", "team"], sort=False):
        used_c: set[int] = set()
        used_u: set[int] = set()
        for r in cast(list[dict[str, Any]], g.to_dict("records")):
            if r["code"] in used_c or r["understat_player_id"] in used_u:
                continue
            method = _method(r["score"], r["known"], r["mins_ok"], r["mins_exact"])
            if method is None:
                review.append(r)
                continue
            links.append({**r, "method": method})
            used_c.add(r["code"])
            used_u.add(r["understat_player_id"])

    link_df = pd.DataFrame(links, columns=[*cand.columns, "method"])[LINK_COLUMNS]
    link_df = _apply_overrides(link_df, fpl_minutes, overrides or [])
    linked = set(zip(link_df["season"], link_df["team"], link_df["code"], strict=True))
    review_df = pd.DataFrame(review, columns=cand.columns)
    review_keys = zip(review_df["season"], review_df["team"], review_df["code"], strict=True)
    review_df = review_df.loc[np.array([k not in linked for k in review_keys], dtype=bool)]

    fm = fpl_minutes[fpl_minutes["season"].isin(set(us_minutes["season"]))].copy()
    is_linked = np.array(
        [k in linked for k in zip(fm["season"], fm["team"], fm["code"], strict=True)], dtype=bool
    )
    fm["linked_minutes"] = fm["minutes"].where(is_linked, 0)
    coverage = (
        fm.groupby("season", as_index=False)[["minutes", "linked_minutes"]]
        .sum()
        .rename(columns={"minutes": "fpl_minutes"})
    )
    coverage["share"] = coverage["linked_minutes"] / coverage["fpl_minutes"].clip(lower=1)
    return LinkResult(
        link_df.sort_values(["season", "team", "code"]).reset_index(drop=True),
        review_df.sort_values(["season", "team", "code"]).reset_index(drop=True),
        coverage,
    )


def build_dim_player(fpl_players: pd.DataFrame, links: pd.DataFrame) -> pd.DataFrame:
    """One row per real person (FPL code), latest names, and the Understat id if linked
    (the most frequent across seasons; conflicts are reported by the quality gates)."""
    latest = fpl_players.sort_values("season").drop_duplicates("code", keep="last")
    seasons = fpl_players.groupby("code")["season"].agg(first_season="min", last_season="max")
    dim = latest.set_index("code")[["first_name", "second_name", "web_name", "position"]].join(
        seasons
    )
    if not links.empty:
        us = links.groupby("code")["understat_player_id"].agg(lambda s: s.value_counts().index[0])
        n_ids = links.groupby("code")["understat_player_id"].nunique()
        dim = dim.join(us.rename("understat_player_id")).join(n_ids.rename("understat_ids_seen"))
    else:
        dim["understat_player_id"] = pd.NA
        dim["understat_ids_seen"] = 0
    dim = dim.reset_index()
    dim.insert(0, "player_uid", "fpl:" + dim["code"].astype(str))
    dim["understat_player_id"] = dim["understat_player_id"].astype("Int64")
    dim["understat_ids_seen"] = dim["understat_ids_seen"].fillna(0).astype("int64")
    return dim.sort_values("player_uid").reset_index(drop=True)
