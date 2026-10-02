"""Player identity: ``player_uid = "fpl:<code>"`` linked to Understat player ids.

FPL's ``code`` is stable across seasons, so FPL is the backbone. Understat links are
made per team-season (ARCHITECTURE.md §6.4):

1. candidates: players who appeared for the same team in the same season;
2. name score: max rapidfuzz ``token_set_ratio`` over FPL name variants (full name,
   web name, first + web) against the Understat name, all accent/case-normalised;
3. appearance overlap: Jaccard index of the fixtures each source says the player
   played in, over fixtures both sources cover. This is independent evidence that,
   unlike summed minutes, doesn't depend on how each source counts substitute minutes
   (they differ by ~1 min per appearance) or on their data windows differing;
4. one-to-one greedy assignment, strongest evidence first;
5. decision: score ≥ 92 with overlap ≥ 0.8 → auto; a pair already linked by name in
   another season with overlap ≥ 0.8 → auto; score 80–92 with overlap ≥ 0.95 over at
   least 3 shared fixtures → auto (corroborated); other candidates ≥ 80 → review queue;
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
OVERLAP = 0.8
OVERLAP_STRICT = 0.95
MIN_SHARED = 3


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
CAND_COLUMNS = ["season", "team", "code", "understat_player_id", "score", "overlap", "shared"]


def load_overrides(path: Path | None = None) -> list[dict[str, Any]]:
    path = path or get_settings().configs_dir / "entities" / "player_overrides.yaml"
    if not path.exists():
        return []
    return list(yaml.safe_load(path.read_text()) or [])


def _played(apps: pd.DataFrame, key: str) -> dict[tuple[str, str, int], frozenset[str]]:
    p = apps[apps["minutes"] > 0]
    out: dict[tuple[str, str, int], frozenset[str]] = {}
    for s, t, k, fx in zip(p["season"], p["team"], p[key], p["fixture_uid"], strict=True):
        ident = (str(s), str(t), int(k))
        out[ident] = out.get(ident, frozenset()) | {str(fx)}
    return out


def _candidates(
    fpl_apps: pd.DataFrame, us_apps: pd.DataFrame, fpl_players: pd.DataFrame
) -> pd.DataFrame:
    variants_by = {
        (str(s), int(c)): fpl_name_variants(str(f), str(sn), str(w))
        for s, c, f, sn, w in fpl_players[
            ["season", "code", "first_name", "second_name", "web_name"]
        ].itertuples(index=False)
    }
    # Only fixtures both sources cover are evidence: during a live season the sources'
    # windows differ (e.g. FPL history to GW1, Understat to GW5).
    common: dict[tuple[str, str], set[str]] = {}
    for (s, t), g in fpl_apps.groupby(["season", "team"]):
        common[(str(s), str(t))] = set(g["fixture_uid"].astype(str))
    for (s, t), g in us_apps.groupby(["season", "team"]):
        key = (str(s), str(t))
        common[key] = common.get(key, set()) & set(g["fixture_uid"].astype(str))
    fpl_sets = {k: v & common.get(k[:2], set()) for k, v in _played(fpl_apps, "code").items()}
    us_sets = {
        k: v & common.get(k[:2], set()) for k, v in _played(us_apps, "understat_player_id").items()
    }
    us_names = (
        us_apps.drop_duplicates(["season", "team", "understat_player_id"])
        .set_index(["season", "team", "understat_player_id"])["player_name"]
        .to_dict()
    )
    us_by_team: dict[tuple[str, str], list[tuple[int, str, frozenset[str]]]] = {}
    for (s, t, u), fixtures in us_sets.items():
        us_by_team.setdefault((s, t), []).append(
            (u, normalise_name(str(us_names[(s, t, u)])), fixtures)
        )
    rows = []
    for (s, t, code), f_fix in fpl_sets.items():
        variants = variants_by[(s, code)]
        for uid, uname, u_fix in us_by_team.get((s, t), []):
            score = max(token_set_ratio(v, uname) for v in variants)
            if score < REVIEW:
                continue
            union = f_fix | u_fix
            if not union:
                continue  # no fixture both sources cover: no evidence either way
            shared = len(f_fix & u_fix)
            rows.append((s, t, code, uid, float(score), shared / len(union), shared))
    cand = pd.DataFrame(rows, columns=CAND_COLUMNS)
    strong = cand[(cand["score"] >= AUTO) & (cand["overlap"] >= OVERLAP)]
    known = set(zip(strong["code"], strong["understat_player_id"], strict=True))
    cand["known"] = [
        k in known for k in zip(cand["code"], cand["understat_player_id"], strict=True)
    ]
    return cand


def _method(score: float, known: bool, overlap: float, shared: int) -> str | None:
    if score >= AUTO and overlap >= OVERLAP:
        return "name"
    if known and overlap >= OVERLAP:
        return "cross_season"
    if score >= REVIEW and overlap >= OVERLAP_STRICT and shared >= MIN_SHARED:
        return "name+appearances"
    return None


def _apply_overrides(
    links: pd.DataFrame, fpl_apps: pd.DataFrame, overrides: list[dict[str, Any]]
) -> pd.DataFrame:
    for o in overrides:
        code = int(o["code"])
        mask = links["code"] == code
        if "season" in o:
            mask &= links["season"] == o["season"]
        links = links[~mask]
        if o.get("understat_player_id") is None:
            continue
        fm = fpl_apps[fpl_apps["code"] == code]
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
    fpl_apps: pd.DataFrame,
    us_apps: pd.DataFrame,
    fpl_players: pd.DataFrame,
    overrides: list[dict[str, Any]] | None = None,
) -> LinkResult:
    """Link per team-season from appearance-level rows.

    ``fpl_apps``: season, team, code, fixture_uid, minutes (one row per player-fixture);
    ``us_apps``: season, team, understat_player_id, player_name, fixture_uid, minutes;
    ``fpl_players``: season, code, first_name, second_name, web_name.
    """
    cand = _candidates(fpl_apps, us_apps, fpl_players)
    links: list[dict[str, Any]] = []
    review: list[dict[str, Any]] = []
    order = ["known", "score", "overlap"]
    for _, g in cand.sort_values(order, ascending=False).groupby(["season", "team"], sort=False):
        used_c: set[int] = set()
        used_u: set[int] = set()
        for r in cast(list[dict[str, Any]], g.to_dict("records")):
            if r["code"] in used_c or r["understat_player_id"] in used_u:
                continue
            method = _method(r["score"], r["known"], r["overlap"], r["shared"])
            if method is None:
                review.append(r)
                continue
            links.append({**r, "method": method})
            used_c.add(r["code"])
            used_u.add(r["understat_player_id"])

    link_df = pd.DataFrame(links, columns=[*cand.columns, "method"])[LINK_COLUMNS]
    link_df = _apply_overrides(link_df, fpl_apps, overrides or [])
    linked = set(zip(link_df["season"], link_df["team"], link_df["code"], strict=True))
    review_df = pd.DataFrame(review, columns=cand.columns)
    review_keys = zip(review_df["season"], review_df["team"], review_df["code"], strict=True)
    review_df = review_df.loc[np.array([k not in linked for k in review_keys], dtype=bool)]

    fm = (
        fpl_apps[fpl_apps["season"].isin(set(us_apps["season"]))]
        .groupby(["season", "team", "code"])["minutes"]
        .sum()
        .reset_index()
    )
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
