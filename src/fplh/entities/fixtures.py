"""Fixture identity: ``fixture_uid = "<season>:<home uid>:<away uid>"``.

In a league season each ordered pair of teams meets exactly once, so the uid is
source-independent: every source joins on (season, home, away) with canonical team uids.
Kickoff dates from different sources become a cross-check, not the join key.
"""

from __future__ import annotations

import pandas as pd


def with_fixture_uid(
    df: pd.DataFrame, home: str = "home_team", away: str = "away_team"
) -> pd.DataFrame:
    out = df.copy()
    out["fixture_uid"] = (
        out["season"].astype(str) + ":" + out[home].astype(str) + ":" + out[away].astype(str)
    )
    return out


def build_dim_fixture(
    fpl_fixture: pd.DataFrame, fd_match: pd.DataFrame, us_match: pd.DataFrame
) -> pd.DataFrame:
    """One row per EPL fixture seen in any source. Kickoff and score prefer FPL, then
    football-data, then Understat."""
    frames = []
    if not fpl_fixture.empty:
        f = with_fixture_uid(fpl_fixture)
        frames.append(
            f[
                [
                    "fixture_uid",
                    "season",
                    "home_team",
                    "away_team",
                    "kickoff_at",
                    "round",
                    "fpl_fixture_id",
                    "home_goals",
                    "away_goals",
                ]
            ].assign(src_rank=0)
        )
    if not fd_match.empty:
        d = with_fixture_uid(fd_match[fd_match["division"] == "E0"])
        frames.append(
            d[
                [
                    "fixture_uid",
                    "season",
                    "home_team",
                    "away_team",
                    "kickoff_at",
                    "home_goals",
                    "away_goals",
                ]
            ].assign(src_rank=1)
        )
    if not us_match.empty:
        u = with_fixture_uid(us_match)
        frames.append(
            u[
                [
                    "fixture_uid",
                    "season",
                    "home_team",
                    "away_team",
                    "kickoff_at",
                    "understat_match_id",
                    "home_goals",
                    "away_goals",
                ]
            ].assign(src_rank=2)
        )
    if not frames:
        return pd.DataFrame()
    allf = pd.concat(frames, ignore_index=True)
    best = allf.sort_values(["fixture_uid", "src_rank"]).drop_duplicates("fixture_uid")
    dim = best[
        [
            "fixture_uid",
            "season",
            "home_team",
            "away_team",
            "kickoff_at",
            "home_goals",
            "away_goals",
        ]
    ].copy()
    for col, rank in (("round", 0), ("fpl_fixture_id", 0), ("understat_match_id", 2)):
        src = allf[allf["src_rank"] == rank]
        if col in src:
            dim = dim.merge(
                src[["fixture_uid", col]].dropna().drop_duplicates("fixture_uid"), how="left"
            )
        else:
            dim[col] = pd.NA
    dim["in_fpl"] = dim["fixture_uid"].isin(allf.loc[allf["src_rank"] == 0, "fixture_uid"])
    dim["in_football_data"] = dim["fixture_uid"].isin(
        allf.loc[allf["src_rank"] == 1, "fixture_uid"]
    )
    dim["in_understat"] = dim["fixture_uid"].isin(allf.loc[allf["src_rank"] == 2, "fixture_uid"])
    for col in ("round", "fpl_fixture_id", "understat_match_id", "home_goals", "away_goals"):
        dim[col] = pd.to_numeric(dim[col], errors="coerce").astype("Int64")
    return dim.sort_values("fixture_uid").reset_index(drop=True)
