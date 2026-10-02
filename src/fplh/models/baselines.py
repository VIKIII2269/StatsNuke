"""A0 baseline (ARCHITECTURE.md §11.7): market-only rates, naive minutes, raw per-90.

* **Match:** de-vig (multiplicative for A0) of the latest pre-match 1X2 and O/U 2.5
  prices observed at the deadline (market average preferred, then Pinnacle, then any
  single book; the market maximum is not a book and is never used).
* **Player:** expected FPL points from
  - minutes: the last 3 registered fixtures this season (P(60+), P(1–59), mean minutes);
  - events: season-to-date per-90 rates × expected minutes × the rules-config values;
  - clean sheet / goals conceded: Poisson rates inverted from the market prices; when no
    price was observable at D, league-average rates (flagged ``market_available``).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from fplh.features.builders import naive_minutes, season_totals
from fplh.features.information_set import InformationSet
from fplh.features.spine import SPINE_KEYS
from fplh.models.market import devig, expected_floor_div, invert_poisson
from fplh.rules.config import Rules

BOOK_PRIORITY = ("market_avg", "pinnacle", "bet365")
NOT_A_BOOK = frozenset({"market_max"})  # best price per outcome across books
LEAGUE_AVG_RATES = (1.5, 1.2)  # home, away goals per match (EPL long-run ≈)
FIXTURE_KEYS = ["fixture_uid", "deadline_at", "horizon"]


def market_probabilities(
    info: InformationSet, fixtures: pd.DataFrame, method: str = "multiplicative"
) -> pd.DataFrame:
    """Per fixture: de-vigged p_home/p_draw/p_away, p_over25, rates, market flag."""
    out = fixtures[["fixture_uid"]].drop_duplicates().copy()
    odds = info.table("snap_odds")
    cols = ["p_home", "p_draw", "p_away", "p_over25", "book"]
    if odds.empty:
        out[cols] = np.nan
    else:
        pre = odds[~odds["is_closing"] & odds["fixture_uid"].isin(out["fixture_uid"])]
        latest = pre.sort_values("observed_at").drop_duplicates(
            ["fixture_uid", "bookmaker", "market", "line", "outcome"], keep="last"
        )
        rows = []
        for uid, g in latest.groupby("fixture_uid"):
            books = list(set(g["bookmaker"]) - NOT_A_BOOK)
            order = [b for b in BOOK_PRIORITY if b in books] + sorted(
                set(books) - set(BOOK_PRIORITY)
            )
            row: dict[str, object] = {"fixture_uid": uid}
            for book in order:
                x = g[(g["bookmaker"] == book) & (g["market"] == "1x2")].set_index("outcome")[
                    "price"
                ]
                if {"home", "draw", "away"} <= set(x.index):
                    p = devig(x[["home", "draw", "away"]].to_numpy(), method)
                    row.update(p_home=p[0], p_draw=p[1], p_away=p[2], book=book)
                    break
            for book in order:
                t = g[
                    (g["bookmaker"] == book) & (g["market"] == "total") & (g["line"] == 2.5)
                ].set_index("outcome")["price"]
                if {"over", "under"} <= set(t.index):
                    row["p_over25"] = devig(t[["over", "under"]].to_numpy(), method)[0]
                    break
            rows.append(row)
        found = pd.DataFrame(rows, columns=["fixture_uid", *cols])
        out = out.merge(found, how="left")
    rates: list[tuple[float, float, bool]] = []
    probs = out[["p_home", "p_draw", "p_away", "p_over25"]].to_numpy(dtype=float)
    for ph, pd_, pa, po in probs:
        if np.isnan(ph):
            rates.append((*LEAGUE_AVG_RATES, False))
        else:
            lh, la = invert_poisson([ph, pd_, pa], None if np.isnan(po) else float(po))
            rates.append((lh, la, True))
    out[["lambda_home", "lambda_away", "market_available"]] = pd.DataFrame(rates, index=out.index)
    return out


@dataclass
class A0Match:
    name: str = "a0_match"
    version: str = "1"

    def predict_fixtures(self, info: InformationSet, fixtures: pd.DataFrame) -> pd.DataFrame:
        probs = market_probabilities(info, fixtures)
        base = fixtures[FIXTURE_KEYS].drop_duplicates()
        return base.merge(probs, on="fixture_uid", how="left")


@dataclass
class A0Player:
    rules: Rules
    name: str = "a0_player"
    version: str = "1"

    def predict(self, info: InformationSet, spine: pd.DataFrame) -> pd.DataFrame:
        r = self.rules
        f = spine.merge(naive_minutes(info, spine), on=SPINE_KEYS, how="left").merge(
            season_totals(info, spine), on=SPINE_KEYS, how="left"
        )
        market = market_probabilities(info, spine[["fixture_uid"]])
        f = f.merge(
            market[["fixture_uid", "lambda_home", "lambda_away", "market_available"]],
            on="fixture_uid",
            how="left",
        )
        lam_against = np.where(f["was_home"], f["lambda_away"], f["lambda_home"]).astype(float)

        def num(col: str) -> np.ndarray:
            return f[col].astype(float).fillna(0.0).to_numpy()

        p60, p_sub, mins, s_min = num("p60"), num("p1_59"), num("mins_mean"), num("s_minutes")
        per90 = np.divide(90.0, s_min, out=np.zeros_like(s_min), where=s_min > 0)

        def rate(col: str) -> np.ndarray:
            r_: np.ndarray = num(col) * per90 * mins / 90.0
            return r_

        pos = f["position"].to_numpy()
        goal_pts = np.vectorize(lambda p: r.goal.get(p, 0))(pos)
        cs_pts = np.vectorize(lambda p: getattr(r.clean_sheet, p))(pos)
        gc_rule = {p: (v.per, v.points) for p, v in r.goals_conceded.items()}
        gc = np.array(
            [
                expected_floor_div(lam, gc_rule[p][0]) * gc_rule[p][1] if p in gc_rule else 0.0
                for p, lam in zip(pos, lam_against, strict=True)
            ]
        )
        dc_group = np.where(
            pos == "DEF",
            r.defensive_contribution.DEF.points,
            r.defensive_contribution.MID_FWD.points,
        )
        apps = num("s_apps")
        dc_rate = np.divide(num("s_dc_hits"), apps, out=np.zeros_like(apps), where=apps > 0)
        dc_rate = np.where(pos == "GK", 0.0, dc_rate)

        exp_pts = (
            r.appearance.gte_60 * p60
            + r.appearance.lt_60 * p_sub
            + rate("s_goals_scored") * goal_pts
            + rate("s_assists") * r.assist
            + rate("s_saves") / r.saves.per * r.saves.points
            + rate("s_penalties_saved") * r.penalty_save
            + rate("s_penalties_missed") * r.penalty_miss
            + rate("s_yellow_cards") * r.yellow_card
            + rate("s_red_cards") * r.red_card
            + rate("s_own_goals") * r.own_goal
            + rate("s_bonus")
            + p60 * np.exp(-lam_against) * cs_pts
            + p60 * gc
            + (p60 + p_sub) * dc_rate * dc_group
        )
        out = f[[*SPINE_KEYS, "position"]].copy()
        out["expected_points"] = exp_pts
        out["p_60"] = p60
        out["expected_minutes"] = mins
        out["market_available"] = f["market_available"].to_numpy()
        return out
