"""The player simulator as a walk-forward predictor (ARCHITECTURE.md §8, ticket 3.7).

At each deadline D, for every fixture of the next round:

1. **Team rates.** Mean goals μ per side from a precomputed walk-forward team run (the
   Phase 2 fused rates by default), inverted to the goal process's nominal rates with the
   emulator, so the simulated team means equal μ.
2. **Players.** The deadline spine gives each side's registered players. M4 gives π^S,
   π^60, π^B (refit every ``refit_every`` deadlines); M5/M6 goal, assist and penalty
   weights; M9 yellow rates; M8 keeper multipliers; M7 defensive rates when the season's
   rules score them; M10 the BPS weights and player residuals. Every model is fitted on
   𝓘(D) only.
3. **Simulation.** ``sim.simulator.simulate_fixture`` with ``n_sims`` draws per fixture,
   seeded by (seed, fixture), summarised per player: expected points, the points pmf,
   P(60+), expected minutes, goals, assists, saves, bonus, clean-sheet and haul
   probabilities.

Ablations (ARCHITECTURE.md §11.6): ``minutes="naive"`` replaces M4 with the shares of the
last three matches (A4); ``attack="raw"`` replaces the shrunk goal rates with the raw
decayed per-90 rates (A5).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import pandas as pd

from fplh.evaluate.team_level import championship, observed_matches, season_teams
from fplh.features.information_set import InformationSet
from fplh.features.minutes import minutes_features
from fplh.features.spine import SPINE_KEYS
from fplh.models.attack import AttackRates, fit_attack
from fplh.models.bonus import fit_bonus
from fplh.models.cards import fit_cards
from fplh.models.defence import fit_defence
from fplh.models.gk import fit_saves
from fplh.models.goal_process import GoalProcessParams
from fplh.models.minutes import MinutesModel
from fplh.models.team_strength import TeamStrengthParams, run_filter
from fplh.rules.config import load_rules
from fplh.rules.football import load_substitution_rules
from fplh.sim.emulator import Emulator
from fplh.sim.simulator import (
    Components,
    FixtureInputs,
    League,
    SideInputs,
    SideResult,
    TimingModel,
    simulate_fixture,
)

PMF_RANGE = (-4, 25)  # points pmf columns pmf_-4 … pmf_25 (tails folded in)


def prematch_from_info(info: InformationSet, params: TeamStrengthParams) -> pd.DataFrame:
    """M1 pre-update rates of every match observable at D (each from earlier matches)."""
    matches = observed_matches(info)
    if matches.empty:
        return pd.DataFrame(columns=["fixture_uid", "pred_home", "pred_away"])
    f = run_filter(
        matches, params, season_teams(info), record=True, championship=championship(info)
    )
    hist = pd.DataFrame(f.history)
    dim = info.table("dim_fixture")
    keys = ["season", "home_team", "away_team"]
    out: pd.DataFrame = hist[[*keys, "pred_home", "pred_away"]].merge(
        dim[[*keys, "fixture_uid"]], on=keys
    )
    return out


def naive_minutes(x: pd.DataFrame) -> pd.DataFrame:
    """A4: shares of the last three matches (league means where a player has none)."""
    start = x["h_started_3"].fillna(0.0).to_numpy()
    appear = x["h_appeared_3"].fillna(0.0).to_numpy()
    full = x["hs_full_if_started_5"].fillna(x["hs_full_if_started_5"].mean()).fillna(0.8)
    sub = np.clip((appear - start) / np.clip(1 - start, 1e-9, None), 0, 1)
    return pd.DataFrame(
        {"p_start": start, "p_full": full.to_numpy(), "p_sub": sub}, index=x.index
    ).fillna(0.0)


def summarise_side(side: SideResult, uids: np.ndarray, fixture_uid: str) -> pd.DataFrame:
    pts = side.points
    ev = side.events
    lo, hi = PMF_RANGE
    clipped = np.clip(pts, lo, hi)
    frame: dict[str, object] = {
        "player_uid": uids,
        "fixture_uid": fixture_uid,
        "expected_points": pts.mean(axis=0),
        "sd_points": pts.std(axis=0),
        "p_start": side.started.mean(axis=0),
        "p_60": (ev["minutes"] >= 60).mean(axis=0),
        "p_play": (ev["minutes"] > 0).mean(axis=0),
        "expected_minutes": ev["minutes"].mean(axis=0),
        "e_goals": ev["goals_scored"].mean(axis=0),
        "e_assists": ev["assists"].mean(axis=0),
        "e_saves": ev["saves"].mean(axis=0),
        "e_bonus": ev["bonus"].mean(axis=0),
        "p_clean_sheet": ((ev["minutes"] >= 60) & (ev["goals_conceded"] == 0)).mean(axis=0),
        "p_haul": (pts >= 10).mean(axis=0),
    }
    for k in range(lo, hi + 1):
        frame[f"pmf_{k}"] = (clipped == k).mean(axis=0)
    return pd.DataFrame(frame)


@dataclass
class PlayerSimulator:
    """Walk-forward predictor (``Predictor`` protocol) over the deadline spine."""

    rates: pd.DataFrame  # fixture_uid, deadline_at, mu_home, mu_away
    emulator: Emulator
    params: GoalProcessParams
    m1_params: TeamStrengthParams
    n_sims: int = 2000
    minutes: Literal["model", "naive"] = "model"
    attack: Literal["shrunk", "raw"] = "shrunk"
    refit_every: int = 4
    seed: int = 0
    name: str = "player_sim"
    version: str = "1"
    _minutes_model: MinutesModel | None = field(default=None, repr=False)
    _calls: int = field(default=0, repr=False)

    def _minutes(self, info: InformationSet, spine: pd.DataFrame) -> pd.DataFrame:
        if self.minutes == "naive":
            return naive_minutes(minutes_features(info, spine))
        if self._minutes_model is None or self._calls % self.refit_every == 0:
            self._minutes_model = MinutesModel.fit(info)
        return self._minutes_model.predict(info, spine)

    def _goal_rates(self, attack: AttackRates, spine: pd.DataFrame) -> pd.DataFrame:
        r = attack.rates_for(spine["player_uid"], spine["position"])
        if self.attack == "raw" and not attack.players.empty:
            raw = attack.players["raw_goal_rate"].reindex(spine["player_uid"].to_numpy())
            by_pos = attack.players.groupby("position")["raw_goal_rate"].mean()
            fill = spine["position"].map(by_pos).fillna(0.0).to_numpy()
            r["goal_rate"] = np.where(raw.isna(), fill, raw.to_numpy())
        return r

    def predict(self, info: InformationSet, spine: pd.DataFrame) -> pd.DataFrame:
        spine = spine.reset_index(drop=True)
        mins = self._minutes(info, spine).reset_index(drop=True)
        self._calls += 1
        attack = fit_attack(info)
        goal = self._goal_rates(attack, spine)
        cards = fit_cards(info)
        defence = fit_defence(info)
        saves = fit_saves(info, prematch_from_info(info, self.m1_params))
        bonus = fit_bonus(info)
        timing = TimingModel.from_lineups(info.table("fact_player_match_understat"))
        lg = attack.league
        league = League(
            penalty_share=lg.get("penalty_share", 0.075),
            own_goal_share=lg.get("own_goal_share", 0.033),
            assist_share=lg.get("assist_share", 0.88),
            penalty_misses_per_side=lg.get("penalty_misses_per_side", 0.021),
        )
        components = Components(saves, bonus)
        subs = load_substitution_rules()
        now = self.rates[self.rates["deadline_at"] == info.deadline]
        mu_of = {
            str(u): (float(h), float(a))
            for u, h, a in zip(now["fixture_uid"], now["mu_home"], now["mu_away"], strict=True)
        }
        yellow = cards.rates_for(spine["player_uid"], spine["position"])
        keeper = saves.multiplier(spine["player_uid"])
        dc = defence.rates(spine["player_uid"], spine["position"])
        dc = dc * defence.opponent_factor(spine["opponent"])[:, None]
        dc_size = defence.group_size(spine["position"])
        frames = []
        for fixture_uid, rows in spine.groupby("fixture_uid", sort=True):
            if str(fixture_uid) not in mu_of:
                continue
            mu = mu_of[str(fixture_uid)]
            nh, na = self.emulator.nominal_for_means(mu[0], mu[1])
            kickoff = pd.Timestamp(rows["kickoff_at"].iloc[0])
            season = str(fixture_uid).split(":")[0]
            rules = load_rules(season.replace("-", "/"))
            sides = []
            for home in (True, False):
                idx = rows.index[rows["was_home"].to_numpy() == home].to_numpy()
                sides.append(
                    SideInputs(
                        spine.loc[idx, "player_uid"].to_numpy().astype(str),
                        spine.loc[idx, "position"].to_numpy().astype(str),
                        mins.loc[idx, "p_start"].to_numpy(float),
                        mins.loc[idx, "p_full"].to_numpy(float),
                        mins.loc[idx, "p_sub"].to_numpy(float),
                        goal.loc[idx, "goal_rate"].to_numpy(float),
                        goal.loc[idx, "assist_rate"].to_numpy(float),
                        goal.loc[idx, "pen_weight"].to_numpy(float),
                        yellow_rate=yellow[idx],
                        save_mult=keeper[idx],
                        dc_rate=dc[idx],
                        dc_size=dc_size[idx],
                    )
                )
            if not len(sides[0]) or not len(sides[1]):
                continue
            fx = FixtureInputs(
                str(fixture_uid),
                (float(np.exp(nh)), float(np.exp(na))),
                (sides[0], sides[1]),
                subs.limit(kickoff),
                mean_goals=mu,
            )
            res = simulate_fixture(
                fx, self.params, rules, league, timing, self.n_sims, self.seed, components
            )
            for side, inputs in zip(res.sides, sides, strict=True):
                frames.append(summarise_side(side, inputs.player_uid, str(fixture_uid)))
        if not frames:
            return spine[[*SPINE_KEYS, "position"]].assign(expected_points=np.nan)
        summary = pd.concat(frames, ignore_index=True)
        out: pd.DataFrame = spine[[*SPINE_KEYS, "position"]].merge(
            summary, on=["player_uid", "fixture_uid"], how="left"
        )
        return out
