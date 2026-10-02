from __future__ import annotations

import pandas as pd
import pytest

from fplh.features.information_set import InformationSet, LeakageError, SilverStore
from fplh.lake.quality import check_goal_timeline, check_lineups, check_substitution_limit
from tests.synthetic_silver import deadlines, make, with_understat


def frames() -> dict[str, pd.DataFrame]:
    return with_understat(make())


def test_clean_timeline_passes() -> None:
    t = frames()
    for check in (check_lineups, check_goal_timeline, check_substitution_limit):
        assert check(t).passed, check(t).line()


def test_twelve_on_the_pitch_fails() -> None:
    t = frames()
    roster = t["fact_player_match_understat"].copy()
    sub = roster.index[~roster["started"]][0]
    roster.loc[sub, "on_minute"] = 10  # comes on before the player they replace leaves
    assert not check_lineups({**t, "fact_player_match_understat": roster}).passed


def test_missing_goal_event_fails() -> None:
    t = frames()
    ev = t["fact_match_event"]
    first_goal = ev.index[ev["kind"] == "goal"][0]
    res = check_goal_timeline({**t, "fact_match_event": ev.drop(index=first_goal)})
    assert not res.passed and len(res.failures) == 1


def test_too_many_substitutions_warns() -> None:
    t = frames()
    roster = t["fact_player_match_understat"].copy()
    mid = roster["understat_match_id"].iloc[0]
    one_side = roster[(roster["understat_match_id"] == mid) & (roster["side"] == "h")]
    extra = pd.concat([one_side.iloc[:1]] * 6, ignore_index=True).assign(
        started=False, roster_id=range(90_000, 90_006)
    )
    res = check_substitution_limit({**t, "fact_player_match_understat": pd.concat([roster, extra])})
    assert not res.blocking and not res.passed


def test_restrict_only_goes_back_in_time() -> None:
    t = frames()
    store = SilverStore.from_frames(t)
    ds = deadlines(t)
    later = InformationSet.at(ds[-1], store)
    earlier = later.restrict(ds[2])
    assert earlier.deadline == ds[2]
    ev = earlier.table("fact_match_event")
    assert (ev["observed_at"] <= ds[2]).all() and len(ev) < len(later.table("fact_match_event"))
    with pytest.raises(LeakageError):
        earlier.restrict(ds[3])
