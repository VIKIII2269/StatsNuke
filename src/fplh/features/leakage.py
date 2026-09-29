"""Leakage checks run by the tests and by ``fplh evaluate leakage`` on real silver.

1. ``max_observed_at <= D`` for every builder at every deadline;
2. future-shuffle: perturbing every row observed after D leaves features unchanged.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping

import numpy as np
import pandas as pd

from fplh.features.builders import BUILDERS, FeatureBuilder
from fplh.features.information_set import TIME_INDEXED, InformationSet, SilverStore
from fplh.features.spine import SPINE_KEYS, build_spine


def perturb_future(
    frames: Mapping[str, pd.DataFrame], deadline: pd.Timestamp, seed: int = 0
) -> dict[str, pd.DataFrame]:
    """Copy of ``frames`` where every time-indexed row with ``observed_at > deadline``
    has its non-key values scrambled (numbers shuffled and shifted, flags flipped)."""
    rng = np.random.default_rng(seed)
    out: dict[str, pd.DataFrame] = {}
    for name, df in frames.items():
        if name not in TIME_INDEXED or df.empty:
            out[name] = df
            continue
        df = df.copy()
        future = df["observed_at"] > deadline
        if future.any():
            for col in df.columns:
                if (
                    col in ("observed_at", "event_at", "season")
                    or col.endswith("_uid")
                    or col in ("team", "code", "element")
                ):
                    continue
                s = df.loc[future, col]
                if pd.api.types.is_bool_dtype(s):
                    df.loc[future, col] = ~s.astype(bool)
                elif pd.api.types.is_numeric_dtype(s):
                    df.loc[future, col] = rng.permutation(s.to_numpy()) + 1
        out[name] = df
    return out


def features_at(
    store: SilverStore, deadline: pd.Timestamp, builders: Mapping[str, FeatureBuilder], horizon: int
) -> tuple[pd.DataFrame, InformationSet]:
    info = InformationSet.at(deadline, store)
    spine = build_spine(info, horizon)
    out = spine.copy()
    for builder in builders.values():
        out = out.merge(builder(info, spine), on=SPINE_KEYS, how="left")
    return out.sort_values(SPINE_KEYS).reset_index(drop=True), info


def check_leakage(
    frames: Mapping[str, pd.DataFrame],
    deadlines: Iterable[pd.Timestamp],
    builders: Mapping[str, FeatureBuilder] | None = None,
    horizon: int = 3,
) -> list[str]:
    """Problems found (empty list = clean)."""
    builders = builders or BUILDERS
    problems = []
    for d in deadlines:
        feats, info = features_at(SilverStore.from_frames(frames), d, builders, horizon)
        if info.max_observed_at is not None and info.max_observed_at > d:
            problems.append(f"{d}: served data observed at {info.max_observed_at}")
        shuffled, _ = features_at(
            SilverStore.from_frames(perturb_future(frames, d)), d, builders, horizon
        )
        try:
            pd.testing.assert_frame_equal(feats, shuffled)
        except AssertionError as exc:
            problems.append(f"{d}: features depend on future rows ({str(exc).splitlines()[0]})")
    return problems
