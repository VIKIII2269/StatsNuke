"""Walk-forward evaluation (ARCHITECTURE.md §11.1, P5).

For each deadline D: build 𝓘(D), build the spine, ask the predictor, check that nothing
observed after D was served, and collect predictions. The same code path serves live
inference, with D = now.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal, Protocol

import pandas as pd

from fplh.evaluate.manifest import Manifest, config_sha256, data_sha256, git_sha
from fplh.features.information_set import InformationSet, LeakageError, SilverStore
from fplh.features.spine import SPINE_KEYS, build_spine, target_fixtures
from fplh.lake.parquet import read_parquet, to_parquet_bytes, write_parquet
from fplh.lake.storage import Lake

FIXTURE_KEYS = ["fixture_uid", "deadline_at", "horizon"]


class Predictor(Protocol):
    name: str
    version: str

    def predict(self, info: InformationSet, spine: pd.DataFrame) -> pd.DataFrame:
        """Spine keys plus prediction columns."""
        ...


class FixturePredictor(Protocol):
    name: str
    version: str

    def predict_fixtures(self, info: InformationSet, fixtures: pd.DataFrame) -> pd.DataFrame:
        """``fixture_uid, deadline_at, horizon`` plus prediction columns."""
        ...


@dataclass
class WalkForwardResult:
    manifest: Manifest
    predictions: pd.DataFrame
    written: list[str]

    @property
    def run_id(self) -> str:
        return self.manifest.run_id


def run_walk_forward(
    store: SilverStore,
    predictor: Predictor | FixturePredictor,
    deadlines: Sequence[pd.Timestamp],
    *,
    unit: Literal["player", "fixture"] = "player",
    horizon: int = 1,
    rules_config: str = "",
    seeds: dict[str, int] | None = None,
    lake: Lake | None = None,
    write: bool = True,
) -> WalkForwardResult:
    keys = SPINE_KEYS if unit == "player" else FIXTURE_KEYS
    frames = []
    max_obs: pd.Timestamp | None = None
    for d in sorted(deadlines):
        info = InformationSet.at(d, store)
        if unit == "player":
            spine = build_spine(info, horizon)
            if spine.empty:
                continue
            pred = predictor.predict(info, spine)  # type: ignore[union-attr]
        else:
            fixtures = target_fixtures(info, horizon).assign(deadline_at=d)
            if fixtures.empty:
                continue
            pred = predictor.predict_fixtures(info, fixtures)  # type: ignore[union-attr]
        info.assert_no_leakage()
        if info.max_observed_at is not None and info.max_observed_at > d:  # pragma: no cover
            raise LeakageError(f"{predictor.name} at {d}")
        missing = set(keys) - set(pred.columns)
        if missing:
            raise ValueError(f"predictions lack spine keys {missing}")
        frames.append(pred)
        if info.max_observed_at is not None and (max_obs is None or info.max_observed_at > max_obs):
            max_obs = info.max_observed_at

    data_hash, _ = data_sha256(store.lake)
    manifest = Manifest(
        predictor=predictor.name,
        predictor_version=predictor.version,
        extra={"unit": unit},
        deadlines=[d.isoformat() for d in sorted(deadlines)],
        horizon=horizon,
        git_sha=git_sha(),
        data_manifest_sha256=data_hash,
        config_sha256=config_sha256(),
        rules_config=rules_config,
        seeds=seeds or {},
        max_observed_at=max_obs.isoformat() if max_obs is not None else None,
    )
    predictions = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=keys)
    written: list[str] = []
    if write and lake is not None:
        base = f"gold/pred_{unit}/run_id={manifest.run_id}"
        written.append(write_parquet(lake, f"{base}/part-000.parquet", predictions, keys))
        key = f"gold/pred_run/run_id={manifest.run_id}/manifest.json"
        lake.put_bytes(key, manifest.to_json().encode(), overwrite=True)
        written.append(key)
    return WalkForwardResult(manifest, predictions, written)


def output_digest(result: WalkForwardResult) -> str:
    keys = FIXTURE_KEYS if result.manifest.extra.get("unit") == "fixture" else SPINE_KEYS
    return hashlib.sha256(to_parquet_bytes(result.predictions, keys)).hexdigest()


def cached_walk_forward(
    store: SilverStore,
    predictor: Predictor | FixturePredictor,
    deadlines: Sequence[pd.Timestamp],
    *,
    lake: Lake,
    unit: Literal["player", "fixture"] = "player",
    horizon: int = 1,
) -> WalkForwardResult:
    """Reuse a stored run with the same predictor name and version, deadlines, horizon,
    data and configuration, else run and store it. The git SHA is ignored: a predictor's
    ``version`` is what identifies its behaviour, so bump it when the code changes.
    In-memory stores cannot be content-hashed and are never reused."""
    if store.lake is None:
        return run_walk_forward(store, predictor, deadlines, unit=unit, horizon=horizon, lake=lake)
    data_hash, _ = data_sha256(store.lake)
    config = config_sha256()
    wanted = [d.isoformat() for d in sorted(deadlines)]
    for key in lake.list("gold/pred_run/"):
        if not key.endswith("manifest.json"):
            continue
        m = Manifest.from_json(lake.get_bytes(key).decode())
        if (
            m.predictor == predictor.name
            and m.predictor_version == predictor.version
            and m.deadlines == wanted
            and m.horizon == horizon
            and m.extra.get("unit") == unit
            and m.data_manifest_sha256 == data_hash
            and m.config_sha256 == config
        ):
            part = f"gold/pred_{unit}/run_id={m.run_id}/part-000.parquet"
            if lake.exists(part):
                return WalkForwardResult(m, read_parquet(lake, part), [])
    return run_walk_forward(store, predictor, deadlines, unit=unit, horizon=horizon, lake=lake)
