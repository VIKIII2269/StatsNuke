from __future__ import annotations

import gzip
import json
from datetime import UTC, datetime, timedelta

import pytest

from fplh.lake.bronze import (
    BronzeRecord,
    bronze_key,
    list_bronze,
    meta_key,
    read_bronze,
    write_bronze,
)
from fplh.lake.storage import Lake

T0 = datetime(2026, 9, 29, 6, 0, 2, 123456, tzinfo=UTC)


def rec(at: datetime = T0, **kw: object) -> BronzeRecord:
    base: dict[str, object] = {
        "source": "fpl",
        "endpoint": "bootstrap-static",
        "url": "https://fantasy.premierleague.com/api/bootstrap-static/",
        "http_status": 200,
        "observed_at": at,
        "payload": b'{"events": []}',
    }
    base.update(kw)
    return BronzeRecord(**base)  # type: ignore[arg-type]


def test_key_layout_matches_spec() -> None:
    assert bronze_key(rec()) == (
        "bronze/source=fpl/endpoint=bootstrap-static/dt=2026-09-29/obs=2026-09-29T06:00:02Z.json.gz"
    )


def test_key_includes_sorted_params() -> None:
    key = bronze_key(rec(endpoint="event-live", params={"gw": "7", "a": "x"}))
    assert key.endswith("obs=2026-09-29T06:00:02Z__a=x__gw=7.json.gz")


def test_key_normalises_timezone_and_partition_date() -> None:
    from zoneinfo import ZoneInfo

    local = datetime(2026, 9, 29, 1, 30, tzinfo=ZoneInfo("Asia/Kolkata"))  # 2026-09-28T20:00Z
    assert "/dt=2026-09-28/obs=2026-09-28T20:00:00Z" in bronze_key(rec(at=local))


@pytest.mark.parametrize("bad", [{"source": "fp/l"}, {"params": {"q": "a b"}}, {"endpoint": ".."}])
def test_unsafe_key_parts_rejected(bad: dict[str, object]) -> None:
    with pytest.raises(ValueError, match="unsafe"):
        bronze_key(rec(**bad))


def test_naive_timestamp_rejected() -> None:
    with pytest.raises(ValueError, match="timezone"):
        bronze_key(rec(at=datetime(2026, 9, 29)))


def test_roundtrip_and_metadata(lake: Lake) -> None:
    key = write_bronze(lake, rec(params={"element": "1"}))
    meta, payload = read_bronze(lake, key)
    assert payload == b'{"events": []}'
    assert meta["source"] == "fpl"
    assert meta["params"] == {"element": "1"}
    assert meta["http_status"] == 200
    assert meta["observed_at"] == "2026-09-29T06:00:02Z"
    assert meta["observed_at_us"].startswith("2026-09-29T06:00:02.123456")
    assert len(meta["sha256"]) == 64
    # stored gzip is byte-identical to the received payload
    assert gzip.decompress(lake.get_bytes(key)) == rec().payload
    assert lake.exists(meta_key(key))


def test_bronze_is_create_only(lake: Lake) -> None:
    write_bronze(lake, rec())
    with pytest.raises(FileExistsError):
        write_bronze(lake, rec(payload=b"{}"))
    _, payload = read_bronze(lake, bronze_key(rec()))
    assert payload == b'{"events": []}'


def test_new_observation_is_a_new_object(lake: Lake) -> None:
    write_bronze(lake, rec())
    write_bronze(lake, rec(at=T0 + timedelta(hours=3)))
    keys = list_bronze(lake, "fpl", "bootstrap-static")
    assert len(keys) == 2
    assert keys[0] < keys[1]  # chronological


def test_tampering_detected(lake: Lake) -> None:
    key = write_bronze(lake, rec())
    lake.put_bytes(key, gzip.compress(b'{"events": [1]}'), overwrite=True)
    with pytest.raises(ValueError, match="hash mismatch"):
        read_bronze(lake, key)


def test_lake_rejects_path_escape(lake: Lake) -> None:
    with pytest.raises(ValueError, match="invalid lake key"):
        lake.put_bytes("bronze/../../etc/x", b"")


def test_list_ignores_temp_files(lake: Lake) -> None:
    write_bronze(lake, rec())
    lake.fs.open(f"{lake.root}/bronze/.tmp-dead", "wb").close()
    assert all(".tmp-" not in k for k in lake.list())


def test_meta_sidecar_is_json(lake: Lake) -> None:
    key = write_bronze(lake, rec())
    assert json.loads(lake.get_bytes(meta_key(key)))["bytes"] == len(rec().payload)
