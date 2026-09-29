"""Bronze layer: immutable raw responses plus request metadata (ARCHITECTURE.md §6.1).

Each record is stored as two objects:

* ``bronze/source=S/endpoint=E/dt=YYYY-MM-DD/obs=<ts>Z[__k=v...].json.gz``: the payload,
  byte-for-byte as received, gzip-compressed (deterministic, mtime=0);
* the same key with ``.meta.json`` in place of ``.json.gz``: source, endpoint, params,
  URL, HTTP status, ``observed_at`` and the payload's SHA-256.

Objects are create-only. Nothing in this module can overwrite or delete bronze.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import re
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from fplh import __version__
from fplh.clock import isoformat_z
from fplh.lake.storage import Lake

_SAFE = re.compile(r"^[A-Za-z0-9_.\-]+$")
PAYLOAD_SUFFIX = ".json.gz"
META_SUFFIX = ".meta.json"


@dataclass(frozen=True)
class BronzeRecord:
    source: str
    endpoint: str
    url: str
    http_status: int
    observed_at: datetime
    payload: bytes
    params: dict[str, str] = field(default_factory=dict)
    content_type: str | None = None
    # Allowlisted response headers worth keeping (e.g. API credit counters).
    response_headers: dict[str, str] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return 200 <= self.http_status < 300

    def json(self) -> Any:
        return json.loads(self.payload)


def _check(part: str, what: str) -> str:
    if not _SAFE.match(part) or set(part) == {"."}:
        raise ValueError(f"unsafe {what} for a lake key: {part!r}")
    return part


def bronze_key(rec: BronzeRecord) -> str:
    """Payload key for ``rec`` (see module docstring)."""
    if rec.observed_at.tzinfo is None:
        raise ValueError("observed_at must be timezone-aware")
    obs = rec.observed_at.astimezone(UTC)
    name = f"obs={isoformat_z(obs)}"
    for k in sorted(rec.params):
        name += f"__{_check(k, 'param name')}={_check(str(rec.params[k]), 'param value')}"
    return (
        f"bronze/source={_check(rec.source, 'source')}"
        f"/endpoint={_check(rec.endpoint, 'endpoint')}"
        f"/dt={obs:%Y-%m-%d}/{name}{PAYLOAD_SUFFIX}"
    )


def meta_key(payload_key: str) -> str:
    if not payload_key.endswith(PAYLOAD_SUFFIX):
        raise ValueError(f"not a bronze payload key: {payload_key}")
    return payload_key[: -len(PAYLOAD_SUFFIX)] + META_SUFFIX


def write_bronze(lake: Lake, rec: BronzeRecord) -> str:
    """Persist ``rec``; returns the payload key. Raises FileExistsError on collision."""
    key = bronze_key(rec)
    meta = {
        "source": rec.source,
        "endpoint": rec.endpoint,
        "params": rec.params,
        "url": rec.url,
        "http_status": rec.http_status,
        "content_type": rec.content_type,
        "response_headers": rec.response_headers,
        "observed_at": isoformat_z(rec.observed_at),
        "observed_at_us": rec.observed_at.astimezone(UTC).isoformat(),
        "sha256": hashlib.sha256(rec.payload).hexdigest(),
        "bytes": len(rec.payload),
        "collector_version": __version__,
    }
    # Payload first, then the sidecar: a sidecar's presence implies a complete record.
    lake.put_bytes(key, gzip.compress(rec.payload, mtime=0))
    lake.put_bytes(meta_key(key), json.dumps(meta, indent=2, sort_keys=True).encode())
    return key


def read_bronze(lake: Lake, payload_key: str) -> tuple[dict[str, Any], bytes]:
    """Return ``(meta, payload)``, verifying the payload against its recorded hash."""
    meta: dict[str, Any] = json.loads(lake.get_bytes(meta_key(payload_key)))
    payload = gzip.decompress(lake.get_bytes(payload_key))
    if hashlib.sha256(payload).hexdigest() != meta["sha256"]:
        raise ValueError(f"bronze payload hash mismatch: {payload_key}")
    return meta, payload


def list_bronze(lake: Lake, source: str, endpoint: str) -> list[str]:
    """Payload keys for one source/endpoint, oldest observation first."""
    prefix = f"bronze/source={source}/endpoint={endpoint}"
    return [k for k in lake.list(prefix) if k.endswith(PAYLOAD_SUFFIX)]


@dataclass(frozen=True)
class BronzeKey:
    source: str
    endpoint: str
    dt: str
    observed_at: datetime
    params: dict[str, str]


def parse_key(key: str) -> BronzeKey:
    """Inverse of :func:`bronze_key` for payload keys."""
    parts = key.split("/")
    if len(parts) != 5 or parts[0] != "bronze" or not key.endswith(PAYLOAD_SUFFIX):
        raise ValueError(f"not a bronze payload key: {key}")
    source = parts[1].removeprefix("source=")
    endpoint = parts[2].removeprefix("endpoint=")
    dt = parts[3].removeprefix("dt=")
    name = parts[4][: -len(PAYLOAD_SUFFIX)]
    obs, *kvs = name.split("__")
    observed_at = datetime.strptime(obs.removeprefix("obs="), "%Y-%m-%dT%H:%M:%SZ").replace(
        tzinfo=UTC
    )
    params = dict(kv.split("=", 1) for kv in kvs)
    return BronzeKey(source, endpoint, dt, observed_at, params)


def latest_by_params(
    lake: Lake, source: str, endpoint: str
) -> dict[tuple[tuple[str, str], ...], str]:
    """Latest payload key per distinct parameter set (keys sort chronologically)."""
    out: dict[tuple[tuple[str, str], ...], str] = {}
    for key in list_bronze(lake, source, endpoint):
        out[tuple(sorted(parse_key(key).params.items()))] = key
    return out
