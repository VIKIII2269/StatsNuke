"""Deterministic Parquet for silver and gold.

The same frame always serialises to the same bytes: rows are sorted by the given keys,
the index is dropped, and pyarrow writes no wall-clock metadata. That makes "rebuild
twice → identical hashes" and "replay a run → identical outputs" testable.
"""

from __future__ import annotations

import hashlib
import io
from collections.abc import Sequence

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from fplh.lake.storage import Lake


def to_parquet_bytes(df: pd.DataFrame, sort_by: Sequence[str] = ()) -> bytes:
    if sort_by:
        df = df.sort_values(list(sort_by), kind="mergesort")
    table = pa.Table.from_pandas(df.reset_index(drop=True), preserve_index=False)
    buf = io.BytesIO()
    pq.write_table(
        table,
        buf,
        compression="zstd",
        use_dictionary=True,
        write_statistics=True,
        store_schema=True,
    )
    return buf.getvalue()


def write_parquet(lake: Lake, key: str, df: pd.DataFrame, sort_by: Sequence[str] = ()) -> str:
    """Write (overwrite) a silver/gold table part. Refuses bronze keys."""
    if key.startswith("bronze/"):
        raise ValueError("bronze is create-only; use write_bronze")
    lake.put_bytes(key, to_parquet_bytes(df, sort_by), overwrite=True)
    return key


def read_parquet(lake: Lake, key: str) -> pd.DataFrame:
    df: pd.DataFrame = pq.read_table(io.BytesIO(lake.get_bytes(key))).to_pandas()
    return df


def read_table(lake: Lake, prefix: str) -> pd.DataFrame:
    """Concatenate every part under ``prefix`` (e.g. ``silver/fact_shot``)."""
    keys = [k for k in lake.list(prefix) if k.endswith(".parquet")]
    if not keys:
        return pd.DataFrame()
    return pd.concat([read_parquet(lake, k) for k in keys], ignore_index=True)


def sha256_of(lake: Lake, key: str) -> str:
    return hashlib.sha256(lake.get_bytes(key)).hexdigest()
