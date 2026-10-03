"""Exact point-in-time trailing means, vectorised over many query times.

For each query row (a key such as ``player_uid`` and a time such as its ``deadline_at``),
``trailing_means`` averages each value over the last *k* history rows of the same key
that were **observed** at or before the query time. That is exactly what a builder would
compute from ``InformationSet.at(query time)``, so one call builds features for a whole
training set (many past deadlines) without leakage and without re-filtering per deadline.

History rows are taken in observation order (ties broken by ``order``). Missing values
are skipped; a window with no non-missing value is NaN. ``{prefix}n`` is the number of
visible history rows.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd


def _ns(t: pd.Series) -> pd.Series:
    out: pd.Series = pd.to_datetime(t, utc=True).dt.as_unit("ns")
    return out


def trailing_means(
    history: pd.DataFrame,
    key: str,
    values: Sequence[str],
    windows: Sequence[int],
    queries: pd.DataFrame,
    *,
    query_time: str = "deadline_at",
    observed: str = "observed_at",
    order: str = "kickoff_at",
    prefix: str = "",
) -> pd.DataFrame:
    """One row per query (same index): ``{prefix}{value}_{k}`` for each value and window,
    and ``{prefix}n``."""
    out = pd.DataFrame(index=queries.index)
    h = history.dropna(subset=[key]) if key in history else history.iloc[0:0]
    if h.empty or queries.empty:
        out[f"{prefix}n"] = 0
        for v in values:
            for k in windows:
                out[f"{prefix}{v}_{k}"] = np.nan
        return out
    sort_cols = [key, observed, order] if order in h else [key, observed]
    h = h.sort_values(sort_cols, kind="mergesort").reset_index(drop=True)
    codes, _ = pd.factorize(h[key])
    starts = np.r_[0, np.flatnonzero(np.diff(codes)) + 1]
    sizes = np.diff(np.r_[starts, len(h)])
    group_start = np.repeat(starts, sizes)
    seq = np.arange(len(h)) - group_start + 1  # 1-based position within the key

    # integer key codes shared by both sides (merge_asof needs identical key dtypes)
    _, uniques = pd.factorize(h[key])
    lookup = pd.Index(uniques)
    q_codes = lookup.get_indexer(queries[key])
    right = pd.DataFrame(
        {"_k": codes, "_t": _ns(h[observed]), "_seq": seq, "_start": group_start}
    ).sort_values("_t", kind="mergesort")
    left = pd.DataFrame(
        {
            "_k": q_codes,
            "_t": _ns(queries[query_time]).reset_index(drop=True),
            "_q": np.arange(len(queries)),
        }
    )
    valid = (left["_k"] >= 0) & left["_t"].notna()
    found = pd.merge_asof(
        left[valid].sort_values("_t", kind="mergesort"),
        right,
        on="_t",
        by="_k",
        direction="backward",
        allow_exact_matches=True,
    ).sort_values("_q")
    n = np.zeros(len(queries), dtype=np.int64)
    start = np.zeros(len(queries), dtype=np.int64)
    q_idx = found["_q"].to_numpy()
    n[q_idx] = found["_seq"].fillna(0).to_numpy(dtype=np.int64)
    start[q_idx] = found["_start"].fillna(0).to_numpy(dtype=np.int64)
    end = start + n  # exclusive end in the padded cumulative arrays

    columns: dict[str, np.ndarray] = {f"{prefix}n": n}
    for v in values:
        x = h[v].astype("float64").to_numpy()
        ok = ~np.isnan(x)
        s = np.r_[0.0, np.cumsum(np.where(ok, x, 0.0))]
        c = np.r_[0, np.cumsum(ok)]
        for k in windows:
            lo = end - np.minimum(n, k)
            cnt = c[end] - c[lo]
            with np.errstate(invalid="ignore", divide="ignore"):
                columns[f"{prefix}{v}_{k}"] = np.where(cnt > 0, (s[end] - s[lo]) / cnt, np.nan)
    return pd.concat([out, pd.DataFrame(columns, index=queries.index)], axis=1)
