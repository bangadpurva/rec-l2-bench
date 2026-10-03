"""Leakage checks from the plan. Each returns a list of violations (empty = pass)."""
from __future__ import annotations

import pandas as pd

from .split import to_ts

FORBIDDEN_ITEM_FIELDS = ("average_rating", "rating_number", "bought_together")


def check_item_fields(items: pd.DataFrame) -> list[str]:
    return [f"item table has forbidden column {c!r}" for c in FORBIDDEN_ITEM_FIELDS
            if c in items.columns]


def check_pools(pools: pd.DataFrame, ratings: pd.DataFrame, first_seen: pd.Series) -> list[str]:
    """Pools must exclude already-seen items and not-yet-available items.

    pools: user_id, query_time, rank, parent_asin, l1_score
    """
    out = []
    ts = to_ts(ratings["timestamp"])
    seen = ratings.assign(ts=ts)[["user_id", "parent_asin", "ts"]]
    m = pools.merge(seen, on=["user_id", "parent_asin"], how="inner")
    m = m[m["ts"] < m["query_time"]]
    if len(m):
        out.append(f"{len(m)} pool rows are items the user saw before query time")
    fs = pools["parent_asin"].map(first_seen)
    late = pools[fs.isna() | (fs >= pools["query_time"])]
    if len(late):
        out.append(f"{len(late)} pool rows are items not yet available at query time")
    dup = pools.duplicated(["user_id", "parent_asin"]).sum()
    if dup:
        out.append(f"{dup} duplicate (user, item) pool rows")
    ranks = pools.groupby("user_id")["rank"].apply(lambda r: list(r) == list(range(len(r))))
    if not ranks.all():
        out.append(f"{(~ranks).sum()} users have non-contiguous ranks (expect 0..M-1 in order)")
    return out


def check_profiles(profile_events: pd.DataFrame, query_times: pd.Series) -> list[str]:
    """profile_events: user_id, timestamp of every interaction used in a profile."""
    ts = to_ts(profile_events["timestamp"])
    qt = profile_events["user_id"].map(query_times)
    bad = (ts >= qt).sum()
    return [f"{bad} profile events at or after query time"] if bad else []
