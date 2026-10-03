"""Global timestamp split and cohort selection (protocol steps 1-4, 7)."""
from __future__ import annotations

import pandas as pd

# Exact cut points published with the Amazon'23 5-core timestamp split (Unix ms).
VALID_START_MS = 1628643414042   # 2021-08-11 01:16:54 UTC
TEST_START_MS = 1658002729837    # 2022-07-16 20:18:49 UTC
VALID_START = pd.Timestamp(VALID_START_MS, unit="ms", tz="UTC")
TEST_START = pd.Timestamp(TEST_START_MS, unit="ms", tz="UTC")


def to_ts(col: pd.Series) -> pd.Series:
    """Amazon'23 timestamps are Unix milliseconds."""
    if pd.api.types.is_datetime64_any_dtype(col):
        return col.dt.tz_localize("UTC") if col.dt.tz is None else col
    return pd.to_datetime(col, unit="ms", utc=True)


def window(split: str) -> tuple[pd.Timestamp, pd.Timestamp | None]:
    if split == "valid":
        return VALID_START, TEST_START
    if split == "test":
        return TEST_START, None
    raise ValueError(split)


def select_cohort(ratings: pd.DataFrame, split: str, *, min_prior: int = 3,
                  min_pos: int = 1, pos_rating: float = 4, n_users: int | None = None,
                  seed: int = 0) -> pd.DataFrame:
    """Return one row per cohort user: user_id, query_time, n_prior, n_pos.

    `ratings` needs columns user_id, parent_asin, rating, timestamp.
    """
    ts = to_ts(ratings["timestamp"])
    q, end = window(split)
    prior = ratings[ts < q]
    in_win = ratings[(ts >= q) & ((ts < end) if end is not None else True)]
    n_prior = prior.groupby("user_id").size().rename("n_prior")
    n_pos = in_win[in_win["rating"] >= pos_rating].groupby("user_id").size().rename("n_pos")
    users = pd.concat([n_prior, n_pos], axis=1).fillna(0).astype(int)
    users = users[(users.n_prior >= min_prior) & (users.n_pos >= min_pos)]
    users = users.sort_index()   # deterministic before sampling
    if n_users is not None and n_users < len(users):
        users = users.sample(n=n_users, random_state=seed).sort_index()
    out = users.reset_index().rename(columns={"index": "user_id"})
    out.insert(1, "query_time", q)
    return out


def positives(ratings: pd.DataFrame, split: str, pos_rating: float = 4) -> dict[str, set]:
    ts = to_ts(ratings["timestamp"])
    q, end = window(split)
    m = (ts >= q) & ((ts < end) if end is not None else True) & (ratings["rating"] >= pos_rating)
    return ratings[m].groupby("user_id")["parent_asin"].apply(set).to_dict()


def first_seen(ratings: pd.DataFrame) -> pd.Series:
    """Availability proxy: first interaction anywhere in the full data."""
    return to_ts(ratings["timestamp"]).groupby(ratings["parent_asin"]).min()
