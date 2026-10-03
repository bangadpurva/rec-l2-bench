"""Extra L1 channels and the merge into one frozen pool.

Every channel returns a dense score matrix [users, items]; ineligible items are
masked to -inf by the caller via `topm_from_scores`. All counts use only
interactions strictly before each user's query time.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.sparse as sp

DAY_MS = 86_400_000


def _qms(cohort: pd.DataFrame, users: list[str]) -> dict[str, int]:
    qt = cohort.set_index("user_id")["query_time"]
    return {u: int(qt[u].value // 1_000_000) for u in users}


def popularity_scores(ratings, cohort, users, item_index, window_days: int = 90) -> np.ndarray:
    """Interactions per item in the window before each user's query time."""
    q = _qms(cohort, users)
    out = np.zeros((len(users), len(item_index)), dtype=np.float32)
    cache: dict[int, np.ndarray] = {}
    for i, u in enumerate(users):
        t = q[u]
        if t not in cache:
            r = ratings[(ratings.timestamp < t) & (ratings.timestamp >= t - window_days * DAY_MS)]
            v = np.zeros(len(item_index), dtype=np.float32)
            idx = r.parent_asin.map(item_index).dropna().astype(int).to_numpy()
            np.add.at(v, idx, 1.0)
            cache[t] = v
        out[i] = cache[t]
    return out


def cooccurrence_scores(ratings, cohort, users, item_index, like_min: float = 4,
                        half_life_days: float = 180) -> np.ndarray:
    """Item-item co-occurrence over all users' histories before the query time,
    cosine-normalized, scored against each user's recency-weighted liked items.
    Users who share a query time share one co-occurrence matrix."""
    q = _qms(cohort, users)
    n = len(item_index)
    out = np.zeros((len(users), n), dtype=np.float32)
    mats: dict[int, sp.csr_matrix] = {}
    for t in sorted(set(q.values())):
        r = ratings[(ratings.timestamp < t) & ratings.parent_asin.isin(item_index)]
        uix = {u: k for k, u in enumerate(r.user_id.unique())}
        X = sp.csr_matrix((np.ones(len(r), dtype=np.float32),
                           (r.user_id.map(uix).to_numpy(), r.parent_asin.map(item_index).to_numpy())),
                          shape=(len(uix), n))
        X.data[:] = 1.0
        C = (X.T @ X).tocsr()
        C.setdiag(0)
        C.eliminate_zeros()
        deg = np.sqrt(np.asarray(X.sum(0)).ravel()) + 1e-9
        Dinv = sp.diags(1.0 / deg)
        mats[t] = (Dinv @ C @ Dinv).tocsr()
    hist = ratings[ratings.user_id.isin(q) & ratings.parent_asin.isin(item_index)]
    groups = dict(tuple(hist.groupby("user_id")))
    for i, u in enumerate(users):
        g = groups[u]
        g = g[g.timestamp < q[u]]
        liked = g[g.rating >= like_min]
        if liked.empty:
            liked = g
        w = 0.5 ** ((q[u] - liked.timestamp.to_numpy()) / DAY_MS / half_life_days)
        idx = liked.parent_asin.map(item_index).to_numpy()
        s = sp.csr_matrix((w.astype(np.float32), (np.zeros(len(idx), int), idx)), shape=(1, n))
        out[i] = (s @ mats[q[u]]).toarray().ravel()
    return out


def topm_from_scores(S: np.ndarray, mask: np.ndarray, m: int) -> tuple[np.ndarray, np.ndarray]:
    S = np.where(mask, S, -np.inf).astype(np.float32)
    k = min(m, S.shape[1])
    part = np.argpartition(-S, k - 1, axis=1)[:, :k]
    ps = np.take_along_axis(S, part, 1)
    order = np.argsort(-ps, axis=1, kind="stable")
    idx = np.take_along_axis(part, order, 1)
    sc = np.take_along_axis(ps, order, 1)
    idx[~np.isfinite(sc)] = -1
    return idx, sc


def interleave(channels: dict[str, np.ndarray], m: int, quotas: dict[str, int] | None = None):
    """Round-robin merge of per-channel ranked lists, deduplicated, to length m.

    `quotas` caps how many slots each channel may fill (default: no cap).
    Returns (idx [users, m], source [users, m] as channel names).
    Merged rank r gets l1_score = 1/(r+1); the source channel is kept for analysis.
    """
    names = list(channels)
    n_users = next(iter(channels.values())).shape[0]
    idx = np.full((n_users, m), -1, dtype=np.int64)
    src = np.full((n_users, m), "", dtype=object)
    for u in range(n_users):
        seen, out, used = set(), [], {c: 0 for c in names}
        ptr = {c: 0 for c in names}
        while len(out) < m:
            progressed = False
            for c in names:
                if len(out) >= m:
                    break
                if quotas and used[c] >= quotas.get(c, m):
                    continue
                row = channels[c][u]
                while ptr[c] < len(row) and (row[ptr[c]] < 0 or row[ptr[c]] in seen):
                    ptr[c] += 1
                if ptr[c] < len(row):
                    j = int(row[ptr[c]])
                    seen.add(j)
                    out.append((j, c))
                    used[c] += 1
                    ptr[c] += 1
                    progressed = True
            if not progressed:
                break
        for r, (j, c) in enumerate(out):
            idx[u, r], src[u, r] = j, c
    return idx, src
