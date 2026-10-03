"""L1: recency-weighted user vectors, exact and HNSW top-M, filters, diagnostics."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .embed import normalize

DAY_MS = 86_400_000


def user_vectors(ratings: pd.DataFrame, cohort: pd.DataFrame, item_index: dict[str, int],
                 item_vecs: np.ndarray, half_life_days: float, like_min: float = 4):
    """Recency-weighted mean of liked-item vectors before query time.

    Users with no liked items fall back to all prior items (counted in `fallback`).
    """
    qt = {u: int(q.value // 1_000_000) for u, q in zip(cohort.user_id, cohort.query_time)}
    sub = ratings[ratings.user_id.isin(qt) & ratings.parent_asin.isin(item_index)]
    sub = sub[sub.timestamp < sub.user_id.map(qt)]
    users = sorted(qt)
    out = np.zeros((len(users), item_vecs.shape[1]), dtype=np.float32)
    fallback = []
    groups = dict(tuple(sub.groupby("user_id")))
    for i, u in enumerate(users):
        g = groups.get(u)
        if g is None or g.empty:
            raise ValueError(f"user {u} has no prior interactions in the catalog")
        liked = g[g.rating >= like_min]
        if liked.empty:
            liked = g
            fallback.append(u)
        age_days = (qt[u] - liked.timestamp.to_numpy()) / DAY_MS
        w = 0.5 ** (age_days / half_life_days)
        idx = liked.parent_asin.map(item_index).to_numpy()
        out[i] = (w[:, None] * item_vecs[idx]).sum(0) / w.sum()
    return users, normalize(out), fallback


def eligibility(users: list[str], cohort: pd.DataFrame, ratings: pd.DataFrame,
                item_ids: list[str], first_seen: pd.Series):
    """Boolean mask [users, items]: available at query time and not seen by the user."""
    idx = {a: j for j, a in enumerate(item_ids)}
    qt = cohort.set_index("user_id")["query_time"]
    fs = first_seen.reindex(item_ids).to_numpy()
    mask = np.zeros((len(users), len(item_ids)), dtype=bool)
    for i, u in enumerate(users):
        mask[i] = fs < qt[u]          # NaT compares False -> ineligible
    seen = ratings[ratings.user_id.isin(set(users))]
    q_ms = {u: int(qt[u].value // 1_000_000) for u in users}
    seen = seen[seen.timestamp < seen.user_id.map(q_ms)]
    urow = {u: i for i, u in enumerate(users)}
    for u, a in zip(seen.user_id, seen.parent_asin):
        j = idx.get(a)
        if j is not None:
            mask[urow[u], j] = False
    return mask


def exact_topm(U: np.ndarray, V: np.ndarray, mask: np.ndarray, m: int, batch: int = 512):
    """Cosine top-M over eligible items. Returns (indices, scores), each [users, m]."""
    n = U.shape[0]
    idx = np.full((n, m), -1, dtype=np.int64)
    sc = np.full((n, m), -np.inf, dtype=np.float32)
    for s in range(0, n, batch):
        S = U[s:s + batch] @ V.T
        S[~mask[s:s + batch]] = -np.inf
        k = min(m, S.shape[1])
        part = np.argpartition(-S, k - 1, axis=1)[:, :k]
        ps = np.take_along_axis(S, part, 1)
        order = np.argsort(-ps, axis=1, kind="stable")
        idx[s:s + batch, :k] = np.take_along_axis(part, order, 1)
        sc[s:s + batch, :k] = np.take_along_axis(ps, order, 1)
    sc[idx < 0] = -np.inf
    # drop ineligible tail (users with < m eligible items)
    idx[~np.isfinite(sc)] = -1
    return idx, sc


def hnsw_topm(U, V, mask, m: int, M: int = 32, ef_construction: int = 200, ef_search: int = 512,
              overfetch: int = 4, seed: int = 0):
    """FAISS HNSW (inner product on normalized vectors = cosine), filtered after search."""
    import faiss
    faiss.omp_set_num_threads(1)  # deterministic graph build
    index = faiss.IndexHNSWFlat(V.shape[1], M, faiss.METRIC_INNER_PRODUCT)
    index.hnsw.efConstruction = ef_construction
    np.random.seed(seed)
    index.add(V.astype(np.float32))
    index.hnsw.efSearch = max(ef_search, m * overfetch)
    D, I = index.search(U.astype(np.float32), m * overfetch)
    idx = np.full((U.shape[0], m), -1, dtype=np.int64)
    sc = np.full((U.shape[0], m), -np.inf, dtype=np.float32)
    short = 0
    for i in range(U.shape[0]):
        keep = [(j, d) for j, d in zip(I[i], D[i]) if j >= 0 and mask[i, j]][:m]
        short += len(keep) < m
        for r, (j, d) in enumerate(keep):
            idx[i, r], sc[i, r] = j, d
    return idx, sc, short


def overlap_at_k(a: np.ndarray, b: np.ndarray, k: int) -> float:
    return float(np.mean([len(set(x[:k]) & set(y[:k])) / k for x, y in zip(a, b)]))


def to_pool_frame(users, cohort, idx, sc, item_ids, source=None) -> pd.DataFrame:
    qt = cohort.set_index("user_id")["query_time"]
    rows = []
    for i, u in enumerate(users):
        for r, (j, s) in enumerate(zip(idx[i], sc[i])):
            if j < 0:
                break
            rows.append((u, qt[u], r, item_ids[j], float(s),
                         source[i, r] if source is not None else "dense"))
    return pd.DataFrame(rows, columns=["user_id", "query_time", "rank", "parent_asin", "l1_score", "source"])


def recall_table(idx: np.ndarray, users, positives, item_ids, ks=(50, 100, 200)) -> dict:
    out = {}
    for k in ks:
        r, h = [], []
        for i, u in enumerate(users):
            got = {item_ids[j] for j in idx[i, :k] if j >= 0}
            f = len(positives[u] & got)
            r.append(f / len(positives[u]))
            h.append(f > 0)
        out[f"recall@{k}"] = float(np.mean(r))
        out[f"users_with_pos@{k}"] = float(np.mean(h))
    return out


def positive_diagnostics(pools: pd.DataFrame, positives: dict[str, set], users: list[str],
                         first_seen: pd.Series, query_time: pd.Series, seen_before: dict[str, set],
                         ks=(50, 100, 200)) -> dict:
    """Recall@K against all positives, and why positives are unreachable."""
    by_user = {u: g.sort_values("rank").parent_asin.tolist() for u, g in pools.groupby("user_id")}
    rec = {k: [] for k in ks}
    hit = {k: [] for k in ks}
    n_pos = n_new = n_repeat = 0
    for u in users:
        pos = positives[u]
        ranked = by_user.get(u, [])
        for k in ks:
            found = len(pos & set(ranked[:k]))
            rec[k].append(found / len(pos))
            hit[k].append(found > 0)
        n_pos += len(pos)
        n_new += sum(1 for a in pos if not (first_seen.get(a, pd.NaT) < query_time[u]))
        n_repeat += len(pos & seen_before.get(u, set()))
    return {
        **{f"recall@{k}": float(np.mean(v)) for k, v in rec.items()},
        **{f"users_with_pos@{k}": float(np.mean(v)) for k, v in hit.items()},
        "n_users": len(users), "n_positives": n_pos,
        "share_positives_not_yet_available": n_new / n_pos,
        "share_positives_already_seen": n_repeat / n_pos,
        "max_reachable_share": 1 - (n_new + n_repeat) / n_pos,
    }
