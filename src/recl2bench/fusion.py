"""Track C: rank fusion of a reranker with a behavioural partner (L1 order or popularity).

fused(item) = w / (k + rank_reranker) + (1 - w) / (k + rank_partner),  ranks 1-based.

w = 0 reproduces the partner's order, w = 1 the reranker's. Ties break by L1 rank.
The weight is chosen on validation only; test is scored once with that weight.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .eval.metrics import order_by_scores, user_metrics

RRF_K = 60
GRID = tuple(round(x, 2) for x in np.linspace(0, 1, 11))


def ranks(scores: pd.DataFrame) -> pd.Series:
    """1-based rank per (user, item): descending score, ties by L1 rank."""
    s = scores.assign(score=scores["score"].fillna(-np.inf))
    s = s.sort_values(["user_id", "score", "l1_rank"], ascending=[True, False, True], kind="stable")
    s["rank"] = s.groupby("user_id").cumcount() + 1
    return s.set_index(["user_id", "parent_asin"])["rank"]


def fuse_scores(reranker: pd.DataFrame, partner: pd.DataFrame, w: float, k: int = RRF_K) -> pd.DataFrame:
    """Both inputs: user_id, parent_asin, l1_rank, score (same candidates)."""
    r, p = ranks(reranker), ranks(partner)
    if len(r) != len(p) or not r.index.sort_values().equals(p.index.sort_values()):
        raise ValueError("reranker and partner runs cover different candidates")
    base = reranker[["user_id", "parent_asin", "l1_rank"]].set_index(["user_id", "parent_asin"])
    fused = w / (k + r) + (1 - w) / (k + p.reindex(r.index))
    return base.assign(score=fused.reindex(base.index)).reset_index()


def per_user_metrics(scores: pd.DataFrame, positives: dict[str, set]) -> pd.DataFrame:
    rows = []
    for u, g in scores.groupby("user_id", sort=True):
        g = g.sort_values("l1_rank")
        ranked = order_by_scores(g.parent_asin.tolist(), g.score.to_numpy())
        rows.append({"user_id": u, **user_metrics(ranked, positives[u])})
    return pd.DataFrame(rows)


def tune(reranker: pd.DataFrame, partner: pd.DataFrame, positives: dict[str, set],
         grid=GRID, k: int = RRF_K) -> tuple[float, pd.DataFrame]:
    """Mean validation NDCG@10 per weight; ties go to the smaller weight (closer to partner)."""
    rows = [{"w": w, "ndcg@10": per_user_metrics(fuse_scores(reranker, partner, w, k), positives)["ndcg@10"].mean()}
            for w in grid]
    t = pd.DataFrame(rows)
    best = t.loc[t["ndcg@10"].round(12).idxmax(), "w"]
    return float(best), t
