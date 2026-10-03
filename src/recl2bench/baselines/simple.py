"""Non-model baselines scored on the same frozen pools through the same contract."""
from __future__ import annotations

import pandas as pd

from ..rerankers.base import Reranker, RerankResult

DAY_MS = 86_400_000


class PopularityReranker(Reranker):
    """Interaction count in the `window_days` before the user's query time.

    Counts use every rating (any score) in the full timeline, restricted to the
    window, so no future behavior is used. Ties fall back to L1 order.
    """
    name = "popularity"

    def __init__(self, ratings: pd.DataFrame, window_days: int = 90):
        self.ratings, self.window_days = ratings, window_days
        self._cache: dict[int, dict[str, int]] = {}

    def counts_at(self, query_ms: int) -> dict[str, int]:
        if query_ms not in self._cache:
            r = self.ratings
            m = (r.timestamp < query_ms) & (r.timestamp >= query_ms - self.window_days * DAY_MS)
            self._cache[query_ms] = r[m].groupby("parent_asin").size().to_dict()
        return self._cache[query_ms]

    def _score(self, user_profile, candidate_items):
        q_ms = int(pd.Timestamp(user_profile.query_time).value // 1_000_000)
        c = self.counts_at(q_ms)
        return RerankResult(scores=[float(c.get(x.item_id, 0)) for x in candidate_items],
                            model_version=f"popularity-{self.window_days}d")


class OracleReranker(Reranker):
    """Ceiling within the pool: positives first, L1 order otherwise. Uses labels."""
    name = "oracle"

    def __init__(self, positives: dict[str, set]):
        self.positives = positives

    def _score(self, user_profile, candidate_items):
        pos = self.positives[user_profile.user_id]
        return RerankResult(scores=[1.0 if x.item_id in pos else 0.0 for x in candidate_items],
                            model_version="oracle")
