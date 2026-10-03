"""CLM-v0.1-8B adapter (profile as state, items as actions), served by vLLM + clm-serve.

Stub: the clm-serve request shape must come from the CLM model card. Item
embeddings are cacheable across users since each item is encoded separately.
Scores are in-set probabilities, so CLM is excluded from calibration metrics.
"""
from __future__ import annotations

from .base import Reranker


class CLMReranker(Reranker):
    name = "clm"

    def __init__(self, serve_url: str, checkpoint: str = "CLM-v0.1-8B"):
        self.serve_url, self.checkpoint = serve_url, checkpoint

    def _score(self, user_profile, candidate_items):
        raise NotImplementedError("implement against clm-serve's documented API")
