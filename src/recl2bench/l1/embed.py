"""Item embeddings with Qwen3-Embedding-0.6B, cached to disk keyed by model + text hash."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Protocol

import numpy as np


class Encoder(Protocol):
    name: str
    def encode(self, texts: list[str]) -> np.ndarray: ...


class STEncoder:
    """Items are documents, so no query prompt (Qwen3-Embedding applies prompts to queries only)."""

    def __init__(self, model: str = "Qwen/Qwen3-Embedding-0.6B", revision: str | None = None,
                 batch_size: int = 64, device: str | None = None):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model, revision=revision, device=device)
        self.name = f"{model}@{revision}"
        self.batch_size = batch_size

    def encode(self, texts):
        return self.model.encode(texts, batch_size=self.batch_size, normalize_embeddings=True,
                                 convert_to_numpy=True, show_progress_bar=True).astype(np.float32)


def normalize(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.clip(n, 1e-12, None)


def embed_items(ids: list[str], texts: list[str], encoder: Encoder, cache_dir: str | Path):
    key = hashlib.sha256((encoder.name + "\x00" + "\x00".join(texts)).encode()).hexdigest()[:16]
    cache_dir = Path(cache_dir)
    npy, meta = cache_dir / f"item_emb_{key}.npy", cache_dir / f"item_emb_{key}.json"
    if npy.exists() and meta.exists():
        info = json.loads(meta.read_text())
        if info["ids"] == ids:
            return np.load(npy), key
    vecs = normalize(encoder.encode(texts))
    cache_dir.mkdir(parents=True, exist_ok=True)
    np.save(npy, vecs)
    meta.write_text(json.dumps({"encoder": encoder.name, "ids": ids}))
    return vecs, key
