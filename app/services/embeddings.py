"""
Local sentence embeddings (no external API, no per-call cost).

Used for two things:
1. Dedup - the same paper often surfaces from both ArXiv and web search
   phrased differently; embedding the title+abstract lets us merge near-
   duplicates that a raw string match would miss.
2. Memory recall - future queries about an overlapping topic can reuse
   cached paper summaries instead of re-fetching and re-extracting.

Loaded lazily and cached as a module-level singleton since the model
load (~90MB) is the expensive part, not the encode calls.
"""
from __future__ import annotations

import asyncio
from functools import lru_cache

import numpy as np

from app.config import Settings


@lru_cache(maxsize=1)
def _load_model(model_name: str, cache_dir: str):
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_name, cache_folder=cache_dir)


class EmbeddingService:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    async def embed(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, 384), dtype=np.float32)
        model = _load_model(self._settings.embedding_model_name, self._settings.embedding_cache_dir)
        return await asyncio.to_thread(model.encode, texts, normalize_embeddings=True)

    @staticmethod
    def cosine_similarity_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        # Vectors are pre-normalized, so cosine similarity is a plain dot product.
        return a @ b.T
