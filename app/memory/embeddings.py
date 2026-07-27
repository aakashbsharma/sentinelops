"""Embedding provider abstraction.

TRADEOFF (worth explaining in an interview): embeddings run LOCALLY via
sentence-transformers (all-MiniLM-L6-v2, 384 dims) instead of a hosted API
(e.g. Voyage AI, which Anthropic recommends). Local wins here because:
  - the Stage 8 eval harness replays dozens of incidents per run — at API
    prices that's real money for zero quality benefit at this corpus size;
  - retrieval quality at "hundreds of memories" scale is dominated by the
    write policy and thresholds, not by embedding model quality;
  - no network dependency in tests.
The cost: MiniLM is weaker than hosted models on long/technical text, and
384 dims caps recall on a large corpus. When that matters, swapping providers
is a one-class change: implement EmbeddingProvider with an HTTP call and a
different `dimension`, then run one Alembic migration to resize the vector
column.
"""

import asyncio
import hashlib
import math
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from app.models.memory import EMBEDDING_DIM

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer


class EmbeddingProvider(ABC):
    """Swap-point for embedding backends. Everything downstream (stores,
    consolidation, seeding) depends only on this interface."""

    @property
    @abstractmethod
    def dimension(self) -> int: ...

    @abstractmethod
    async def embed(self, text: str) -> list[float]: ...

    @abstractmethod
    async def embed_batch(self, texts: list[str]) -> list[list[float]]: ...


# Module-level singleton: the model costs ~100MB RAM and ~2s to load; doing it
# per-call (or per-instance) would dominate every retrieval's latency.
_model: "SentenceTransformer | None" = None


def _get_model() -> "SentenceTransformer":
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer

        _model = SentenceTransformer("all-MiniLM-L6-v2")
    return _model


class LocalEmbeddingProvider(EmbeddingProvider):
    """sentence-transformers all-MiniLM-L6-v2 — 384 dims, local, free."""

    @property
    def dimension(self) -> int:
        return EMBEDDING_DIM

    async def embed(self, text: str) -> list[float]:
        return (await self.embed_batch([text]))[0]

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        # The model is sync + CPU-bound; run in a worker thread so we don't
        # block the event loop (which would stall every other agent loop).
        model = await asyncio.to_thread(_get_model)
        vectors = await asyncio.to_thread(
            model.encode, texts, normalize_embeddings=True
        )
        return [v.tolist() for v in vectors]


class HashingEmbeddingProvider(EmbeddingProvider):
    """Deterministic, dependency-free bag-of-ngrams embedding.

    NOT for production retrieval quality — this exists so store/policy logic
    can be unit-tested fast without downloading a model. Token-overlap texts
    still land close in cosine space, which is all those tests need.
    """

    def __init__(self, dimension: int = EMBEDDING_DIM) -> None:
        self._dimension = dimension

    @property
    def dimension(self) -> int:
        return self._dimension

    async def embed(self, text: str) -> list[float]:
        vec = [0.0] * self._dimension
        tokens = text.lower().split()
        # unigrams + bigrams: enough signal for similarity sanity checks
        grams = tokens + [f"{a} {b}" for a, b in zip(tokens, tokens[1:])]
        for gram in grams:
            digest = hashlib.md5(gram.encode()).digest()
            index = int.from_bytes(digest[:4], "little") % self._dimension
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vec[index] += sign
        norm = math.sqrt(sum(x * x for x in vec)) or 1.0
        return [x / norm for x in vec]

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [await self.embed(t) for t in texts]


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Pure-python cosine similarity — used by the non-Postgres retrieval
    fallback and by consolidation's pairwise comparison."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)
