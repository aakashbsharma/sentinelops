"""Shared similarity-search helper for both memory stores (internal).

Two code paths:
  - PostgreSQL: pgvector's `<=>` cosine-distance operator, served by the HNSW
    index — the real production path.
  - Anything else (SQLite unit tests): load candidates and rank with
    Python-side cosine. Correct but O(n); exists so store/policy logic is
    testable without a Postgres container.
"""

from typing import Any

from sqlalchemy import Float, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.memory.embeddings import cosine_similarity
from app.models.memory import MemoryRecord
from pgvector.sqlalchemy import Vector
from sqlalchemy import bindparam

def _vector_literal(embedding: list[float]) -> str:
    """pgvector's textual input format: '[0.1,0.2,...]'."""
    return "[" + ",".join(repr(float(x)) for x in embedding) + "]"


async def similarity_search(
    db: AsyncSession,
    query_embedding: list[float],
    memory_type: str,
    k: int,
    min_similarity: float,
) -> list[tuple[MemoryRecord, float]]:
    """Return up to k (record, similarity) pairs above min_similarity,
    best match first.

    min_similarity exists because an irrelevant memory is WORSE than no
    memory: top-k with no floor will happily hand the diagnostician a
    completely unrelated past incident as if it were precedent.
    """
    if _is_postgres(db):
        return await _search_pgvector(
            db, query_embedding, memory_type, k, min_similarity
        )
    return await _search_python(db, query_embedding, memory_type, k, min_similarity)


def _is_postgres(db: AsyncSession) -> bool:
    bind: Any = db.get_bind()
    return bind.dialect.name == "postgresql"


async def _search_pgvector(
    db: AsyncSession,
    query_embedding: list[float],
    memory_type: str,
    k: int,
    min_similarity: float,
) -> list[tuple[MemoryRecord, float]]:
    # cosine DISTANCE (0 = identical, 2 = opposite); similarity = 1 - distance.
    query_vector = bindparam(
        "query_embedding",
        value=query_embedding,
        type_=Vector(len(query_embedding)),
    )

    distance = MemoryRecord.embedding.op("<=>", return_type=Float)(
        query_vector
    )
    max_distance = 1.0 - min_similarity

    stmt = (
        select(MemoryRecord, distance.label("distance"))
        .where(
            MemoryRecord.memory_type == memory_type,
            MemoryRecord.embedding.is_not(None),
            distance <= max_distance,
        )
        .order_by(distance)
        .limit(k)
    )
    rows = (await db.execute(stmt)).all()
    return [(record, 1.0 - dist) for record, dist in rows]


async def _search_python(
    db: AsyncSession,
    query_embedding: list[float],
    memory_type: str,
    k: int,
    min_similarity: float,
) -> list[tuple[MemoryRecord, float]]:
    stmt = select(MemoryRecord).where(
        MemoryRecord.memory_type == memory_type,
        MemoryRecord.embedding.is_not(None),
    )
    records = (await db.execute(stmt)).scalars().all()

    scored = [
        (record, cosine_similarity(query_embedding, record.embedding))
        for record in records
        if record.embedding
    ]
    scored = [(r, s) for r, s in scored if s >= min_similarity]
    scored.sort(key=lambda pair: pair[1], reverse=True)
    return scored[:k]
