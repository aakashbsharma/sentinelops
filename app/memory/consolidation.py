"""Episodic memory consolidation.

WHY THIS EXISTS (a real failure mode, not a hypothetical): a recurring
incident type — say the payment-service OOM that fires weekly — writes a
near-identical episodic memory every time it resolves. Retrieval is top-k by
similarity, so after a few weeks those duplicates occupy ALL k slots for any
payments-related query and drown out the one different-but-relevant memory
that would actually help. Naive episodic memory gets WORSE with use;
consolidation is the fix: keep the most recent instance of each near-
duplicate cluster, delete the rest.
"""

import asyncio

import structlog
from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.celery_app import celery_app
from app.memory.embeddings import cosine_similarity
from app.models.memory import MemoryRecord

logger = structlog.get_logger(__name__)

# Above this cosine similarity two memories are "the same lesson".
# 0.95 is deliberately strict: false merges destroy information permanently,
# false keeps just cost a duplicate until the next run.
NEAR_DUPLICATE_THRESHOLD = 0.95


async def consolidate(db: AsyncSession) -> int:
    """Delete older near-duplicates among episodic memories.

    Pairwise comparison is O(n^2) in Python — fine at this corpus size
    (hundreds of memories); at real scale you'd cluster in the DB with a
    pgvector self-join on `<=>` instead.
    """
    stmt = (
        select(MemoryRecord)
        .where(
            MemoryRecord.memory_type == "episodic",
            MemoryRecord.embedding.is_not(None),
        )
        .order_by(MemoryRecord.created_at.desc())  # newest first -> newest wins
    )
    records = list((await db.execute(stmt)).scalars().all())

    to_delete: set = set()
    for i, newer in enumerate(records):
        if newer.id in to_delete:
            continue
        for older in records[i + 1 :]:
            if older.id in to_delete:
                continue
            similarity = cosine_similarity(list(newer.embedding), list(older.embedding))
            if similarity >= NEAR_DUPLICATE_THRESHOLD:
                to_delete.add(older.id)

    if to_delete:
        await db.execute(delete(MemoryRecord).where(MemoryRecord.id.in_(to_delete)))
        await db.flush()

    logger.info("episodic_consolidation_done", deleted=len(to_delete), kept=len(records) - len(to_delete))
    return len(to_delete)


@celery_app.task(name="memory.consolidate_episodic")
def consolidate_episodic_memory() -> int:
    """Celery entrypoint. Creates its own engine/session because Celery
    workers are separate processes — they must not share the API's pool.
    Scheduling (celery beat) gets wired in a later stage."""

    async def _run() -> int:
        # Imported here so importing this module (e.g. in tests) doesn't
        # require a reachable database.
        from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

        from app.core.config import get_settings

        engine = create_async_engine(get_settings().DATABASE_URL)
        factory = async_sessionmaker(engine, expire_on_commit=False)
        try:
            async with factory() as session:
                deleted = await consolidate(session)
                await session.commit()
                return deleted
        finally:
            await engine.dispose()

    return asyncio.run(_run())
