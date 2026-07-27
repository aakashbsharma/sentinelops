"""Semantic memory: general knowledge not tied to any incident — runbooks,
architecture docs, known-issues docs. Seeded once (scripts/seed_runbooks.py),
retrieved often. Internal module — agents go through MemoryManager.
"""

from pathlib import Path

import structlog
from sqlalchemy.ext.asyncio import AsyncSession

from app.memory._retrieval import similarity_search
from app.memory.embeddings import EmbeddingProvider
from app.models.memory import MemoryRecord

logger = structlog.get_logger(__name__)

RETRIEVE_DEFAULT_K = 3
RETRIEVE_MIN_SIMILARITY = 0.35  # docs phrase things differently than alerts do


class SemanticMemoryStore:
    def __init__(self, provider: EmbeddingProvider) -> None:
        self.provider = provider

    async def retrieve(
        self,
        query: str,
        db: AsyncSession,
        k: int = RETRIEVE_DEFAULT_K,
        min_similarity: float = RETRIEVE_MIN_SIMILARITY,
    ) -> list[tuple[MemoryRecord, float]]:
        query_embedding = await self.provider.embed(query)
        return await self.retrieve_by_embedding(query_embedding, db, k, min_similarity)

    async def retrieve_by_embedding(
        self,
        query_embedding: list[float],
        db: AsyncSession,
        k: int = RETRIEVE_DEFAULT_K,
        min_similarity: float = RETRIEVE_MIN_SIMILARITY,
    ) -> list[tuple[MemoryRecord, float]]:
        """For callers (MemoryManager) that already embedded the query."""
        return await similarity_search(
            db, query_embedding, memory_type="semantic", k=k, min_similarity=min_similarity
        )

    async def seed_directory(
        self, runbooks_dir: Path, db: AsyncSession
    ) -> list[MemoryRecord]:
        """Chunk + embed + store every .md file in a directory.

        KNOWN LIMITATION: chunking is naive paragraph splitting. It can cut a
        procedure off from its heading, and long paragraphs aren't split at
        all. With more time: heading-aware chunking (keep each section under
        its H2/H3 title), a max-token cap with overlap, and prepending the
        doc title to every chunk so orphaned chunks stay attributable.
        """
        records: list[MemoryRecord] = []
        for md_file in sorted(runbooks_dir.glob("*.md")):
            chunks = self._chunk_markdown(md_file.read_text(encoding="utf-8"))
            embeddings = await self.provider.embed_batch(chunks)
            for chunk, embedding in zip(chunks, embeddings):
                record = MemoryRecord(
                    memory_type="semantic",
                    incident_id=None,  # general knowledge, no incident
                    content=chunk,
                    embedding=embedding,
                    metadata_={"source_file": md_file.name},
                )
                db.add(record)
                records.append(record)
            logger.info("runbook_seeded", file=md_file.name, chunks=len(chunks))
        await db.flush()
        return records

    @staticmethod
    def _chunk_markdown(text: str) -> list[str]:
        """Paragraph-level chunks (split on blank lines), tiny ones merged
        forward so lone headings travel with their body."""
        raw = [p.strip() for p in text.split("\n\n") if p.strip()]
        chunks: list[str] = []
        buffer = ""
        for para in raw:
            buffer = f"{buffer}\n\n{para}".strip() if buffer else para
            if len(buffer) >= 200:  # merge until we have a meaningful chunk
                chunks.append(buffer)
                buffer = ""
        if buffer:
            chunks.append(buffer)
        return chunks
