"""Seed semantic memory from the runbooks/ directory.

Usage (after `docker compose up -d` and `alembic upgrade head`):

    python -m scripts.seed_runbooks

Idempotency: existing semantic records seeded from the same files are
replaced (delete-then-insert keyed on metadata source_file), so re-running
after editing a runbook doesn't duplicate chunks.
"""

import asyncio
from pathlib import Path

from sqlalchemy import delete

from app.core.database import async_session_factory
from app.core.logging import configure_logging
from app.memory.embeddings import LocalEmbeddingProvider
from app.memory.semantic import SemanticMemoryStore
from app.models.memory import MemoryRecord

RUNBOOKS_DIR = Path(__file__).resolve().parent.parent / "runbooks"


async def main() -> None:
    configure_logging()
    store = SemanticMemoryStore(LocalEmbeddingProvider())

    async with async_session_factory() as session:
        seeded_files = [p.name for p in RUNBOOKS_DIR.glob("*.md")]
        await session.execute(
            delete(MemoryRecord).where(
                MemoryRecord.memory_type == "semantic",
                MemoryRecord.metadata_["source_file"].as_string().in_(seeded_files),
            )
        )
        records = await store.seed_directory(RUNBOOKS_DIR, session)
        await session.commit()

    print(f"Seeded {len(records)} semantic memory chunks from {len(seeded_files)} runbooks.")


if __name__ == "__main__":
    asyncio.run(main())
