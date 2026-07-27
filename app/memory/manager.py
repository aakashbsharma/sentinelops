"""MemoryManager — the single facade agents and the orchestrator call.

Keeps episodic-vs-semantic internals (and the write policy) behind one
interface, and shapes retrieval output as plain JSON-serializable dicts that
drop straight into the `context` dict AgentLoop.run() consumes (the loop
json.dumps's context when rendering prompts, so no ORM objects may leak out).
"""

from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from app.agents.schemas import AgentOutcome
from app.memory.embeddings import EmbeddingProvider, LocalEmbeddingProvider
from app.memory.episodic import EpisodicMemoryStore
from app.memory.semantic import SemanticMemoryStore
from app.models.incident import Incident
from app.models.memory import MemoryRecord


class MemoryManager:
    def __init__(self, provider: EmbeddingProvider | None = None) -> None:
        provider = provider or LocalEmbeddingProvider()
        self._episodic = EpisodicMemoryStore(provider)
        self._semantic = SemanticMemoryStore(provider)
        self._provider = provider

    async def get_relevant_context(
        self, incident: Incident, db: AsyncSession
    ) -> dict[str, Any]:
        """Retrieve both memory tiers for an incident, shaped for
        AgentLoop.run(context).

        Concurrency note: both tiers search with the SAME query text, so we
        embed exactly once (the expensive, CPU-bound part). The two DB
        queries then run sequentially ON PURPOSE — an AsyncSession is not
        safe for concurrent use (one connection, one transaction), so
        asyncio.gather over the same session would be a race. If retrieval
        latency ever matters, the fix is two sessions, not gather-on-one.
        """
        query = f"{incident.title}. {incident.description or ''}".strip()
        query_embedding = await self._provider.embed(query)

        episodic = await self._episodic.retrieve_by_embedding(query_embedding, db)
        semantic = await self._semantic.retrieve_by_embedding(query_embedding, db)

        return {
            "episodic_memories": [self._to_context_dict(r, s) for r, s in episodic],
            "semantic_memories": [self._to_context_dict(r, s) for r, s in semantic],
        }

    async def write_outcome(
        self, incident: Incident, outcome: AgentOutcome, db: AsyncSession
    ) -> None:
        """The single episodic write path — the write policy lives in exactly
        one place (EpisodicMemoryStore.write)."""
        await self._episodic.write(incident, outcome, db)

    @staticmethod
    def _to_context_dict(record: MemoryRecord, similarity: float) -> dict[str, Any]:
        return {
            "content": record.content,
            "similarity": round(similarity, 3),
            "metadata": record.metadata_ or {},
            "memory_id": str(record.id),
        }
