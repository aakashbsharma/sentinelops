"""Episodic memory: specific past incidents and how they turned out.

Written AFTER an incident resolves; retrieved BEFORE diagnosing a new one
("last time this error pattern appeared, restarting payment-service fixed it
in 4 minutes"). Internal module — agents go through MemoryManager.
"""

from typing import Any

import structlog
from sqlalchemy.ext.asyncio import AsyncSession

from app.agents.schemas import AgentOutcome
from app.memory._retrieval import similarity_search
from app.memory.embeddings import EmbeddingProvider
from app.models.incident import Incident
from app.models.memory import MemoryRecord

logger = structlog.get_logger(__name__)

# Write-policy knobs, named and top-level so tuning them later is a one-line
# diff, not archaeology through nested conditionals.
WRITE_MIN_CONFIDENCE = 0.6
WRITABLE_STATUSES = ("resolved", "closed")

RETRIEVE_DEFAULT_K = 5
RETRIEVE_MIN_SIMILARITY = 0.5


class EpisodicMemoryStore:
    def __init__(self, provider: EmbeddingProvider) -> None:
        self.provider = provider

    async def write(
        self, incident: Incident, outcome: AgentOutcome, db: AsyncSession
    ) -> MemoryRecord | None:
        """Persist a lesson learned — IF it passes the write policy.

        Not everything is worth remembering: memory quality at retrieval time
        is decided here, at write time.
        """
        # POLICY 1 — only terminal incidents. An in-progress incident's
        # "outcome" is noise: the diagnosis may still be wrong, the fix may
        # not have worked yet. Remembering it would teach future agents from
        # unverified guesses.
        if incident.status not in WRITABLE_STATUSES:
            logger.debug(
                "episodic_write_skipped",
                reason="incident_not_terminal",
                status=incident.status,
            )
            return None

        # POLICY 2 — confidence floor. A 0.4-confidence diagnosis that
        # happened to precede resolution is correlation, not a lesson.
        # Enshrining low-confidence guesses as precedent makes future
        # retrievals actively misleading.
        if outcome.confidence < WRITE_MIN_CONFIDENCE:
            logger.debug(
                "episodic_write_skipped",
                reason="confidence_below_floor",
                confidence=outcome.confidence,
            )
            return None

        # POLICY 3 — embed a COMPOSED summary, not raw incident JSON. Raw
        # payloads are full of ids/timestamps that embed to garbage; a
        # structured natural-language summary is what future queries
        # (incident descriptions) will actually be similar to.
        summary = self._compose_summary(incident, outcome)
        embedding = await self.provider.embed(summary)

        record = MemoryRecord(
            memory_type="episodic",
            incident_id=incident.id,
            content=summary,
            embedding=embedding,
            metadata_={
                "severity": incident.severity,
                "outcome": "success" if outcome.status in ("completed",) else "failure",
                "confidence": outcome.confidence,
                "source_status": incident.status,
            },
        )
        db.add(record)
        await db.flush()
        logger.info("episodic_memory_written", incident_id=str(incident.id))
        return record

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
        """For callers (MemoryManager) that already embedded the query —
        avoids embedding the same text once per tier."""
        return await similarity_search(
            db, query_embedding, memory_type="episodic", k=k, min_similarity=min_similarity
        )

    @staticmethod
    def _compose_summary(incident: Incident, outcome: AgentOutcome) -> str:
        payload: dict[str, Any] = outcome.payload
        root_cause = payload.get("root_cause", "root cause not recorded")
        remediation = payload.get("remediation", payload.get("action", "no action recorded"))

        if incident.resolved_at and incident.created_at:
            minutes = (incident.resolved_at - incident.created_at).total_seconds() / 60
            ttr = f"{minutes:.0f} minutes"
        else:
            ttr = "unknown"

        result = "successful" if outcome.status == "completed" else outcome.status
        return (
            f"Incident: {incident.title}. "
            f"Severity: {incident.severity}. "
            f"Root cause: {root_cause}. "
            f"Remediation: {remediation}. "
            f"Outcome: {result}. "
            f"Time to resolution: {ttr}."
        )
