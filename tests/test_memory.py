"""Stage 3 memory-system tests.

Store/policy/consolidation tests use HashingEmbeddingProvider: deterministic,
instant, no model download — they verify STORE LOGIC (write policy, ranking,
thresholds, dedup), not embedding quality. The real LocalEmbeddingProvider
gets one sanity test that is skipped if sentence-transformers isn't installed.
"""

from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.agents.schemas import AgentOutcome
from app.memory.consolidation import consolidate
from app.memory.embeddings import HashingEmbeddingProvider, cosine_similarity
from app.memory.episodic import EpisodicMemoryStore
from app.memory.manager import MemoryManager
from app.memory.semantic import SemanticMemoryStore
from app.models.incident import Incident
from app.models.memory import MemoryRecord

RUNBOOKS_DIR = Path(__file__).resolve().parent.parent / "runbooks"


def make_incident(
    title: str = "payment-service pods OOMKilled repeatedly",
    status: str = "resolved",
    description: str = "payment-service pods restarting with OOMKilled, checkout failing",
) -> Incident:
    now = datetime.now(timezone.utc)
    return Incident(
        title=title,
        description=description,
        source="synthetic",
        severity="high",
        status=status,
        created_at=now - timedelta(minutes=30),
        resolved_at=now if status in ("resolved", "closed") else None,
    )


def make_outcome(confidence: float = 0.9, **payload: Any) -> AgentOutcome:
    return AgentOutcome(
        status="completed",
        summary="diagnosed and remediated",
        confidence=confidence,
        payload={
            "root_cause": "memory leak in payment retry buffer",
            "remediation": "restarted payment-service and capped retry buffer",
            **payload,
        },
    )


@pytest.fixture
def provider() -> HashingEmbeddingProvider:
    return HashingEmbeddingProvider()


# --------------------------------------------------------------------- #
# Embedding providers
# --------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_local_embedding_provider_sanity() -> None:
    """Real model: consistent dimension; near-duplicates embed close together.
    A sanity check, not an eval."""
    pytest.importorskip("sentence_transformers")
    from app.memory.embeddings import LocalEmbeddingProvider

    local = LocalEmbeddingProvider()
    texts = [
        "database connection pool exhausted, requests timing out",
        "db connection pool ran out, requests are timing out",  # near-duplicate
        "the office coffee machine is broken again",  # unrelated
    ]
    vectors = await local.embed_batch(texts)

    assert all(len(v) == local.dimension == 384 for v in vectors)
    near_dup = cosine_similarity(vectors[0], vectors[1])
    unrelated = cosine_similarity(vectors[0], vectors[2])
    assert near_dup > 0.8
    assert near_dup > unrelated + 0.3


@pytest.mark.asyncio
async def test_hashing_provider_dimension_and_determinism(
    provider: HashingEmbeddingProvider,
) -> None:
    a = await provider.embed("connection pool exhausted")
    b = await provider.embed("connection pool exhausted")
    assert len(a) == provider.dimension == 384
    assert a == b  # deterministic
    assert cosine_similarity(a, b) == pytest.approx(1.0)


# --------------------------------------------------------------------- #
# Episodic write policy
# --------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_write_policy_rejects_unresolved_incident(
    db_session: AsyncSession, provider: HashingEmbeddingProvider
) -> None:
    store = EpisodicMemoryStore(provider)
    incident = make_incident(status="diagnosing")  # not terminal
    db_session.add(incident)
    await db_session.flush()

    record = await store.write(incident, make_outcome(confidence=0.9), db_session)

    assert record is None
    count = (await db_session.execute(select(MemoryRecord))).scalars().all()
    assert count == []


@pytest.mark.asyncio
async def test_write_policy_rejects_low_confidence(
    db_session: AsyncSession, provider: HashingEmbeddingProvider
) -> None:
    store = EpisodicMemoryStore(provider)
    incident = make_incident(status="resolved")
    db_session.add(incident)
    await db_session.flush()

    record = await store.write(incident, make_outcome(confidence=0.4), db_session)

    assert record is None


@pytest.mark.asyncio
async def test_write_policy_accepts_resolved_confident_outcome(
    db_session: AsyncSession, provider: HashingEmbeddingProvider
) -> None:
    store = EpisodicMemoryStore(provider)
    incident = make_incident(status="resolved")
    db_session.add(incident)
    await db_session.flush()

    record = await store.write(incident, make_outcome(confidence=0.85), db_session)

    assert record is not None
    assert record.memory_type == "episodic"
    assert record.incident_id == incident.id
    assert record.embedding is not None
    # composed summary, not raw JSON
    assert "Root cause: memory leak in payment retry buffer" in record.content
    assert record.metadata_["outcome"] == "success"


# --------------------------------------------------------------------- #
# Episodic round-trip
# --------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_episodic_write_then_retrieve_roundtrip(
    db_session: AsyncSession, provider: HashingEmbeddingProvider
) -> None:
    store = EpisodicMemoryStore(provider)

    incident = make_incident(
        title="payment-service pods OOMKilled repeatedly",
        description="payment-service pods restarting with OOMKilled, checkout failing",
    )
    unrelated = make_incident(
        title="TLS certificate expired on public ingress",
        description="browsers rejecting the certificate on the marketing site",
    )
    db_session.add_all([incident, unrelated])
    await db_session.flush()

    written = await store.write(incident, make_outcome(), db_session)
    written_unrelated = await store.write(unrelated, make_outcome(), db_session)
    assert written is not None and written_unrelated is not None

    # Semantically similar query — not the exact stored text.
    results = await store.retrieve(
        "payment service pods keep getting OOMKilled and restarting",
        db_session,
        k=5,
        min_similarity=0.05,  # hashing embeddings score lower than real ones
    )

    assert results, "expected the OOM memory to come back"
    top_record, top_similarity = results[0]
    assert top_record.id == written.id
    assert top_similarity > 0.05
    # ranked above the unrelated TLS memory (which may not even pass the floor)
    ids = [r.id for r, _ in results]
    assert ids.index(written.id) == 0


# --------------------------------------------------------------------- #
# Semantic retrieval over seeded runbooks
# --------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_semantic_retrieval_ranks_relevant_runbook(
    db_session: AsyncSession, provider: HashingEmbeddingProvider
) -> None:
    store = SemanticMemoryStore(provider)
    records = await store.seed_directory(RUNBOOKS_DIR, db_session)
    assert len(records) >= 3  # three runbooks, at least one chunk each

    results = await store.retrieve(
        "database connections timing out and requests failing",
        db_session,
        k=3,
        min_similarity=0.0,  # test ranking, not the threshold
    )

    assert results
    top_record, _ = results[0]
    assert top_record.metadata_["source_file"] == "database-connection-pool-exhaustion.md"


# --------------------------------------------------------------------- #
# MemoryManager facade
# --------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_manager_context_shape_is_json_serializable(
    db_session: AsyncSession, provider: HashingEmbeddingProvider
) -> None:
    import json

    manager = MemoryManager(provider=provider)
    incident = make_incident()
    db_session.add(incident)
    await db_session.flush()

    await manager.write_outcome(incident, make_outcome(), db_session)
    context = await manager.get_relevant_context(incident, db_session)

    assert set(context) == {"episodic_memories", "semantic_memories"}
    json.dumps(context)  # must drop into AgentLoop context without ORM leakage
    for memory in context["episodic_memories"]:
        assert {"content", "similarity", "metadata", "memory_id"} <= set(memory)


# --------------------------------------------------------------------- #
# Consolidation
# --------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_consolidation_removes_older_near_duplicate(
    db_session: AsyncSession, provider: HashingEmbeddingProvider
) -> None:
    base_time = datetime.now(timezone.utc)
    duplicate_text = (
        "Incident: payment-service OOMKilled. Root cause: retry buffer leak. "
        "Remediation: restart and cap buffer. Outcome: successful."
    )
    embedding = await provider.embed(duplicate_text)
    distinct_embedding = await provider.embed(
        "Incident: ingress TLS certificate expired. Remediation: rotate cert."
    )

    older = MemoryRecord(
        memory_type="episodic",
        content=duplicate_text,
        embedding=embedding,
        created_at=base_time - timedelta(days=7),
    )
    newer = MemoryRecord(
        memory_type="episodic",
        content=duplicate_text,
        embedding=embedding,  # identical -> similarity 1.0 > 0.95
        created_at=base_time,
    )
    distinct = MemoryRecord(
        memory_type="episodic",
        content="TLS cert incident",
        embedding=distinct_embedding,
        created_at=base_time - timedelta(days=3),
    )
    db_session.add_all([older, newer, distinct])
    await db_session.flush()

    deleted = await consolidate(db_session)

    assert deleted == 1
    remaining = (await db_session.execute(select(MemoryRecord))).scalars().all()
    remaining_ids = {r.id for r in remaining}
    assert newer.id in remaining_ids  # most recent duplicate kept
    assert distinct.id in remaining_ids  # unrelated memory untouched
    assert older.id not in remaining_ids  # older duplicate removed
