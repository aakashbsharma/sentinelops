import uuid

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import AgentMessage, ApprovalRequest, Incident


@pytest.mark.asyncio
async def test_create_and_query_incident(db_session: AsyncSession) -> None:
    incident = Incident(
        title="High p99 latency on checkout-service",
        description="p99 latency exceeded 2s for 5 consecutive minutes",
        source="synthetic",
        severity="high",
        status="open",
        raw_payload={"metric": "p99_latency_ms", "value": 2140, "threshold": 2000},
    )
    db_session.add(incident)
    await db_session.flush()

    result = await db_session.execute(
        select(Incident).where(Incident.id == incident.id)
    )
    fetched = result.scalar_one()

    assert isinstance(fetched.id, uuid.UUID)
    assert fetched.title == "High p99 latency on checkout-service"
    assert fetched.severity == "high"
    assert fetched.status == "open"
    assert fetched.source == "synthetic"
    assert fetched.raw_payload == {
        "metric": "p99_latency_ms",
        "value": 2140,
        "threshold": 2000,
    }
    assert fetched.created_at is not None
    assert fetched.resolved_at is None


@pytest.mark.asyncio
async def test_incident_relationships_and_defaults(db_session: AsyncSession) -> None:
    incident = Incident(title="Pod crash loop", severity="critical", status="open")
    message = AgentMessage(
        incident=incident,
        agent_name="triage-1",
        role="triage",
        content={"assessment": "crash loop in payments namespace"},
        confidence=0.82,
    )
    approval = ApprovalRequest(
        incident=incident,
        proposed_action={"tool": "restart_service", "target": "payments", "dry_run": True},
        risk_level="medium",
    )
    db_session.add(incident)
    await db_session.flush()

    assert message.needs_more_data is False
    assert approval.status == "pending"
    assert incident.messages == [message]
    assert incident.approval_requests == [approval]
    assert message.incident_id == incident.id
