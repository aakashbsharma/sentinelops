"""Orchestrator integration tests.

All LLM-driven agents are faked with Stage 2/4's scripted _plan_step
injection; the Executor runs for REAL (it has no LLM — its planner is the
approved action queue), against the mock infra tools. The DB is the real
(test) database: transitions, AgentMessages and ApprovalRequests are asserted
from actual rows.
"""

import json
from typing import Any

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.agents.base import AgentLoop
from app.agents.diagnostician import DiagnosticianAgent
from app.agents.executor import ExecutorAgent
from app.agents.schemas import AgentOutcome, LoopConfig
from app.agents.remediation_planner import RemediationPlannerAgent
from app.agents.triage import TriageAgent
from app.core.config import get_settings
from app.models.agent_message import AgentMessage
from app.models.approval import ApprovalRequest
from app.models.incident import Incident
from app.orchestrator.core import Orchestrator
from app.orchestrator.state_machine import IllegalTransitionError, transition
from tests._scripting import make_infra_registry, plan_final, scripted

CONFIG = LoopConfig(max_iterations=6, token_budget=10_000, reflect_every_step=False)

# ---------------------------------------------------------------------- #
# Scripted agent payloads (each is one final answer for one invocation)
# ---------------------------------------------------------------------- #

TRIAGE_OK = {
    "severity": "high",
    "category": "resource_exhaustion",
    "confidence": 0.9,
    "needs_more_data": False,
    "reasoning": "latency ramp with pool timeouts",
}
TRIAGE_NEEDS_DATA = {
    "severity": "medium",
    "category": "unknown",
    "confidence": 0.3,
    "needs_more_data": True,
    "reasoning": "alert alone is ambiguous",
}
DIAGNOSIS_OK = {
    "root_cause": "connection pool exhaustion in payment-service",
    "evidence": ["pool timeout errors", "latency_p99 ramp"],
    "contributing_factors": ["pool size 10"],
    "confidence": 0.85,
    "needs_more_data": False,
}
DIAGNOSIS_LOW_CONFIDENCE = {
    "root_cause": "possibly a bad deploy, unclear",
    "evidence": [],
    "contributing_factors": [],
    "confidence": 0.3,
    "needs_more_data": False,
}
PLAN_OK = {
    "proposed_actions": [
        {
            "action_type": "restart",
            "target": "payment-service",
            "tool_name": "restart_service",
            "parameters": {"service_name": "payment-service"},
            "rationale": "clear the exhausted connection pool",
        }
    ],
    "risk_level": "medium",
    "requires_approval": True,
    "rollback_plan": "re-page on-call if latency does not recover in 10m",
    "confidence": 0.9,
}

HAPPY_SCRIPTS: dict[str, list[dict[str, Any]]] = {
    "triage": [TRIAGE_OK],
    "diagnostician": [DIAGNOSIS_OK],
    "remediation_planner": [PLAN_OK],
}

SCRIPTED_CLASSES: dict[str, type[AgentLoop]] = {
    "triage": TriageAgent,
    "diagnostician": DiagnosticianAgent,
    "remediation_planner": RemediationPlannerAgent,
}


class FakeMemoryManager:
    """LLM-free, embedding-free stand-in recording write_outcome calls."""

    def __init__(self) -> None:
        self.write_calls: list[tuple[Any, AgentOutcome]] = []

    async def get_relevant_context(self, incident: Incident, db: AsyncSession) -> dict:
        return {"episodic_memories": [], "semantic_memories": []}

    async def write_outcome(
        self, incident: Incident, outcome: AgentOutcome, db: AsyncSession
    ) -> None:
        self.write_calls.append((incident.id, outcome))


def make_agent_factory(scripts: dict[str, list[dict[str, Any]]], invoked: list[str]):
    """scripts[role][i] = the payload the role's i-th invocation answers with.
    The Executor is never scripted — it runs for real (deterministic)."""
    calls: dict[str, int] = {}

    def factory(role: str, incident: Incident) -> AgentLoop:
        invoked.append(role)
        if role == "executor":
            return ExecutorAgent(CONFIG, make_infra_registry())
        idx = calls.get(role, 0)
        calls[role] = idx + 1
        payloads = scripts[role]
        payload = payloads[min(idx, len(payloads) - 1)]
        return scripted(SCRIPTED_CLASSES[role])(
            CONFIG, make_infra_registry(), script=[plan_final(json.dumps(payload))]
        )

    return factory


async def make_incident(db: AsyncSession) -> Incident:
    incident = Incident(
        title="payment-service p99 latency above 2s",
        description="checkout requests timing out",
        source="synthetic",
        severity="unknown",
        status="open",
        raw_payload={"scenario": "connection_pool_exhaustion"},
    )
    db.add(incident)
    await db.flush()
    return incident


async def roles_seen(db: AsyncSession, incident: Incident) -> list[str]:
    rows = await db.execute(
        select(AgentMessage.role).where(AgentMessage.incident_id == incident.id)
    )
    return [r[0] for r in rows]


async def orchestrator_reasons(db: AsyncSession, incident: Incident) -> list[str]:
    rows = await db.execute(
        select(AgentMessage.content).where(
            AgentMessage.incident_id == incident.id,
            AgentMessage.role == "orchestrator",
        )
    )
    return [r[0].get("reasoning", "") for r in rows]


# ---------------------------------------------------------------------- #
# State machine
# ---------------------------------------------------------------------- #


def test_illegal_transition_raises() -> None:
    incident = Incident(title="t", severity="low", status="open")
    with pytest.raises(IllegalTransitionError, match="open.*resolved"):
        transition(incident, "resolved")  # nonsense jump, skips every gate
    assert incident.status == "open"  # unchanged — never silently applied


def test_legal_transition_applies() -> None:
    incident = Incident(title="t", severity="low", status="open")
    transition(incident, "triaging")
    assert incident.status == "triaging"


def test_unknown_status_raises() -> None:
    incident = Incident(title="t", severity="low", status="bogus")
    with pytest.raises(IllegalTransitionError, match="unknown status"):
        transition(incident, "triaging")


# ---------------------------------------------------------------------- #
# Full happy path: open -> ... -> awaiting_approval -> (approve) -> resolved
# ---------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_happy_path_open_to_resolved(db_session: AsyncSession) -> None:
    incident = await make_incident(db_session)
    invoked: list[str] = []
    memory = FakeMemoryManager()
    orch = Orchestrator(
        memory=memory, agent_factory=make_agent_factory(HAPPY_SCRIPTS, invoked)
    )

    await orch.run_incident(incident.id, db_session)

    # Stopped at the approval gate, with triage's severity now on the record.
    assert incident.status == "awaiting_approval"
    assert incident.severity == "high"
    assert invoked == ["triage", "diagnostician", "remediation_planner"]

    approval = (
        await db_session.execute(
            select(ApprovalRequest).where(ApprovalRequest.incident_id == incident.id)
        )
    ).scalar_one()
    assert approval.status == "pending"
    assert approval.risk_level == "medium"

    # Human approves -> executor runs -> resolved.
    await orch.resume_after_approval(incident.id, True, "alice@example.com", db_session)

    assert incident.status == "resolved"
    assert incident.resolved_at is not None
    assert invoked == ["triage", "diagnostician", "remediation_planner", "executor"]
    assert approval.status == "approved"
    assert approval.resolved_by == "alice@example.com"

    # Outcome written to episodic memory exactly once, dry_run enforced.
    assert len(memory.write_calls) == 1
    _, execution_outcome = memory.write_calls[0]
    assert execution_outcome.payload["success"] is True
    assert execution_outcome.payload["dry_run"] is True

    roles = await roles_seen(db_session, incident)
    for role in ("triage", "diagnostician", "remediation_planner", "executor"):
        assert role in roles


# ---------------------------------------------------------------------- #
# Escalation path: high severity + low-confidence diagnosis stops the run
# ---------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_escalation_stops_before_remediation(db_session: AsyncSession) -> None:
    incident = await make_incident(db_session)
    invoked: list[str] = []
    scripts = {
        "triage": [{**TRIAGE_OK, "severity": "critical"}],
        "diagnostician": [DIAGNOSIS_LOW_CONFIDENCE],
        "remediation_planner": [PLAN_OK],  # must never be reached
    }
    orch = Orchestrator(
        memory=FakeMemoryManager(), agent_factory=make_agent_factory(scripts, invoked)
    )

    await orch.run_incident(incident.id, db_session)

    assert incident.status == "escalated"
    # The run stopped at the gate: no planner, no executor, no approval row.
    assert invoked == ["triage", "diagnostician"]
    approvals = (
        await db_session.execute(
            select(ApprovalRequest).where(ApprovalRequest.incident_id == incident.id)
        )
    ).scalars().all()
    assert approvals == []

    # The escalation reason is on the record as an orchestrator message.
    reasons = await orchestrator_reasons(db_session, incident)
    assert any("confidence" in r for r in reasons)


# ---------------------------------------------------------------------- #
# Approval gate: executor untouchable until a human decides; rejection closes
# ---------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_approval_gate_blocks_executor_and_rejection_closes(
    db_session: AsyncSession,
) -> None:
    incident = await make_incident(db_session)
    invoked: list[str] = []
    orch = Orchestrator(
        memory=FakeMemoryManager(),
        agent_factory=make_agent_factory(HAPPY_SCRIPTS, invoked),
    )

    await orch.run_incident(incident.id, db_session)

    # run_incident STOPS here — the Executor was never even instantiated.
    assert incident.status == "awaiting_approval"
    assert "executor" not in invoked

    await orch.resume_after_approval(incident.id, False, "bob@example.com", db_session)

    assert incident.status == "closed"
    assert "executor" not in invoked  # rejected -> still never invoked
    approval = (
        await db_session.execute(
            select(ApprovalRequest).where(ApprovalRequest.incident_id == incident.id)
        )
    ).scalar_one()
    assert approval.status == "rejected"
    assert approval.resolved_by == "bob@example.com"
    reasons = await orchestrator_reasons(db_session, incident)
    assert any("rejected" in r.lower() for r in reasons)


# ---------------------------------------------------------------------- #
# needs_more_data self-loop: stage re-runs with augmented context
# ---------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_triage_retries_once_on_needs_more_data(db_session: AsyncSession) -> None:
    incident = await make_incident(db_session)
    invoked: list[str] = []
    scripts = {
        **HAPPY_SCRIPTS,
        "triage": [TRIAGE_NEEDS_DATA, TRIAGE_OK],  # 1st run starved, 2nd fine
    }
    orch = Orchestrator(
        memory=FakeMemoryManager(), agent_factory=make_agent_factory(scripts, invoked)
    )

    await orch.run_incident(incident.id, db_session)

    assert invoked.count("triage") == 2  # one retry, then satisfied
    assert incident.status == "awaiting_approval"  # pipeline still completed


# ---------------------------------------------------------------------- #
# Incident-level safety cap: too many agent invocations force-escalates
# ---------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_orchestration_rounds_cap_force_escalates(
    db_session: AsyncSession, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(get_settings(), "MAX_ORCHESTRATION_ROUNDS", 2)
    incident = await make_incident(db_session)
    invoked: list[str] = []
    orch = Orchestrator(
        memory=FakeMemoryManager(),
        agent_factory=make_agent_factory(HAPPY_SCRIPTS, invoked),
    )

    await orch.run_incident(incident.id, db_session)

    # Cap of 2: triage + diagnostician ran, the planner tripped the breaker.
    assert incident.status == "escalated"
    assert invoked == ["triage", "diagnostician"]
    reasons = await orchestrator_reasons(db_session, incident)
    assert any("safety cap" in r for r in reasons)
