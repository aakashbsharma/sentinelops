"""Stage 7 observability: trace correlation, cost aggregation, trace-bus shape."""

import asyncio
import json
import uuid

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.agents.schemas import AgentStep
from app.core import tracing
from app.models.agent_message import AgentMessage
from app.observability.cost import compute_cost_summary
from app.observability.trace_bus import step_payload
from app.orchestrator.core import Orchestrator
from tests.test_orchestrator import (
    HAPPY_SCRIPTS,
    FakeMemoryManager,
    make_agent_factory,
    make_incident,
)

# ---------------------------------------------------------------------- #
# Trace correlation (contextvars)
# ---------------------------------------------------------------------- #


def test_no_ids_outside_a_run() -> None:
    assert tracing.current_incident_id() is None
    assert tracing.current_run_id() is None


def test_incident_run_sets_and_restores_ids() -> None:
    incident_id = uuid.uuid4()
    with tracing.incident_run(incident_id) as run_id:
        assert tracing.current_incident_id() == str(incident_id)
        assert tracing.current_run_id() == run_id
    # Token-based reset: nothing leaks past the context manager.
    assert tracing.current_incident_id() is None
    assert tracing.current_run_id() is None


def test_two_runs_get_distinct_run_ids() -> None:
    incident_id = uuid.uuid4()
    with tracing.incident_run(incident_id) as first:
        pass
    with tracing.incident_run(incident_id) as second:
        pass
    assert first != second  # e.g. escalate-then-resume = two runs


@pytest.mark.asyncio
async def test_ids_propagate_into_nested_async_calls() -> None:
    """THE design property: nested awaits and spawned tasks inherit the ids
    without any parameter threading (asyncio copies the ambient Context)."""

    async def innermost() -> tuple[str | None, str | None]:
        return tracing.current_incident_id(), tracing.current_run_id()

    async def middle() -> tuple[str | None, str | None]:
        # a spawned task, not just an await — tasks copy the context too
        return await asyncio.create_task(innermost())

    incident_id = uuid.uuid4()
    with tracing.incident_run(incident_id) as run_id:
        seen_incident, seen_run = await middle()
    assert seen_incident == str(incident_id)
    assert seen_run == run_id


def test_agent_steps_self_stamp_correlation_ids() -> None:
    outside = AgentStep(step_number=1, thought="outside any run")
    assert outside.incident_id is None and outside.run_id is None

    incident_id = uuid.uuid4()
    with tracing.incident_run(incident_id) as run_id:
        inside = AgentStep(step_number=1, thought="inside a run")
    assert inside.incident_id == str(incident_id)
    assert inside.run_id == run_id


@pytest.mark.asyncio
async def test_orchestrator_run_stamps_one_run_id_everywhere(
    db_session: AsyncSession,
) -> None:
    """Every AgentMessage from one run_incident call (agents AND orchestrator
    decisions) carries the same run_id, and the steps inside carry it too."""
    incident = await make_incident(db_session)
    orch = Orchestrator(
        memory=FakeMemoryManager(),
        agent_factory=make_agent_factory(HAPPY_SCRIPTS, []),
    )

    await orch.run_incident(incident.id, db_session)

    messages = (
        (
            await db_session.execute(
                select(AgentMessage).where(AgentMessage.incident_id == incident.id)
            )
        )
        .scalars()
        .all()
    )
    assert messages
    run_ids = {m.run_id for m in messages}
    assert len(run_ids) == 1 and None not in run_ids

    (run_id,) = run_ids
    step_run_ids = {
        step["run_id"]
        for m in messages
        for step in m.content.get("steps", [])
    }
    assert step_run_ids <= {run_id}

    # A resume is a NEW run: distinct run_id on its messages.
    await orch.resume_after_approval(incident.id, True, "eve@example.com", db_session)
    resumed = (
        (
            await db_session.execute(
                select(AgentMessage).where(
                    AgentMessage.incident_id == incident.id,
                    AgentMessage.run_id != run_id,
                )
            )
        )
        .scalars()
        .all()
    )
    assert resumed  # executor + resolve decision came from a second run


# ---------------------------------------------------------------------- #
# Cost aggregation
# ---------------------------------------------------------------------- #


def _message(agent: str, tokens: int, model: str | None = "claude-sonnet-4-6"):
    content = {"total_tokens_used": tokens}
    if model is not None:
        content["model"] = model
    return AgentMessage(
        incident_id=uuid.uuid4(), agent_name=agent, role=agent, content=content
    )


def test_cost_summary_aggregates_per_agent(monkeypatch: pytest.MonkeyPatch) -> None:
    from app.core.config import get_settings

    monkeypatch.setattr(
        get_settings(),
        "MODEL_COST_PER_TOKEN",
        {"claude-sonnet-4-6": 1e-5, "default": 2e-5},
    )
    messages = [
        _message("triage", 1000),
        _message("diagnostician", 3000),
        _message("diagnostician", 2000),
        _message("legacy-agent", 500, model=None),  # pre-Stage-7 row -> default rate
        # Orchestrator decisions carry no token usage and must not count.
        AgentMessage(
            incident_id=uuid.uuid4(),
            agent_name="orchestrator",
            role="orchestrator",
            content={"next_action": "resolve"},
        ),
    ]

    summary = compute_cost_summary(messages)

    assert summary.total_tokens == 6500
    assert summary.estimated_cost_usd == pytest.approx(0.06 + 0.01, abs=1e-9)
    assert summary.per_agent_breakdown["diagnostician"].total_tokens == 5000
    assert summary.per_agent_breakdown["diagnostician"].invocations == 2
    assert summary.per_agent_breakdown["legacy-agent"].estimated_cost_usd == (
        pytest.approx(0.01)
    )
    assert "orchestrator" not in summary.per_agent_breakdown


def test_cost_summary_empty() -> None:
    summary = compute_cost_summary([])
    assert summary.total_tokens == 0
    assert summary.estimated_cost_usd == 0.0
    assert summary.per_agent_breakdown == {}


# ---------------------------------------------------------------------- #
# Trace bus wire shape (what Stage 9's panel consumes)
# ---------------------------------------------------------------------- #

EXPECTED_FRAME_KEYS = {
    "step_number",
    "agent_name",
    "thought",
    "action",
    "observation",
    "reflection",
    "timestamp",
    "run_id",
}


def test_step_payload_shape() -> None:
    with tracing.incident_run(uuid.uuid4()) as run_id:
        step = AgentStep(step_number=3, thought="checking metrics")
    payload = step_payload("diagnostician", step)

    assert set(payload) == EXPECTED_FRAME_KEYS
    assert payload["step_number"] == 3
    assert payload["agent_name"] == "diagnostician"
    assert payload["action"] is None
    assert payload["run_id"] == run_id
    json.dumps(payload)  # must be wire-serializable as-is
