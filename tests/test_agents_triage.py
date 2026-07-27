import json

import pytest

from app.agents.schemas import LoopConfig
from app.agents.triage import TriageAgent
from tests._scripting import make_infra_registry, plan_final, plan_tool, scripted

CONFIG = LoopConfig(max_iterations=6, token_budget=10_000, reflect_every_step=False)

INCIDENT_CONTEXT = {
    "incident": {
        "title": "payment-service p99 latency above 2s",
        "description": "checkout requests timing out",
        "source": "synthetic",
    }
}

VALID_TRIAGE_JSON = json.dumps(
    {
        "severity": "high",
        "category": "resource_exhaustion",
        "confidence": 0.87,
        "needs_more_data": False,
        "reasoning": "Metrics show latency ramping with connection errors in logs.",
    }
)


@pytest.mark.asyncio
async def test_triage_extracts_payload_and_terminates() -> None:
    agent = scripted(TriageAgent)(
        CONFIG,
        make_infra_registry("connection_pool_exhaustion"),
        script=[
            plan_tool("query_metrics", service_name="payment-service"),
            plan_final(f"Based on the metrics:\n```json\n{VALID_TRIAGE_JSON}\n```"),
        ],
    )
    outcome = await agent.run(dict(INCIDENT_CONTEXT))

    assert outcome.status == "completed"
    assert len(outcome.steps) == 2
    assert outcome.payload["severity"] == "high"
    assert outcome.payload["category"] == "resource_exhaustion"
    assert outcome.confidence == 0.87
    assert outcome.summary == "Triaged as high / resource_exhaustion"


@pytest.mark.asyncio
async def test_triage_rejects_invalid_final_answer_and_retries() -> None:
    """An unparseable final answer must NOT terminate the loop — the agent
    gets the rejection as an observation and can fix its format."""
    agent = scripted(TriageAgent)(
        CONFIG,
        make_infra_registry(),
        script=[
            plan_final("The severity is high and it's probably the database."),  # no JSON
            plan_final(VALID_TRIAGE_JSON),
        ],
    )
    outcome = await agent.run(dict(INCIDENT_CONTEXT))

    assert outcome.status == "completed"
    assert len(outcome.steps) == 2  # rejected step + accepted step
    assert "rejected" in (outcome.steps[0].observation or "")
    assert outcome.payload["severity"] == "high"


@pytest.mark.asyncio
async def test_triage_guardrail_exit_reports_failure() -> None:
    """If the loop caps out before a valid payload, confidence is 0 and
    needs_more_data is flagged — never a fabricated classification."""
    agent = scripted(TriageAgent)(
        LoopConfig(max_iterations=2, token_budget=10_000, reflect_every_step=False),
        make_infra_registry(),
        script=[plan_tool("query_logs", service_name="payment-service")],
    )
    outcome = await agent.run(dict(INCIDENT_CONTEXT))

    assert outcome.status == "max_iterations_reached"
    assert outcome.confidence == 0.0
    assert outcome.needs_more_data is True
