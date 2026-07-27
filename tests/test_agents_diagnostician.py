import json

import pytest

from app.agents.diagnostician import DiagnosticianAgent
from app.agents.schemas import LoopConfig, Reflection
from tests._scripting import make_infra_registry, plan_final, plan_tool, scripted

CONFIG = LoopConfig(max_iterations=8, token_budget=20_000, reflect_every_step=True)

CONTEXT = {
    "incident": {"title": "API 500s spiking", "description": "checkout failing"},
    "triage": {"severity": "high", "category": "database", "confidence": 0.8},
    "episodic_memories": [
        {
            "content": "Incident: pool exhaustion. Remediation: pg_terminate_backend. Outcome: successful.",
            "similarity": 0.71,
            "metadata": {"outcome": "success"},
        }
    ],
    "semantic_memories": [
        {
            "content": "If most connections are idle in transaction, it's a leak or stuck transaction.",
            "similarity": 0.66,
            "metadata": {"source_file": "database-connection-pool-exhaustion.md"},
        }
    ],
}

VALID_DIAGNOSIS = json.dumps(
    {
        "root_cause": "connection pool exhausted by stuck idle-in-transaction sessions",
        "evidence": ["logs show TimeoutError acquiring connections", "latency_p99 ramping"],
        "contributing_factors": ["pool_size too small for replica count"],
        "confidence": 0.82,
        "needs_more_data": False,
    }
)

THIN_EVIDENCE_DIAGNOSIS = json.dumps(
    {
        "root_cause": "possibly a connection leak, but evidence is inconclusive",
        "evidence": ["one timeout line in logs"],
        "contributing_factors": [],
        "confidence": 0.4,
        "needs_more_data": True,
    }
)


@pytest.mark.asyncio
async def test_diagnostician_happy_path() -> None:
    agent = scripted(DiagnosticianAgent)(
        CONFIG,
        make_infra_registry("connection_pool_exhaustion"),
        script=[
            plan_tool("query_logs", service_name="api"),
            plan_tool("query_metrics", service_name="api"),
            plan_final(VALID_DIAGNOSIS),
        ],
        reflections=[Reflection(critique="evidence accumulating", needs_more_data=False)],
    )
    outcome = await agent.run(dict(CONTEXT))

    assert outcome.status == "completed"
    assert outcome.payload["root_cause"].startswith("connection pool exhausted")
    assert outcome.confidence == 0.82
    assert len(outcome.steps) == 3


@pytest.mark.asyncio
async def test_diagnostician_keeps_gathering_when_evidence_thin() -> None:
    """Fake reflections keep saying 'not enough data' — the agent must NOT
    terminate early; it keeps tool-calling until its final answer, and the
    outcome carries needs_more_data."""
    agent = scripted(DiagnosticianAgent)(
        CONFIG,
        make_infra_registry(),
        script=[
            plan_tool("query_logs", service_name="api", time_range="15m"),
            plan_tool("query_logs", service_name="api", time_range="1h"),
            plan_tool("query_metrics", service_name="api"),
            plan_final(THIN_EVIDENCE_DIAGNOSIS),
        ],
        reflections=[
            Reflection(critique="not enough data to conclude", needs_more_data=True)
        ],
    )
    outcome = await agent.run(dict(CONTEXT))

    # All four scripted steps ran — no premature termination at step 1 or 2.
    assert len(outcome.steps) == 4
    # An honest "I don't know yet" surfaces as needs_more_data, not completed.
    assert outcome.status == "needs_more_data"
    assert outcome.needs_more_data is True
    assert outcome.confidence == 0.4


@pytest.mark.asyncio
async def test_diagnostician_prompt_injects_memories_and_no_guess_rule() -> None:
    agent = scripted(DiagnosticianAgent)(
        CONFIG, make_infra_registry(), script=[plan_final(VALID_DIAGNOSIS)]
    )
    prompt = await agent.build_system_prompt(dict(CONTEXT))

    assert "pg_terminate_backend" in prompt  # episodic memory injected
    assert "idle in transaction" in prompt  # semantic memory injected
    assert "DO NOT GUESS" in prompt  # thin-evidence instruction present
    assert "needs_more_data=true" in prompt
