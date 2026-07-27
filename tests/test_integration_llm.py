"""Integration tests that hit the real Anthropic API.

Excluded by default (see pytest.ini addopts). Run with:

    pytest -m integration -o addopts=""

Requires ANTHROPIC_API_KEY in the environment / .env. Each test costs a few
thousand tokens.
"""

import os

import pytest

from app.agents.schemas import LoopConfig
from app.agents.triage import TriageAgent
from tests._scripting import make_infra_registry

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not os.environ.get("ANTHROPIC_API_KEY"),
        reason="ANTHROPIC_API_KEY not set",
    ),
]


@pytest.mark.asyncio
async def test_triage_real_llm_end_to_end() -> None:
    """One real plan/act/observe/reflect run: the model should query the mock
    infra, see the connection-pool evidence, and emit a valid TriagePayload."""
    agent = TriageAgent(
        LoopConfig(max_iterations=5, token_budget=30_000, reflect_every_step=True),
        make_infra_registry(scenario="connection_pool_exhaustion"),
    )
    outcome = await agent.run(
        {
            "incident": {
                "title": "api-service requests timing out, 500s rising",
                "description": (
                    "Alerts: p99 latency ramping past 4s, error rate climbing. "
                    "Logs mention connection acquisition timeouts."
                ),
                "source": "synthetic",
            }
        }
    )

    assert outcome.status in ("completed", "needs_more_data")
    assert outcome.payload.get("severity") in ("low", "medium", "high", "critical")
    assert 0.0 <= outcome.confidence <= 1.0
    assert outcome.total_tokens_used > 0  # real usage counted from the API
