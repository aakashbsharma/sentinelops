import json

import pytest

from app.agents.remediation_planner import RemediationPlannerAgent
from app.agents.schemas import LoopConfig
from tests._scripting import make_infra_registry, plan_final, scripted

CONFIG = LoopConfig(max_iterations=4, token_budget=10_000, reflect_every_step=False)

CONTEXT = {
    "incident": {"title": "payment-service OOMKilled"},
    "diagnosis": {
        "root_cause": "memory leak in retry buffer",
        "confidence": 0.85,
    },
    "semantic_memories": [],
}


def plan_json(
    tool_name: str,
    risk_level: str = "low",
    confidence: float = 0.99,
    requires_approval: bool = False,
) -> str:
    """A plan where the model itself claims requires_approval=False."""
    return json.dumps(
        {
            "proposed_actions": [
                {
                    "action_type": "restart",
                    "target": "payment-service",
                    "tool_name": tool_name,
                    "parameters": {"service_name": "payment-service"},
                    "rationale": "clear the leaked retry buffer",
                }
            ],
            "risk_level": risk_level,
            "requires_approval": requires_approval,
            "rollback_plan": "previous replicaset is retained; roll back via deployment undo",
            "confidence": confidence,
        }
    )


async def run_planner(final_json: str):
    agent = scripted(RemediationPlannerAgent)(
        CONFIG, make_infra_registry(), script=[plan_final(final_json)]
    )
    return await agent.run(dict(CONTEXT))


@pytest.mark.asyncio
async def test_mutating_action_requires_approval_even_at_confidence_099() -> None:
    """THE core safety test: the model proposed restart_service, claimed
    risk=low, confidence=0.99, and even set requires_approval=false itself.
    The guardrail must overwrite that to True — an LLM never auto-approves
    its own destructive action."""
    outcome = await run_planner(
        plan_json("restart_service", risk_level="low", confidence=0.99, requires_approval=False)
    )

    assert outcome.status == "completed"
    assert outcome.payload["requires_approval"] is True


@pytest.mark.asyncio
async def test_readonly_low_risk_high_confidence_may_skip_approval() -> None:
    outcome = await run_planner(
        plan_json("query_logs", risk_level="low", confidence=0.9)
    )
    assert outcome.payload["requires_approval"] is False


@pytest.mark.asyncio
async def test_readonly_but_low_confidence_requires_approval() -> None:
    outcome = await run_planner(
        plan_json("query_logs", risk_level="low", confidence=0.7)
    )
    assert outcome.payload["requires_approval"] is True


@pytest.mark.asyncio
async def test_readonly_but_high_risk_requires_approval() -> None:
    outcome = await run_planner(
        plan_json("query_logs", risk_level="high", confidence=0.95)
    )
    assert outcome.payload["requires_approval"] is True


@pytest.mark.asyncio
async def test_mixed_plan_one_mutating_action_poisons_auto_approval() -> None:
    """One mutating action among read-only ones forces approval for the
    whole plan."""
    plan = json.loads(plan_json("query_logs"))
    plan["proposed_actions"].append(
        {
            "action_type": "scale",
            "target": "payment-service",
            "tool_name": "scale_deployment",
            "parameters": {"deployment": "payment-service", "replicas": 6},
            "rationale": "absorb load during restart",
        }
    )
    outcome = await run_planner(json.dumps(plan))

    assert outcome.payload["requires_approval"] is True
