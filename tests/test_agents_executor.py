from typing import Any

import pytest
from pydantic import BaseModel

from app.agents.executor import ExecutorAgent
from app.agents.schemas import LoopConfig
from app.agents.tools.base import Tool, ToolRegistry
from app.agents.tools.mock_infra import RestartServiceTool, ScaleDeploymentTool

CONFIG = LoopConfig(max_iterations=10, token_budget=10_000)


class FailingInput(BaseModel):
    service_name: str
    dry_run: bool = True


class FailingTool(Tool):
    name = "failing_tool"
    description = "Always fails — simulates a broken infra backend."
    input_schema = FailingInput

    async def run(self, **kwargs: Any) -> Any:
        raise RuntimeError("infra backend unreachable")


def make_registry() -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(RestartServiceTool())
    registry.register(ScaleDeploymentTool())
    registry.register(FailingTool())
    return registry


RESTART_ACTION = {
    "action_type": "restart",
    "target": "payment-service",
    "tool_name": "restart_service",
    "parameters": {"service_name": "payment-service"},
    "rationale": "clear leaked buffer",
}
SCALE_ACTION = {
    "action_type": "scale",
    "target": "payment-service",
    "tool_name": "scale_deployment",
    "parameters": {"deployment": "payment-service", "replicas": 6},
    "rationale": "absorb load",
}
FAILING_ACTION = {
    "action_type": "restart",
    "target": "payment-service",
    "tool_name": "failing_tool",
    "parameters": {"service_name": "payment-service"},
    "rationale": "this will fail",
}


@pytest.mark.asyncio
async def test_refuses_without_approved_approval_request() -> None:
    agent = ExecutorAgent(CONFIG, make_registry())

    for bad_status in ("pending", "rejected", "auto_denied", None):
        outcome = await agent.run(
            {"approval_status": bad_status, "proposed_actions": [RESTART_ACTION]}
        )
        assert outcome.status == "error"
        assert "refused" in outcome.summary.lower()
        assert outcome.payload["success"] is False
        assert outcome.payload["actions_taken"] == []  # nothing leaked


@pytest.mark.asyncio
async def test_executes_all_approved_actions_dry_run() -> None:
    agent = ExecutorAgent(CONFIG, make_registry())
    outcome = await agent.run(
        {
            "approval_status": "approved",
            "proposed_actions": [RESTART_ACTION, SCALE_ACTION],
        }
    )

    assert outcome.status == "completed"
    assert outcome.payload["success"] is True
    assert outcome.payload["dry_run"] is True  # project-wide default
    assert len(outcome.payload["actions_taken"]) == 2
    assert all(a["success"] for a in outcome.payload["actions_taken"])
    # dry_run results describe what WOULD happen, not what happened
    assert "would_do" in outcome.payload["actions_taken"][0]["result"]
    assert outcome.total_tokens_used == 0  # deterministic executor: no LLM


@pytest.mark.asyncio
async def test_aborts_remaining_actions_on_first_failure() -> None:
    agent = ExecutorAgent(CONFIG, make_registry())
    outcome = await agent.run(
        {
            "approval_status": "approved",
            "proposed_actions": [FAILING_ACTION, RESTART_ACTION, SCALE_ACTION],
        }
    )

    assert outcome.status == "completed"  # clean abort, not a crash
    assert outcome.payload["success"] is False
    # Only the failed action ran; the remaining two were aborted.
    assert len(outcome.payload["actions_taken"]) == 1
    assert outcome.payload["actions_taken"][0]["tool_name"] == "failing_tool"
    assert outcome.payload["errors"]
    assert "Aborting" in outcome.steps[-1].thought


@pytest.mark.asyncio
async def test_dry_run_forced_even_if_plan_smuggles_dry_run_false() -> None:
    """An approved plan whose parameters claim dry_run=false must still run
    as a drill: the executor overwrites dry_run from config."""
    smuggled = {**RESTART_ACTION, "parameters": {"service_name": "payment-service", "dry_run": False}}
    agent = ExecutorAgent(CONFIG, make_registry())
    outcome = await agent.run(
        {"approval_status": "approved", "proposed_actions": [smuggled]}
    )

    action = outcome.payload["actions_taken"][0]
    assert action["arguments"]["dry_run"] is True
    assert action["result"]["dry_run"] is True
