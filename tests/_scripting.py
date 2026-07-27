"""Shared helper: run real agent classes with scripted plan/reflect steps
instead of LLM calls (same injection pattern as Stage 2's DummyAgentLoop)."""

from typing import Any

from app.agents.base import AgentLoop
from app.agents.schemas import AgentStep, PlanDecision, Reflection, ToolCall
from app.agents.tools.base import ToolRegistry
from app.agents.tools.mock_infra import (
    QueryLogsTool,
    QueryMetricsTool,
    RestartServiceTool,
    ScaleDeploymentTool,
)


def make_infra_registry(scenario: str | None = None) -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(QueryLogsTool(scenario=scenario))
    registry.register(QueryMetricsTool(scenario=scenario))
    registry.register(RestartServiceTool())
    registry.register(ScaleDeploymentTool())
    return registry


def scripted(agent_cls: type[AgentLoop]) -> type[AgentLoop]:
    """Subclass any agent so _plan_step/_reflect_step pop scripted results.

    The rest of the agent — accept_final_answer, should_terminate,
    parse_final_output, guardrails — runs for real.
    """

    class ScriptedAgent(agent_cls):  # type: ignore[valid-type, misc]
        def __init__(
            self,
            *args: Any,
            script: list[tuple[PlanDecision, int]],
            reflections: list[Reflection] | None = None,
            **kwargs: Any,
        ) -> None:
            super().__init__(*args, **kwargs)
            self._script = script
            self._script_pos = 0
            self._reflections = reflections or []
            self._reflection_pos = 0

        async def _plan_step(
            self, context: dict[str, Any], prior_steps: list[AgentStep]
        ) -> tuple[PlanDecision, int]:
            item = self._script[min(self._script_pos, len(self._script) - 1)]
            self._script_pos += 1
            return item

        async def _reflect_step(
            self,
            context: dict[str, Any],
            prior_steps: list[AgentStep],
            current_step: AgentStep,
        ) -> tuple[Reflection, int]:
            if not self._reflections:
                return Reflection(critique="on track"), 5
            item = self._reflections[
                min(self._reflection_pos, len(self._reflections) - 1)
            ]
            self._reflection_pos += 1
            return item, 5

    ScriptedAgent.__name__ = f"Scripted{agent_cls.__name__}"
    return ScriptedAgent


def plan_tool(tool_name: str, thought: str = "", **arguments: Any) -> tuple[PlanDecision, int]:
    return (
        PlanDecision(
            thought=thought or f"Calling {tool_name}",
            tool_call=ToolCall(tool_name=tool_name, arguments=arguments),
        ),
        50,
    )


def plan_final(text: str) -> tuple[PlanDecision, int]:
    return PlanDecision(thought=text, final_answer=text), 50
