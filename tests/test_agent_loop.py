"""Guardrail tests for the AgentLoop state machine.

DummyAgentLoop injects scripted PLAN/REFLECT results instead of calling an
LLM, so these tests exercise the real control flow (iteration caps, budget,
termination hooks, tool-failure recovery, trace capture) deterministically.
"""

from typing import Any

import pytest
from pydantic import BaseModel

from app.agents.base import AgentLoop
from app.agents.schemas import (
    AgentStep,
    LoopConfig,
    PlanDecision,
    Reflection,
    ToolCall,
)
from app.agents.tools.base import EchoTool, Tool, ToolRegistry


class ExplodingInput(BaseModel):
    message: str


class ExplodingTool(Tool):
    """A tool that always raises — simulates a buggy/unavailable backend."""

    name = "exploding"
    description = "Always raises RuntimeError."
    input_schema = ExplodingInput

    async def run(self, **kwargs: Any) -> Any:
        raise RuntimeError("backend unavailable")


class DummyAgentLoop(AgentLoop):
    """Runs the real loop with a scripted sequence of plan decisions.

    `script` items are (PlanDecision, token_cost). When the script runs dry,
    the last item repeats — convenient for 'never terminates' tests.
    """

    agent_name = "dummy"

    def __init__(
        self,
        config: LoopConfig,
        registry: ToolRegistry,
        script: list[tuple[PlanDecision, int]],
        terminate_after: int | None = None,
        reflection: Reflection | None = None,
    ) -> None:
        super().__init__(config, registry)
        self.script = script
        self.script_pos = 0
        self.terminate_after = terminate_after
        self.reflection = reflection or Reflection(critique="on track")

    async def build_system_prompt(self, context: dict[str, Any]) -> str:
        return "You are a dummy agent."

    async def should_terminate(
        self, steps: list[AgentStep], context: dict[str, Any]
    ) -> bool:
        return self.terminate_after is not None and len(steps) >= self.terminate_after

    def parse_final_output(
        self, steps: list[AgentStep], context: dict[str, Any]
    ) -> dict[str, Any]:
        return {
            "summary": f"dummy done after {len(steps)} steps",
            "confidence": 0.9,
            "steps_taken": len(steps),
        }

    # Inject scripted results in place of the two LLM calls.
    async def _plan_step(
        self, context: dict[str, Any], prior_steps: list[AgentStep]
    ) -> tuple[PlanDecision, int]:
        item = self.script[min(self.script_pos, len(self.script) - 1)]
        self.script_pos += 1
        return item

    async def _reflect_step(
        self,
        context: dict[str, Any],
        prior_steps: list[AgentStep],
        current_step: AgentStep,
    ) -> tuple[Reflection, int]:
        return self.reflection, 5


def tool_plan(tool: str = "echo", tokens: int = 10, **arguments: Any) -> tuple[PlanDecision, int]:
    args = arguments or {"message": "hi"}
    return (
        PlanDecision(
            thought=f"I should call {tool}",
            tool_call=ToolCall(tool_name=tool, arguments=args),
        ),
        tokens,
    )


def final_plan(answer: str = "done", tokens: int = 10) -> tuple[PlanDecision, int]:
    return PlanDecision(thought=answer, final_answer=answer), tokens


def make_registry() -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(EchoTool())
    registry.register(ExplodingTool())
    return registry


@pytest.mark.asyncio
async def test_max_iterations_respected() -> None:
    """An agent that keeps calling tools forever is cut off at max_iterations."""
    loop = DummyAgentLoop(
        LoopConfig(max_iterations=3, token_budget=10_000, reflect_every_step=False),
        make_registry(),
        script=[tool_plan()],  # repeats forever, never a final answer
    )
    outcome = await loop.run({"incident": "test"})

    assert outcome.status == "max_iterations_reached"
    assert len(outcome.steps) == 3
    assert outcome.payload["steps_taken"] == 3  # partial output still extracted


@pytest.mark.asyncio
async def test_token_budget_respected() -> None:
    """Budget is checked before each plan call; the loop stops once spent."""
    loop = DummyAgentLoop(
        LoopConfig(max_iterations=50, token_budget=1000, reflect_every_step=False),
        make_registry(),
        script=[tool_plan(tokens=600)],
    )
    outcome = await loop.run({})

    # 600 after step 1 (< 1000, continue), 1200 after step 2 (>= 1000, stop).
    assert outcome.status == "budget_exceeded"
    assert len(outcome.steps) == 2
    assert outcome.total_tokens_used == 1200


@pytest.mark.asyncio
async def test_should_terminate_stops_loop_early() -> None:
    """The subclass termination hook ends the loop before caps are hit."""
    loop = DummyAgentLoop(
        LoopConfig(max_iterations=10, token_budget=10_000, reflect_every_step=False),
        make_registry(),
        script=[tool_plan()],
        terminate_after=2,
    )
    outcome = await loop.run({})

    assert outcome.status == "completed"
    assert len(outcome.steps) == 2
    assert outcome.confidence == 0.9


@pytest.mark.asyncio
async def test_final_answer_terminates_loop() -> None:
    """A plan with no tool call is a final answer and ends the loop."""
    loop = DummyAgentLoop(
        LoopConfig(max_iterations=10, token_budget=10_000, reflect_every_step=False),
        make_registry(),
        script=[tool_plan(), final_plan("severity assigned")],
    )
    outcome = await loop.run({})

    assert outcome.status == "completed"
    assert len(outcome.steps) == 2
    assert outcome.steps[1].action is None


@pytest.mark.asyncio
async def test_tool_failure_does_not_crash_loop() -> None:
    """A raising tool becomes a failed ToolResult observation; the loop
    continues and can still terminate cleanly."""
    loop = DummyAgentLoop(
        LoopConfig(max_iterations=10, token_budget=10_000, reflect_every_step=False),
        make_registry(),
        script=[
            tool_plan(tool="exploding", message="boom"),
            final_plan("recovered"),
        ],
    )
    outcome = await loop.run({})

    assert outcome.status == "completed"
    assert len(outcome.steps) == 2
    failed_observation = outcome.steps[0].observation
    assert failed_observation is not None
    assert '"success": false' in failed_observation
    assert "RuntimeError" in failed_observation


@pytest.mark.asyncio
async def test_unknown_tool_becomes_failed_observation() -> None:
    """A hallucinated tool name is an observation, not an exception."""
    loop = DummyAgentLoop(
        LoopConfig(max_iterations=10, token_budget=10_000, reflect_every_step=False),
        make_registry(),
        script=[tool_plan(tool="does_not_exist"), final_plan()],
    )
    outcome = await loop.run({})

    assert outcome.status == "completed"
    observation = outcome.steps[0].observation
    assert observation is not None
    assert '"success": false' in observation


@pytest.mark.asyncio
async def test_trace_captured_in_order() -> None:
    """The full AgentStep trace is captured, numbered, and ordered."""
    loop = DummyAgentLoop(
        LoopConfig(max_iterations=10, token_budget=10_000, reflect_every_step=True),
        make_registry(),
        script=[tool_plan(), tool_plan(), final_plan("all done")],
        reflection=Reflection(critique="still on track", needs_more_data=False),
    )
    outcome = await loop.run({"incident": "trace-test"})

    assert outcome.status == "completed"
    assert [s.step_number for s in outcome.steps] == [1, 2, 3]
    # tool steps have action+observation, final step has neither
    assert outcome.steps[0].action is not None
    assert outcome.steps[0].observation is not None
    assert outcome.steps[2].action is None
    # reflections recorded on every step
    assert all(s.reflection == "still on track" for s in outcome.steps)


@pytest.mark.asyncio
async def test_reflection_needs_more_data_changes_status() -> None:
    """A clean completion that flagged missing data reports needs_more_data
    so the orchestrator can route for enrichment."""
    loop = DummyAgentLoop(
        LoopConfig(max_iterations=10, token_budget=10_000, reflect_every_step=True),
        make_registry(),
        script=[final_plan("best guess")],
        reflection=Reflection(critique="logs were empty", needs_more_data=True),
    )
    outcome = await loop.run({})

    assert outcome.status == "needs_more_data"
    assert outcome.needs_more_data is True


@pytest.mark.asyncio
async def test_unhandled_exception_returns_error_outcome() -> None:
    """A bug anywhere inside the loop yields status='error', never a raise."""

    class BrokenLoop(DummyAgentLoop):
        def parse_final_output(
            self, steps: list[AgentStep], context: dict[str, Any]
        ) -> dict[str, Any]:
            raise ValueError("subclass bug")

    loop = BrokenLoop(
        LoopConfig(max_iterations=2, token_budget=10_000, reflect_every_step=False),
        make_registry(),
        script=[final_plan()],
    )
    outcome = await loop.run({})

    assert outcome.status == "error"
    assert "ValueError" in outcome.summary
    assert outcome.confidence == 0.0
