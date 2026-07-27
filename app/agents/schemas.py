"""Typed contracts shared by every agent loop.

These schemas ARE the inter-agent protocol: the orchestrator (Stage 5) routes
on `AgentOutcome.confidence` / `needs_more_data`, and the observability panel
(Stage 7) renders `AgentOutcome.steps` directly. Keep them stable.
"""

from datetime import datetime, timezone
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, Field

from app.core.tracing import current_incident_id, current_run_id


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class ToolCall(BaseModel):
    """An agent's request to invoke a tool."""

    tool_name: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    call_id: str = Field(default_factory=lambda: str(uuid4()))


class ToolResult(BaseModel):
    """What actually happened when a ToolCall was executed.

    `success=False` + `error` is a normal, expected outcome — tool failures
    are data for the agent to reason about, not exceptions to crash on.
    """

    call_id: str
    success: bool
    output: Any = None
    error: str | None = None
    latency_ms: float


class AgentStep(BaseModel):
    """One full iteration of the plan/act/observe/reflect loop."""

    step_number: int
    thought: str  # the PLAN: what the agent decided to do and why
    action: ToolCall | None = None  # the ACT (None = agent answered directly)
    observation: str | None = None  # the OBSERVE: serialized ToolResult
    reflection: str | None = None  # the REFLECT: self-critique of progress
    timestamp: datetime = Field(default_factory=_utcnow)
    # Correlation ids (Stage 7): steps self-stamp from the ambient contextvars
    # set by tracing.incident_run() — the loop code never threads them through.
    # None when constructed outside a run (e.g. unit tests).
    incident_id: str | None = Field(default_factory=current_incident_id)
    run_id: str | None = Field(default_factory=current_run_id)


class AgentOutcome(BaseModel):
    """The final, typed result of a loop run — every exit path produces one."""

    status: Literal[
        "completed",
        "needs_more_data",
        "max_iterations_reached",
        "budget_exceeded",
        "error",
    ]
    summary: str
    confidence: float = Field(ge=0.0, le=1.0)
    needs_more_data: bool = False
    payload: dict[str, Any] = Field(default_factory=dict)
    steps: list[AgentStep] = Field(default_factory=list)
    total_tokens_used: int = 0


class LoopConfig(BaseModel):
    """Per-agent loop guardrails. Defaults come from Settings at wiring time."""

    max_iterations: int = 8
    token_budget: int = 50_000
    # Placeholder model id — the real Anthropic call is wired in a later task.
    model: str = "claude-sonnet-4-6"
    reflect_every_step: bool = True
    tool_timeout_s: float = 30.0


class PlanDecision(BaseModel):
    """Parsed output of one PLAN call: the LLM either picks a tool or answers.

    Exactly one of `tool_call` / `final_answer` is normally set; if the LLM
    returns only prose with no tool call, we treat it as a final answer.
    """

    thought: str
    tool_call: ToolCall | None = None
    final_answer: str | None = None

    @property
    def is_final(self) -> bool:
        return self.tool_call is None


class Reflection(BaseModel):
    """Parsed output of one REFLECT call."""

    on_track: bool = True
    critique: str
    needs_more_data: bool = False
