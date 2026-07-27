"""Agent-specific payload schemas — the typed contracts between agents.

Each agent's final answer must parse into its payload model; the orchestrator
(Stage 5) routes on these, and they're persisted as AgentMessage.content.
"""

from typing import Any

from pydantic import BaseModel, Field


class TriagePayload(BaseModel):
    severity: str  # low / medium / high / critical
    category: str  # e.g. resource_exhaustion / bad_deploy / network / database
    confidence: float = Field(ge=0.0, le=1.0)
    needs_more_data: bool = False
    reasoning: str


class DiagnosisPayload(BaseModel):
    root_cause: str
    evidence: list[str] = Field(default_factory=list)
    contributing_factors: list[str] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)
    needs_more_data: bool = False


class ProposedAction(BaseModel):
    action_type: str  # e.g. restart / scale / config_change / rollback
    target: str  # service or deployment name
    tool_name: str  # which tool the Executor would call
    parameters: dict[str, Any] = Field(default_factory=dict)
    rationale: str


class RemediationPlanPayload(BaseModel):
    proposed_actions: list[ProposedAction] = Field(default_factory=list)
    risk_level: str  # low / medium / high / critical
    requires_approval: bool = True  # default-safe; guardrail enforces the real value
    rollback_plan: str
    confidence: float = Field(ge=0.0, le=1.0)


class ExecutionPayload(BaseModel):
    actions_taken: list[dict[str, Any]] = Field(default_factory=list)
    success: bool
    dry_run: bool
    errors: list[str] = Field(default_factory=list)
