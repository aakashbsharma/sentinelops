"""Orchestrator decision schema.

Every routing decision the orchestrator makes at a stage transition is
expressed as an OrchestratorDecision and persisted as an AgentMessage with
role="orchestrator" — so the incident trace shows not just what each agent
said, but WHY the orchestrator routed the way it did.

The decisions themselves are currently produced deterministically from the
pure functions in policy.py (informed by each agent's confidence and
needs_more_data signals). This schema is the seam where a small, scoped LLM
call could produce the decision instead — scoped meaning: it may only choose
among the actions the state machine permits from the current status, and it
can never override the two non-negotiable guardrails (should_escalate and
requires_approval), which stay in code.
"""

from typing import Literal

from pydantic import BaseModel, Field

NextAction = Literal[
    "run_triage",
    "run_diagnostician",
    "run_remediation_planner",
    "await_approval",
    "run_executor",
    "escalate",
    "resolve",
]


class OrchestratorDecision(BaseModel):
    next_action: NextAction
    reasoning: str
    confidence: float = Field(ge=0.0, le=1.0)
