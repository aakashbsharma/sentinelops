"""API request/response schemas for the incident endpoints."""

from datetime import datetime
from typing import Any
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from app.observability.cost import CostSummary


class IncidentCreate(BaseModel):
    title: str = Field(min_length=1, max_length=500)
    description: str | None = None
    source: str = "synthetic"
    # Provisional — the Triage agent's verdict overwrites this.
    severity: str = "unknown"
    raw_payload: dict[str, Any] | None = None


class IncidentAccepted(BaseModel):
    id: UUID
    status: str


class AgentMessageOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    agent_name: str
    role: str
    run_id: str | None = None
    content: dict[str, Any]
    confidence: float | None
    needs_more_data: bool
    created_at: datetime


class IncidentOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    title: str
    description: str | None
    source: str
    severity: str
    status: str
    raw_payload: dict[str, Any] | None
    created_at: datetime
    updated_at: datetime
    resolved_at: datetime | None


class IncidentDetailOut(IncidentOut):
    messages: list[AgentMessageOut]
    # Computed by the route from the messages, not a DB attribute — see
    # app/observability/cost.py for why cost is derived on read.
    cost_summary: CostSummary = Field(default_factory=CostSummary)


class IncidentListOut(BaseModel):
    items: list[IncidentOut]
    total: int
    limit: int
    offset: int


class ApprovalRequestOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    incident_id: UUID
    proposed_action: dict[str, Any]
    risk_level: str
    status: str
    requested_at: datetime
    resolved_at: datetime | None
    resolved_by: str | None


class ApprovalDecisionIn(BaseModel):
    approved: bool
    resolved_by: str = Field(min_length=1, max_length=200)
