"""TriageAgent: classify an incoming incident's severity and category."""

import json
from typing import Any

from app.agents._parsing import extract_latest_payload, parse_payload
from app.agents.base import AgentLoop
from app.agents.payloads import TriagePayload
from app.agents.schemas import AgentStep, PlanDecision


class TriageAgent(AgentLoop):
    agent_name = "triage"

    async def build_system_prompt(self, context: dict[str, Any]) -> str:
        incident = context.get("incident", {})
        return f"""You are the TRIAGE agent in SentinelOps, an incident response system.

Your job: classify the incoming incident's SEVERITY and CATEGORY. Nothing else —
do not diagnose root cause (a specialist agent does that next).

Incident under triage:
{json.dumps(incident, default=str, indent=2)}

Severity scale:
- critical: user-facing outage or data loss in progress
- high: major degradation, revenue-impacting, or imminent outage
- medium: partial degradation with a workaround, or non-critical service
- low: cosmetic, informational, or self-recovered

Category: one short snake_case label, e.g. resource_exhaustion, bad_deploy,
database, network, capacity, security.

Use query_logs / query_metrics if the alert alone is ambiguous. Don't burn
tool calls when the alert is already conclusive.

When you are confident, reply WITHOUT any tool call, with a single JSON object:
{{"severity": "...", "category": "...", "confidence": 0.0-1.0,
  "needs_more_data": true|false, "reasoning": "<2-3 sentences>"}}"""

    async def accept_final_answer(
        self, decision: PlanDecision, steps: list[AgentStep], context: dict[str, Any]
    ) -> bool:
        # A final answer only counts if it parses into a valid TriagePayload —
        # otherwise the loop rejects it and the model gets to fix its format.
        return parse_payload(decision.thought, TriagePayload) is not None

    async def should_terminate(
        self, steps: list[AgentStep], context: dict[str, Any]
    ) -> bool:
        # Termination happens via an accepted final answer; there's no
        # mid-loop early exit for triage (classification IS the whole job).
        return False

    def parse_final_output(
        self, steps: list[AgentStep], context: dict[str, Any]
    ) -> dict[str, Any]:
        payload = extract_latest_payload(steps, TriagePayload)
        if payload is None:
            # Guardrail exit (max_iterations/budget) before a valid answer.
            return {
                "summary": "triage did not produce a valid TriagePayload",
                "confidence": 0.0,
                "needs_more_data": True,
            }
        return {
            "summary": f"Triaged as {payload.severity} / {payload.category}",
            "confidence": payload.confidence,
            "needs_more_data": payload.needs_more_data,
            **payload.model_dump(),
        }
