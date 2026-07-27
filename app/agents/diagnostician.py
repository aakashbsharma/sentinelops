"""DiagnosticianAgent: form a root-cause hypothesis from evidence + memory.

Expects `context` to contain (assembled by the Stage 5 orchestrator):
  - "incident": incident summary dict
  - "triage": the TriagePayload dict from the triage agent
  - "episodic_memories" / "semantic_memories": MemoryManager output
"""

import json
from typing import Any

from app.agents._parsing import extract_latest_payload, parse_payload
from app.agents.base import AgentLoop
from app.agents.payloads import DiagnosisPayload
from app.agents.schemas import AgentStep, PlanDecision


def _render_memories(memories: list[dict[str, Any]], empty_note: str) -> str:
    if not memories:
        return empty_note
    return "\n".join(
        f"- (similarity {m.get('similarity', '?')}) {m.get('content', '')}"
        for m in memories
    )


class DiagnosticianAgent(AgentLoop):
    agent_name = "diagnostician"

    async def build_system_prompt(self, context: dict[str, Any]) -> str:
        incident = context.get("incident", {})
        triage = context.get("triage", {})
        episodic = _render_memories(
            context.get("episodic_memories", []), "(no similar past incidents found)"
        )
        semantic = _render_memories(
            context.get("semantic_memories", []), "(no relevant runbook excerpts found)"
        )
        return f"""You are the DIAGNOSTICIAN agent in SentinelOps, an incident response system.

Your job: determine the ROOT CAUSE of this incident, backed by evidence.

Incident:
{json.dumps(incident, default=str, indent=2)}

Triage assessment:
{json.dumps(triage, default=str, indent=2)}

SIMILAR PAST INCIDENTS (episodic memory — how comparable incidents were resolved):
{episodic}

RELEVANT RUNBOOK EXCERPTS (semantic memory):
{semantic}

Method:
1. Start from the triage category and the past-incident/runbook context above.
2. Use query_logs and query_metrics to gather CONCRETE evidence. Every claim
   in your final answer must be traceable to an observation or a memory.
3. If the evidence is thin or contradictory, DO NOT GUESS: set
   needs_more_data=true and say what data is missing. A wrong root cause sent
   to the remediation planner is far more expensive than asking for more data.

When confident (or when you've concluded data is insufficient), reply WITHOUT
any tool call, with a single JSON object:
{{"root_cause": "...", "evidence": ["..."], "contributing_factors": ["..."],
  "confidence": 0.0-1.0, "needs_more_data": true|false}}"""

    async def accept_final_answer(
        self, decision: PlanDecision, steps: list[AgentStep], context: dict[str, Any]
    ) -> bool:
        return parse_payload(decision.thought, DiagnosisPayload) is not None

    async def should_terminate(
        self, steps: list[AgentStep], context: dict[str, Any]
    ) -> bool:
        # Never early-terminate while the latest reflection says evidence is
        # thin — a diagnostician that stops early on thin evidence ships
        # guesses. Termination happens via an accepted final answer, which
        # can itself carry needs_more_data=true (an honest "I don't know
        # yet" is a valid, useful result for the orchestrator to route on).
        return False

    def parse_final_output(
        self, steps: list[AgentStep], context: dict[str, Any]
    ) -> dict[str, Any]:
        payload = extract_latest_payload(steps, DiagnosisPayload)
        if payload is None:
            return {
                "summary": "diagnosis did not produce a valid DiagnosisPayload",
                "confidence": 0.0,
                "needs_more_data": True,
            }
        return {
            "summary": f"Root cause hypothesis: {payload.root_cause}",
            "confidence": payload.confidence,
            "needs_more_data": payload.needs_more_data,
            **payload.model_dump(),
        }
