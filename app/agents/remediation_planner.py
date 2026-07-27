"""RemediationPlannerAgent: turn a diagnosis into concrete proposed actions.

It PROPOSES tool calls for the Executor — it never calls mutating tools
itself. The approval guardrail here is the platform's core safety property.
"""

import json
from typing import Any

from app.agents._parsing import extract_latest_payload, parse_payload
from app.agents.base import AgentLoop
from app.agents.payloads import RemediationPlanPayload
from app.agents.schemas import AgentStep, PlanDecision

# Tools that CANNOT mutate infrastructure. Only plans composed exclusively of
# these may ever skip human approval. Deliberately an explicit allowlist, not
# a "denylist of dangerous tools": a new tool added tomorrow is
# approval-required by default until someone consciously adds it here.
READ_ONLY_TOOL_ALLOWLIST = frozenset({"query_logs", "query_metrics", "search_runbooks"})

AUTO_APPROVE_MIN_CONFIDENCE = 0.85


class RemediationPlannerAgent(AgentLoop):
    agent_name = "remediation_planner"

    async def build_system_prompt(self, context: dict[str, Any]) -> str:
        incident = context.get("incident", {})
        diagnosis = context.get("diagnosis", {})
        semantic = context.get("semantic_memories", [])
        runbooks = (
            "\n".join(f"- {m.get('content', '')}" for m in semantic)
            or "(no relevant runbook excerpts found)"
        )
        # The infra tools' schemas are provided as REFERENCE for proposing
        # actions — this agent must not execute them.
        tool_specs = json.dumps(self.tools.list_tools(), indent=2)
        return f"""You are the REMEDIATION PLANNER agent in SentinelOps, an incident response system.

Your job: given a diagnosis, propose a concrete remediation plan. You DO NOT
execute anything — a separate Executor agent runs approved plans.

Incident:
{json.dumps(incident, default=str, indent=2)}

Diagnosis:
{json.dumps(diagnosis, default=str, indent=2)}

Relevant runbook excerpts:
{runbooks}

Available executor tools (for reference when proposing actions — you may NOT call them):
{tool_specs}

Rules:
- Prefer the least invasive action that addresses the root cause. Runbooks
  often warn which actions make things worse — heed them.
- Every proposed action needs a rationale tied to the diagnosis.
- Always include a rollback plan: what to do if the remediation fails or worsens things.
- risk_level reflects blast radius: restarting one service is usually medium,
  scaling is usually low-to-medium, anything touching data is high+.

Reply WITHOUT any tool call, with a single JSON object:
{{"proposed_actions": [{{"action_type": "...", "target": "...", "tool_name": "...",
   "parameters": {{...}}, "rationale": "..."}}],
  "risk_level": "low|medium|high|critical", "requires_approval": true,
  "rollback_plan": "...", "confidence": 0.0-1.0}}"""

    async def accept_final_answer(
        self, decision: PlanDecision, steps: list[AgentStep], context: dict[str, Any]
    ) -> bool:
        return parse_payload(decision.thought, RemediationPlanPayload) is not None

    async def should_terminate(
        self, steps: list[AgentStep], context: dict[str, Any]
    ) -> bool:
        return False  # terminates via an accepted final answer

    def parse_final_output(
        self, steps: list[AgentStep], context: dict[str, Any]
    ) -> dict[str, Any]:
        payload = extract_latest_payload(steps, RemediationPlanPayload)
        if payload is None:
            return {
                "summary": "planner did not produce a valid RemediationPlanPayload",
                "confidence": 0.0,
                "needs_more_data": True,
            }
        payload = self._enforce_approval_guardrail(payload)
        return {
            "summary": (
                f"Plan: {len(payload.proposed_actions)} action(s), "
                f"risk={payload.risk_level}, "
                f"requires_approval={payload.requires_approval}"
            ),
            "confidence": payload.confidence,
            **payload.model_dump(),
        }

    @staticmethod
    def _enforce_approval_guardrail(
        plan: RemediationPlanPayload,
    ) -> RemediationPlanPayload:
        """GUARDRAIL — enforced in code, NOT trusted from the model.

        The model emits a requires_approval field, but whatever it says is
        OVERWRITTEN here: an LLM must never auto-approve its own destructive
        action. If the plan is risky, the model is exactly the wrong entity
        to certify it as safe (it's grading its own homework, and it wrote
        the plan under the same blind spots that would make it wrong).

        Approval may be skipped ONLY when ALL of:
          1. risk_level == "low"  (the model's own assessment, lowest stake)
          2. confidence >= 0.85   (no low-conviction auto-runs)
          3. every proposed action's tool is in READ_ONLY_TOOL_ALLOWLIST —
             the condition that actually carries the safety weight: every
             infra-MUTATING tool requires human approval REGARDLESS of
             confidence. 0.99 confident in a restart is still a restart.
        """
        all_read_only = all(
            action.tool_name in READ_ONLY_TOOL_ALLOWLIST
            for action in plan.proposed_actions
        )
        auto_approvable = (
            plan.risk_level == "low"
            and plan.confidence >= AUTO_APPROVE_MIN_CONFIDENCE
            and all_read_only
        )
        plan.requires_approval = not auto_approvable
        return plan
