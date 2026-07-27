"""ExecutorAgent: execute an APPROVED remediation plan, action by action.

Deliberately the most constrained agent in the system. It subclasses
AgentLoop for consistency (same outcome type, same trace, same guardrails),
but its "planner" is NOT an LLM — it's a queue of pre-approved actions.
Giving the executor LLM planning freedom would void the approval: a human
approved THIS plan, not whatever a model improvises at execution time.

Expected context (assembled by the Stage 5 orchestrator):
  - "approval_status": the incident's ApprovalRequest.status (must be "approved")
  - "proposed_actions": list of ProposedAction dicts from the approved plan
"""

import json
from typing import Any

from app.agents.base import AgentLoop
from app.agents.payloads import ExecutionPayload
from app.agents.schemas import (
    AgentOutcome,
    AgentStep,
    LoopConfig,
    PlanDecision,
    ToolCall,
)
from app.agents.tools.base import ToolRegistry
from app.core.config import get_settings


class ExecutorAgent(AgentLoop):
    agent_name = "executor"

    def __init__(self, config: LoopConfig, tool_registry: ToolRegistry) -> None:
        # Reflection is an LLM self-critique step; the executor must stay
        # fully deterministic (no LLM calls at all), so it's forced off.
        super().__init__(
            config.model_copy(update={"reflect_every_step": False}), tool_registry
        )

    async def run(self, context: dict[str, Any]) -> AgentOutcome:
        # GUARDRAIL: no approval record, no execution. This check is
        # deliberately BEFORE the loop even starts — an executor that begins
        # working and checks approval later has already leaked actions. The
        # orchestrator passes ApprovalRequest.status here; anything but
        # "approved" (pending, rejected, auto_denied, missing) is a refusal.
        if context.get("approval_status") != "approved":
            return AgentOutcome(
                status="error",
                summary=(
                    "Execution refused: incident has no approved ApprovalRequest "
                    f"(approval_status={context.get('approval_status')!r})."
                ),
                confidence=0.0,
                needs_more_data=False,
                payload=ExecutionPayload(
                    actions_taken=[],
                    success=False,
                    dry_run=True,
                    errors=["no approved ApprovalRequest"],
                ).model_dump(),
                steps=[],
                total_tokens_used=0,
            )

        context["_dry_run"] = self._resolve_dry_run()
        return await super().run(context)

    @staticmethod
    def _resolve_dry_run() -> bool:
        """PRODUCTION SWITCH — the single place real execution is enabled.

        For this project every execution is dry_run=True, even for approved
        plans. A real deployment flips EXECUTOR_ALLOW_REAL_EXECUTION=true in
        the environment — one line, greppable, auditable in config diffs and
        deploy history. That is deliberate: if "is this a drill?" were decided
        by scattered per-callsite logic, you could never answer "can this
        system touch prod right now?" with a single look.
        (A hardened version would also gate on ENVIRONMENT and require the
        flag plus a per-incident override — same single-switch principle.)
        """
        return not get_settings().EXECUTOR_ALLOW_REAL_EXECUTION

    # ------------------------------------------------------------------ #
    # Deterministic "planner": pop the next approved action off the queue.
    # ------------------------------------------------------------------ #

    async def _plan_step(
        self, context: dict[str, Any], prior_steps: list[AgentStep]
    ) -> tuple[PlanDecision, int]:
        actions: list[dict[str, Any]] = context.get("proposed_actions", [])
        executed = [s for s in prior_steps if s.action is not None]

        # FAILURE POLICY: abort remaining actions on the first failure.
        # A multi-step remediation is usually order-dependent (restart THEN
        # scale); pushing on after step 1 failed can leave the system in a
        # worse, undocumented half-remediated state that no runbook describes.
        # Stop, report, let a human decide.
        if executed and self._step_failed(executed[-1]):
            return (
                PlanDecision(
                    thought=(
                        f"Aborting: action {len(executed)} of {len(actions)} failed. "
                        "Remaining actions will not be executed."
                    ),
                    final_answer="aborted on first failure",
                ),
                0,  # no LLM call, no tokens
            )

        if len(executed) >= len(actions):
            return (
                PlanDecision(
                    thought=f"All {len(actions)} approved action(s) executed.",
                    final_answer="done",
                ),
                0,
            )

        action = actions[len(executed)]
        arguments = dict(action.get("parameters", {}))
        # GUARDRAIL: the executor FORCES dry_run from config, overriding
        # whatever the approved plan's parameters said. A plan that claims
        # dry_run=false must not be able to smuggle real execution through
        # a system configured for drills.
        arguments["dry_run"] = context.get("_dry_run", True)

        return (
            PlanDecision(
                thought=(
                    f"Executing approved action {len(executed) + 1}/{len(actions)}: "
                    f"{action.get('tool_name')} on {action.get('target', '?')}"
                ),
                tool_call=ToolCall(
                    tool_name=str(action.get("tool_name")), arguments=arguments
                ),
            ),
            0,
        )

    async def build_system_prompt(self, context: dict[str, Any]) -> str:
        return "Executor runs pre-approved actions; it does not prompt an LLM."

    async def should_terminate(
        self, steps: list[AgentStep], context: dict[str, Any]
    ) -> bool:
        return False  # _plan_step emits the terminating final answers

    def parse_final_output(
        self, steps: list[AgentStep], context: dict[str, Any]
    ) -> dict[str, Any]:
        actions_taken: list[dict[str, Any]] = []
        errors: list[str] = []

        for step in steps:
            if step.action is None:
                continue
            observation = self._parse_observation(step)
            entry: dict[str, Any] = {
                "tool_name": step.action.tool_name,
                "arguments": step.action.arguments,
                "success": bool(observation.get("success")),
            }
            if entry["success"]:
                entry["result"] = observation.get("output")
            else:
                error = str(observation.get("error", "unknown error"))
                entry["error"] = error
                errors.append(f"{step.action.tool_name}: {error}")
            actions_taken.append(entry)

        planned = len(context.get("proposed_actions", []))
        success = not errors and len(actions_taken) == planned
        payload = ExecutionPayload(
            actions_taken=actions_taken,
            success=success,
            dry_run=context.get("_dry_run", True),
            errors=errors,
        )
        return {
            "summary": (
                f"Executed {len(actions_taken)}/{planned} action(s) "
                f"(dry_run={payload.dry_run}); success={success}"
            ),
            # Deterministic executor: outcomes are facts, not hypotheses.
            "confidence": 1.0 if success else 0.2,
            **payload.model_dump(),
        }

    @staticmethod
    def _step_failed(step: AgentStep) -> bool:
        return not ExecutorAgent._parse_observation(step).get("success", False)

    @staticmethod
    def _parse_observation(step: AgentStep) -> dict[str, Any]:
        if not step.observation:
            return {}
        try:
            data = json.loads(step.observation)
            return data if isinstance(data, dict) else {}
        except json.JSONDecodeError:
            return {}
