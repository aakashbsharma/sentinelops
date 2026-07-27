"""Orchestrator core — deterministic control flow over the four agents.

THE HYBRID, in one paragraph: incident handling must be predictable (it can
trigger infra changes), so the orchestrator is NOT another free-form LLM loop
calling arbitrary tools. A deterministic state machine (state_machine.py)
decides WHICH stage runs next; at each transition a small, scoped decision —
informed by the stage agent's confidence and needs_more_data signals, computed
by the pure functions in policy.py — decides HOW to proceed within that stage:
retry with more context, escalate to a human, or continue. Every decision is
persisted as an OrchestratorDecision in a role="orchestrator" AgentMessage, so
the trace shows the routing rationale, not just the agents' outputs.

Two guardrails here are non-negotiable and can never be overridden by any
model output:
  1. should_escalate() == True  ->  the run STOPS at "escalated".
  2. plan.requires_approval     ->  the run STOPS at "awaiting_approval";
     execution only resumes through resume_after_approval().
"""

from collections.abc import Callable
from datetime import datetime, timezone
from typing import Any
from uuid import UUID

import structlog
from pydantic import ValidationError
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.agents.base import AgentLoop
from app.agents.payloads import (
    DiagnosisPayload,
    RemediationPlanPayload,
    TriagePayload,
)
from app.agents.registry import get_agent_class
from app.agents.schemas import AgentOutcome, LoopConfig
from app.agents.tools.base import ToolRegistry
from app.agents.tools.mock_infra import (
    QueryLogsTool,
    QueryMetricsTool,
    RestartServiceTool,
    ScaleDeploymentTool,
)
from app.core import tracing
from app.core.config import get_settings
from app.memory.manager import MemoryManager
from app.models.agent_message import AgentMessage
from app.models.approval import ApprovalRequest
from app.models.incident import Incident
from app.orchestrator import policy
from app.orchestrator.schemas import OrchestratorDecision
from app.orchestrator.state_machine import transition

logger = structlog.get_logger(__name__)

# role -> instantiated agent. Injectable so tests swap in scripted agents;
# the incident is passed so the default factory can seed the mock-infra
# scenario from incident.raw_payload (see mock_infra's determinism note).
AgentFactory = Callable[[str, Incident], AgentLoop]


class OrchestrationRoundsExceeded(RuntimeError):
    """The incident-level safety cap (MAX_ORCHESTRATION_ROUNDS) fired."""


def _default_agent_factory(role: str, incident: Incident) -> AgentLoop:
    settings = get_settings()
    scenario = (incident.raw_payload or {}).get("scenario")
    registry = ToolRegistry()
    registry.register(QueryLogsTool(scenario=scenario))
    registry.register(QueryMetricsTool(scenario=scenario))
    registry.register(RestartServiceTool())
    registry.register(ScaleDeploymentTool())
    config = LoopConfig(
        max_iterations=settings.MAX_AGENT_ITERATIONS,
        token_budget=settings.AGENT_TOKEN_BUDGET,
    )
    return get_agent_class(role)(config, registry)


class Orchestrator:
    def __init__(
        self,
        memory: MemoryManager | None = None,
        agent_factory: AgentFactory | None = None,
    ) -> None:
        self._memory = memory or MemoryManager()
        self._agent_factory = agent_factory or _default_agent_factory

    # ------------------------------------------------------------------ #
    # Entry point 1: a new incident (Celery: orchestrate_incident_task)
    # ------------------------------------------------------------------ #

    async def run_incident(self, incident_id: UUID, db: AsyncSession) -> None:
        """Drive an incident from "open" to one of the three STOP points:
        awaiting_approval (normal), escalated, or resolved (read-only plans).

        The caller owns the transaction (Celery task / test); this method
        only flushes so a mid-run crash can't half-commit an incident.
        """
        # Correlation context (Stage 7): one run_id per run_incident call.
        # Everything below — agents, tools, log lines, persisted rows —
        # inherits incident_id/run_id through contextvars, no threading.
        with tracing.incident_run(incident_id):
            await self._run_incident_inner(incident_id, db)

    async def _run_incident_inner(
        self, incident_id: UUID, db: AsyncSession
    ) -> None:
        incident = await db.get(Incident, incident_id)
        if incident is None:
            raise ValueError(f"Incident {incident_id} not found")
        log = logger.bind(incident_id=str(incident_id))

        # GUARDRAIL (step 10): whatever goes wrong inside — agent bug, DB
        # error, illegal transition from a duplicate task delivery — the
        # incident ends up "escalated" with the error on the record, never
        # silently stuck in an intermediate status.
        try:
            # ---------------- TRIAGE ---------------- #
            transition(incident, "triaging")
            await db.flush()

            triage_context: dict[str, Any] = {
                "incident": self._incident_context(incident),
                **await self._memory.get_relevant_context(incident, db),
            }
            triage_outcome, _ = await self._run_stage_with_retries(
                "triage", incident, triage_context, db, retry_transition=None
            )
            triage_payload = self._parse_payload(triage_outcome, TriagePayload)
            if triage_payload is None:
                # No valid classification even after retries: nothing
                # downstream can be trusted, and "proceed regardless" only
                # applies when we at least have a payload to route on.
                await self._escalate(
                    incident,
                    "Triage did not produce a valid TriagePayload after retries; "
                    "cannot classify the incident.",
                    db,
                )
                return
            # Triage's verdict becomes the incident's severity of record.
            incident.severity = triage_payload.severity

            # ---------------- DIAGNOSE ---------------- #
            transition(incident, "diagnosing")
            await db.flush()

            diagnosis_context: dict[str, Any] = {
                "incident": self._incident_context(incident),
                "triage": triage_payload.model_dump(),
                # Episodic (similar past incidents) + semantic (runbooks).
                **await self._memory.get_relevant_context(incident, db),
            }
            diagnosis_outcome, diagnosis_attempts = await self._run_stage_with_retries(
                "diagnostician",
                incident,
                diagnosis_context,
                db,
                # The documented self-loop: each retry is an explicit
                # diagnosing -> diagnosing transition.
                retry_transition="diagnosing",
            )
            diagnosis_payload = self._parse_payload(diagnosis_outcome, DiagnosisPayload)

            # ------- ESCALATION GATE #1 (non-negotiable) ------- #
            escalate, reason = policy.should_escalate(
                triage_payload,
                diagnosis_payload,
                diagnosis_attempts=diagnosis_attempts,
            )
            if escalate:
                await self._escalate(incident, reason, db)
                return  # a human resumes this incident out-of-band
            if diagnosis_payload is None:
                # Completed retries, no escalation rule fired, but there is
                # no diagnosis to plan from — that's human territory too.
                await self._escalate(
                    incident,
                    "Diagnostician did not produce a valid DiagnosisPayload; "
                    "cannot plan a remediation without a root cause.",
                    db,
                )
                return

            # ---------------- PLAN ---------------- #
            # Runs within the "diagnosing" status — see state_machine.py.
            plan_context: dict[str, Any] = {
                "incident": self._incident_context(incident),
                "triage": triage_payload.model_dump(),
                "diagnosis": diagnosis_payload.model_dump(),
                "semantic_memories": diagnosis_context.get("semantic_memories", []),
            }
            plan_outcome = await self._invoke_agent(
                "remediation_planner", incident, plan_context, db
            )
            plan_payload = self._parse_payload(plan_outcome, RemediationPlanPayload)
            if plan_payload is None:
                await self._escalate(
                    incident,
                    "Remediation planner did not produce a valid "
                    "RemediationPlanPayload; refusing to proceed without a plan.",
                    db,
                )
                return

            # ------- ESCALATION GATE #2 (non-negotiable) ------- #
            # Re-check with the plan present: rule 3 (critical risk) can only
            # fire now.
            escalate, reason = policy.should_escalate(
                triage_payload,
                diagnosis_payload,
                plan=plan_payload,
                diagnosis_attempts=diagnosis_attempts,
            )
            if escalate:
                await self._escalate(incident, reason, db)
                return

            # ------- APPROVAL GATE (non-negotiable) ------- #
            if plan_payload.requires_approval:
                db.add(
                    ApprovalRequest(
                        incident_id=incident.id,
                        proposed_action=plan_payload.model_dump(),
                        risk_level=plan_payload.risk_level,
                        status="pending",
                    )
                )
                transition(incident, "awaiting_approval")
                await self._record_decision(
                    incident,
                    OrchestratorDecision(
                        next_action="await_approval",
                        reasoning=(
                            f"Plan requires approval (risk={plan_payload.risk_level}); "
                            "execution is blocked until a human decides."
                        ),
                        confidence=1.0,
                    ),
                    db,
                )
                await db.flush()
                log.info("stopped_awaiting_approval")
                return  # resumes ONLY via resume_after_approval()

            # ---------------- EXECUTE (auto-approved path) ---------------- #
            # Only reachable for low-risk, high-confidence plans composed
            # entirely of read-only tools (Stage 4's allowlist guardrail).
            # Still leave an ApprovalRequest row so the audit trail records
            # WHO approved: the code guardrail, not a human.
            db.add(
                ApprovalRequest(
                    incident_id=incident.id,
                    proposed_action=plan_payload.model_dump(),
                    risk_level=plan_payload.risk_level,
                    status="approved",
                    resolved_by="guardrail:read_only_allowlist",
                    resolved_at=datetime.now(timezone.utc),
                )
            )
            transition(incident, "remediating")
            await db.flush()
            await self._execute_plan(incident, plan_payload, db)

        except OrchestrationRoundsExceeded as exc:
            await self._force_escalate(incident_id, str(exc), db)
        except Exception as exc:  # noqa: BLE001 — deliberate catch-all, see above
            log.exception("orchestration_failed")
            await self._force_escalate(
                incident_id,
                f"Unhandled orchestration error: {type(exc).__name__}: {exc}",
                db,
            )

    # ------------------------------------------------------------------ #
    # Entry point 2: a human decided (Celery: resume_incident_task)
    # ------------------------------------------------------------------ #

    async def resume_after_approval(
        self, incident_id: UUID, approved: bool, resolved_by: str, db: AsyncSession
    ) -> None:
        # A resume is its OWN run: distinct run_id from the original
        # run_incident call, same incident_id — the trace panel can show
        # "act one" and "act two" separately.
        with tracing.incident_run(incident_id):
            await self._resume_after_approval_inner(
                incident_id, approved, resolved_by, db
            )

    async def _resume_after_approval_inner(
        self, incident_id: UUID, approved: bool, resolved_by: str, db: AsyncSession
    ) -> None:
        incident = await db.get(Incident, incident_id)
        if incident is None:
            raise ValueError(f"Incident {incident_id} not found")

        # Validate BEFORE the escalation-on-error envelope: a bad resume call
        # (no pending approval, e.g. a double-delivered task) must fail the
        # caller, not escalate a perfectly healthy incident.
        approval = (
            await db.execute(
                select(ApprovalRequest)
                .where(
                    ApprovalRequest.incident_id == incident_id,
                    ApprovalRequest.status == "pending",
                )
                .order_by(ApprovalRequest.requested_at.desc())
                .limit(1)
            )
        ).scalar_one_or_none()
        if approval is None:
            raise ValueError(
                f"Incident {incident_id} has no pending ApprovalRequest to resolve"
            )

        try:
            approval.status = "approved" if approved else "rejected"
            approval.resolved_by = resolved_by
            approval.resolved_at = datetime.now(timezone.utc)

            if not approved:
                transition(incident, "closed")
                await self._record_decision(
                    incident,
                    OrchestratorDecision(
                        next_action="escalate",
                        reasoning=(
                            f"Remediation plan rejected by {resolved_by}; "
                            "incident closed without execution."
                        ),
                        confidence=1.0,
                    ),
                    db,
                )
                await db.flush()
                return

            transition(incident, "remediating")
            await db.flush()
            plan_payload = RemediationPlanPayload(**approval.proposed_action)
            await self._execute_plan(incident, plan_payload, db)

        except OrchestrationRoundsExceeded as exc:
            await self._force_escalate(incident_id, str(exc), db)
        except Exception as exc:  # noqa: BLE001
            logger.exception("resume_failed", incident_id=str(incident_id))
            await self._force_escalate(
                incident_id,
                f"Unhandled error while resuming after approval: "
                f"{type(exc).__name__}: {exc}",
                db,
            )

    # ------------------------------------------------------------------ #
    # Stage runners
    # ------------------------------------------------------------------ #

    async def _run_stage_with_retries(
        self,
        role: str,
        incident: Incident,
        context: dict[str, Any],
        db: AsyncSession,
        retry_transition: str | None,
    ) -> tuple[AgentOutcome, int]:
        """Run one stage's agent, re-running (with an augmented context noting
        what was insufficient) while policy.should_retry_with_more_data allows.
        Returns (last outcome, attempts made)."""
        attempts = 0
        while True:
            outcome = await self._invoke_agent(role, incident, context, db)
            attempts += 1
            if not policy.should_retry_with_more_data(outcome, attempts):
                break
            if retry_transition is not None:
                transition(incident, retry_transition)  # explicit self-loop
            context["previous_attempt_insufficient"] = (
                f"Attempt {attempts} ended with needs_more_data=true: "
                f"{outcome.summary}. Use the available tools to gather the "
                "missing evidence before answering this time."
            )
            logger.info(
                "stage_retry",
                incident_id=str(incident.id),
                role=role,
                attempt=attempts + 1,
            )

        if outcome.needs_more_data:
            # Retry budget exhausted and the agent still wants more data.
            # Proceed regardless — the escalation policy downstream decides
            # whether that's tolerable — but never hang here forever.
            logger.warning(
                "stage_needs_more_data_after_retries",
                incident_id=str(incident.id),
                role=role,
                attempts=attempts,
            )
        return outcome, attempts

    async def _invoke_agent(
        self,
        role: str,
        incident: Incident,
        context: dict[str, Any],
        db: AsyncSession,
    ) -> AgentOutcome:
        """Instantiate one agent, run it, persist its outcome as an AgentMessage."""
        # SAFETY CAP: total agent invocations across the WHOLE incident,
        # separate from each agent's own max_iterations. Counted from
        # persisted AgentMessages so it also spans run/resume boundaries.
        # Without this, a triage<->diagnose retry loop multiplied by
        # per-agent iteration budgets could compound into a very expensive
        # run — this is the circuit breaker above all the per-agent ones.
        settings = get_settings()
        used = await self._agent_invocations_so_far(incident.id, db)
        if used >= settings.MAX_ORCHESTRATION_ROUNDS:
            raise OrchestrationRoundsExceeded(
                f"Incident {incident.id} reached the orchestration safety cap: "
                f"{used} agent invocations "
                f"(MAX_ORCHESTRATION_ROUNDS={settings.MAX_ORCHESTRATION_ROUNDS}). "
                "Force-escalating instead of spending further."
            )

        agent = self._agent_factory(role, incident)
        with tracing.span("agent.run", agent_role=role, agent_name=agent.agent_name):
            outcome = await agent.run(context)

        db.add(
            AgentMessage(
                incident_id=incident.id,
                agent_name=agent.agent_name,
                role=role,
                run_id=tracing.current_run_id(),
                content={
                    "status": outcome.status,
                    "summary": outcome.summary,
                    "payload": outcome.payload,
                    "steps": [s.model_dump(mode="json") for s in outcome.steps],
                    "total_tokens_used": outcome.total_tokens_used,
                    # Which model priced these tokens — cost aggregation
                    # (Stage 7) reads this per message.
                    "model": agent.config.model,
                },
                confidence=outcome.confidence,
                needs_more_data=outcome.needs_more_data,
            )
        )
        await db.flush()
        return outcome

    async def _execute_plan(
        self,
        incident: Incident,
        plan: RemediationPlanPayload,
        db: AsyncSession,
    ) -> None:
        """Run the Executor on an approved plan; incident must be 'remediating'."""
        executor_context: dict[str, Any] = {
            "incident": self._incident_context(incident),
            # The Executor's own guardrail re-checks this before running.
            "approval_status": "approved",
            "proposed_actions": [a.model_dump() for a in plan.proposed_actions],
        }
        outcome = await self._invoke_agent("executor", incident, executor_context, db)

        if bool(outcome.payload.get("success")):
            transition(incident, "resolved")
            incident.resolved_at = datetime.now(timezone.utc)
            await self._record_decision(
                incident,
                OrchestratorDecision(
                    next_action="resolve",
                    reasoning=outcome.summary,
                    confidence=outcome.confidence,
                ),
                db,
            )
            # Only now (terminal status) is the outcome a lesson worth
            # remembering — MemoryManager's write policy checks this.
            await self._memory.write_outcome(incident, outcome, db)
        else:
            # The executor aborts on first failure (Stage 4 policy); a
            # half-executed remediation is human territory, not "resolved".
            await self._escalate(
                incident,
                f"Execution did not succeed: {outcome.summary}. "
                f"Errors: {outcome.payload.get('errors', [])}",
                db,
            )
        await db.flush()

    # ------------------------------------------------------------------ #
    # Escalation + bookkeeping
    # ------------------------------------------------------------------ #

    async def _escalate(self, incident: Incident, reason: str, db: AsyncSession) -> None:
        """The legal escalation path: state-machine transition + reason on record."""
        transition(incident, "escalated")
        await self._record_decision(
            incident,
            OrchestratorDecision(next_action="escalate", reasoning=reason, confidence=1.0),
            db,
        )
        await db.flush()
        logger.warning(
            "incident_escalated", incident_id=str(incident.id), reason=reason
        )

    async def _force_escalate(
        self, incident_id: UUID, reason: str, db: AsyncSession
    ) -> None:
        """Last-resort handler: make sure the incident ends up 'escalated'
        with the failure on record, even if the session or status is in an
        arbitrary state. Never raises."""
        try:
            incident = await db.get(Incident, incident_id)
            self._apply_forced_escalation(incident, reason, db)
            await db.flush()
        except Exception:  # noqa: BLE001 — the session may be poisoned; retry clean
            try:
                await db.rollback()
                incident = await db.get(Incident, incident_id)
                self._apply_forced_escalation(incident, reason, db)
                await db.flush()
            except Exception:  # noqa: BLE001
                # Even the recovery failed (DB down?). Log and give up —
                # raising here would just crash the Celery task redundantly.
                logger.exception(
                    "force_escalation_failed", incident_id=str(incident_id)
                )

    @staticmethod
    def _apply_forced_escalation(
        incident: Incident | None, reason: str, db: AsyncSession
    ) -> None:
        if incident is None:
            return
        # Deliberately bypasses the state machine: after an error/rollback the
        # current status is arbitrary, and "never silently stuck" outranks
        # transition legality on this one path.
        incident.status = "escalated"
        db.add(
            AgentMessage(
                incident_id=incident.id,
                agent_name="orchestrator",
                role="orchestrator",
                run_id=tracing.current_run_id(),
                content=OrchestratorDecision(
                    next_action="escalate", reasoning=reason, confidence=1.0
                ).model_dump(),
                confidence=1.0,
                needs_more_data=False,
            )
        )

    async def _record_decision(
        self, incident: Incident, decision: OrchestratorDecision, db: AsyncSession
    ) -> None:
        db.add(
            AgentMessage(
                incident_id=incident.id,
                agent_name="orchestrator",
                role="orchestrator",
                run_id=tracing.current_run_id(),
                content=decision.model_dump(),
                confidence=decision.confidence,
                needs_more_data=False,
            )
        )

    @staticmethod
    async def _agent_invocations_so_far(incident_id: UUID, db: AsyncSession) -> int:
        return (
            await db.execute(
                select(func.count())
                .select_from(AgentMessage)
                .where(
                    AgentMessage.incident_id == incident_id,
                    AgentMessage.role != "orchestrator",
                )
            )
        ).scalar_one()

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def _incident_context(incident: Incident) -> dict[str, Any]:
        return {
            "id": str(incident.id),
            "title": incident.title,
            "description": incident.description,
            "source": incident.source,
            "severity": incident.severity,
            "status": incident.status,
            "raw_payload": incident.raw_payload or {},
        }

    @staticmethod
    def _parse_payload(
        outcome: AgentOutcome, model_cls: type[TriagePayload] | type[DiagnosisPayload] | type[RemediationPlanPayload]
    ) -> Any:
        """Rebuild a typed payload from an AgentOutcome (the loop lifts
        confidence/needs_more_data off the payload dict; put them back).
        Returns None when the outcome carries no valid payload — the caller
        decides how to route that."""
        data = {
            **outcome.payload,
            "confidence": outcome.confidence,
            "needs_more_data": outcome.needs_more_data,
        }
        try:
            return model_cls(**data)
        except ValidationError:
            return None
