"""The Incident.status state machine — the deterministic half of the orchestrator.

WHICH stage runs next is never an LLM's call: the legal moves are this explicit
dict, checked on every transition. An illegal transition is a programming error
(or a duplicate/late task delivery) and raises loudly — it is never silently
allowed, because a status that skipped a gate (approval, escalation) is exactly
the failure mode this whole design exists to prevent.

Note on remediation *planning*: there is deliberately no "planning" status.
The RemediationPlanner runs while the incident is still in "diagnosing" — a
plan is the output of the diagnostic phase — and the status only changes once
the plan's disposition is known: awaiting_approval (the normal case),
remediating (rare auto-approved read-only plans), or escalated.
"""

import structlog

from app.models.incident import Incident

logger = structlog.get_logger(__name__)

VALID_TRANSITIONS: dict[str, set[str]] = {
    "open": {"triaging"},
    "triaging": {"diagnosing", "escalated"},
    # Self-loop = "retry diagnosis with more data" (bounded by
    # settings.MAX_RETRY_ATTEMPTS; see policy.should_retry_with_more_data).
    # "remediating" is the rare auto-approved read-only-plan path.
    "diagnosing": {"awaiting_approval", "escalated", "diagnosing", "remediating"},
    # closed = a human rejected the proposed plan.
    "awaiting_approval": {"remediating", "escalated", "closed"},
    "remediating": {"resolved", "escalated"},
    "resolved": {"closed"},
    # A human can close an escalated incident or hand it back to the pipeline.
    "escalated": {"closed", "triaging"},
    "closed": set(),  # terminal
}


class IllegalTransitionError(RuntimeError):
    """Raised on any Incident.status change not in VALID_TRANSITIONS."""


def transition(incident: Incident, new_status: str) -> None:
    """Move an incident to `new_status`, or raise IllegalTransitionError.

    This is the ONLY sanctioned way to change Incident.status during
    orchestration. (The single exception: the orchestrator's last-resort
    error handler force-sets "escalated" when the incident may be in an
    arbitrary state after a rollback — see Orchestrator._force_escalate.)
    """
    current = incident.status
    allowed = VALID_TRANSITIONS.get(current)
    if allowed is None:
        raise IllegalTransitionError(
            f"Incident {incident.id} has unknown status {current!r}; "
            f"known statuses: {sorted(VALID_TRANSITIONS)}"
        )
    if new_status not in allowed:
        raise IllegalTransitionError(
            f"Illegal incident transition {current!r} -> {new_status!r} "
            f"for incident {incident.id}; allowed from {current!r}: "
            f"{sorted(allowed) or '(none — terminal state)'}"
        )
    incident.status = new_status
    logger.info(
        "incident_transition",
        incident_id=str(incident.id),
        from_status=current,
        to_status=new_status,
    )
