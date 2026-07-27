"""Disagreement / low-confidence routing policy — pure functions, on purpose.

These decisions (retry? escalate?) are the part of the orchestrator most worth
being able to reason about precisely, so they are NOT buried in core.py's
control flow. Everything here is a pure function of typed payloads + config:
no LLM call, no DB session, no I/O of any kind. That's what makes each rule
unit-testable in complete isolation (see tests/test_policy.py) and auditable
in a code review. The orchestrator's job is to call these and act on the
result — never to second-guess them.
"""

from app.agents.payloads import DiagnosisPayload, RemediationPlanPayload, TriagePayload
from app.agents.schemas import AgentOutcome
from app.core.config import get_settings

# Severities where acting on a shaky diagnosis is unacceptable.
HIGH_STAKES_SEVERITIES = frozenset({"high", "critical"})

# Below this diagnostician confidence, a high-stakes incident goes to a human.
MIN_HIGH_STAKES_DIAGNOSIS_CONFIDENCE = 0.5


def should_retry_with_more_data(outcome: AgentOutcome, attempts_so_far: int) -> bool:
    """Retry a stage iff its agent asked for more data AND we have retry budget.

    `attempts_so_far` counts attempts already made (the first run included),
    so with MAX_RETRY_ATTEMPTS=2 a stage runs at most twice.
    """
    return (
        outcome.needs_more_data
        and attempts_so_far < get_settings().MAX_RETRY_ATTEMPTS
    )


def should_escalate(
    triage: TriagePayload,
    diagnosis: DiagnosisPayload | None,
    plan: RemediationPlanPayload | None = None,
    diagnosis_attempts: int = 0,
) -> tuple[bool, str]:
    """Return (True, reason) when there is a genuine disagreement or risk
    signal worth a human's attention; (False, "") otherwise.

    Called twice per incident: after diagnosis (plan=None) and again after
    planning (plan set) — each rule only fires once its inputs exist.
    """
    # RULE 1 — high-stakes severity, low-confidence diagnosis.
    # Triage says the incident is high/critical, but the diagnostician is
    # less than 50% sure of the root cause. Auto-proceeding would mean
    # remediating a critical incident on a coin-flip diagnosis — exactly
    # the disagreement a human should arbitrate.
    if (
        triage.severity in HIGH_STAKES_SEVERITIES
        and diagnosis is not None
        and diagnosis.confidence < MIN_HIGH_STAKES_DIAGNOSIS_CONFIDENCE
    ):
        return (
            True,
            f"Triage rated severity {triage.severity!r} but the diagnostician's "
            f"confidence is only {diagnosis.confidence:.2f} "
            f"(< {MIN_HIGH_STAKES_DIAGNOSIS_CONFIDENCE}) — high-stakes incident "
            "with a low-confidence diagnosis must not auto-proceed.",
        )

    # RULE 2 — diagnostician still starved after exhausting retries.
    # needs_more_data after MAX_RETRY_ATTEMPTS means the agent is stuck: the
    # data it wants doesn't exist or its tools can't reach it. Looping further
    # burns tokens without converging — hand it to a human instead.
    if (
        diagnosis is not None
        and diagnosis.needs_more_data
        and diagnosis_attempts >= get_settings().MAX_RETRY_ATTEMPTS
    ):
        return (
            True,
            f"Diagnostician still reports needs_more_data after "
            f"{diagnosis_attempts} attempt(s) "
            f"(MAX_RETRY_ATTEMPTS={get_settings().MAX_RETRY_ATTEMPTS}) — "
            "the agent is stuck; escalating instead of looping forever.",
        )

    # RULE 3 — critical-risk remediation plan, regardless of confidence.
    # The highest-risk category always gets a human, full stop. Confidence is
    # irrelevant here: a model 0.99-confident in a critical-blast-radius
    # action is still proposing a critical-blast-radius action.
    if plan is not None and plan.risk_level == "critical":
        return (
            True,
            "Remediation plan risk_level is 'critical' — the highest-risk "
            "category always requires human attention, regardless of the "
            f"planner's confidence ({plan.confidence:.2f}).",
        )

    return (False, "")
