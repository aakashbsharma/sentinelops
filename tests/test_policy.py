"""Policy tests — completely isolated, exactly as designed.

policy.py is pure functions of typed payloads + config: no agents, no DB, no
LLM anywhere in this file. Each should_escalate rule is checked both on its
firing condition AND on the adjacent conditions that must NOT fire it.
"""

from app.agents.payloads import (
    DiagnosisPayload,
    ProposedAction,
    RemediationPlanPayload,
    TriagePayload,
)
from app.agents.schemas import AgentOutcome
from app.core.config import get_settings
from app.orchestrator.policy import should_escalate, should_retry_with_more_data

# ---------------------------------------------------------------------- #
# Constructed payloads — no agent ever runs in this file.
# ---------------------------------------------------------------------- #


def make_triage(severity: str = "medium", confidence: float = 0.9) -> TriagePayload:
    return TriagePayload(
        severity=severity,
        category="resource_exhaustion",
        confidence=confidence,
        needs_more_data=False,
        reasoning="constructed for tests",
    )


def make_diagnosis(
    confidence: float = 0.9, needs_more_data: bool = False
) -> DiagnosisPayload:
    return DiagnosisPayload(
        root_cause="connection pool exhaustion",
        evidence=["pool timeout errors in logs"],
        confidence=confidence,
        needs_more_data=needs_more_data,
    )


def make_plan(
    risk_level: str = "medium", confidence: float = 0.9
) -> RemediationPlanPayload:
    return RemediationPlanPayload(
        proposed_actions=[
            ProposedAction(
                action_type="restart",
                target="payment-service",
                tool_name="restart_service",
                parameters={"service_name": "payment-service"},
                rationale="clear exhausted pool",
            )
        ],
        risk_level=risk_level,
        requires_approval=True,
        rollback_plan="scale back and page on-call",
        confidence=confidence,
    )


def make_outcome(needs_more_data: bool) -> AgentOutcome:
    return AgentOutcome(
        status="needs_more_data" if needs_more_data else "completed",
        summary="constructed for tests",
        confidence=0.5,
        needs_more_data=needs_more_data,
    )


# ---------------------------------------------------------------------- #
# should_retry_with_more_data
# ---------------------------------------------------------------------- #


def test_retry_when_needs_more_data_and_budget_remains() -> None:
    assert get_settings().MAX_RETRY_ATTEMPTS == 2  # default this suite assumes
    assert should_retry_with_more_data(make_outcome(True), attempts_so_far=1) is True


def test_no_retry_once_attempts_reach_cap() -> None:
    assert should_retry_with_more_data(make_outcome(True), attempts_so_far=2) is False


def test_no_retry_when_agent_is_satisfied() -> None:
    # needs_more_data=False must never trigger a retry, even with budget left.
    assert should_retry_with_more_data(make_outcome(False), attempts_so_far=1) is False


# ---------------------------------------------------------------------- #
# should_escalate — RULE 1: high-stakes severity + low-confidence diagnosis
# ---------------------------------------------------------------------- #


def test_rule1_fires_on_high_severity_low_confidence() -> None:
    escalate, reason = should_escalate(
        make_triage(severity="high"), make_diagnosis(confidence=0.4)
    )
    assert escalate is True
    assert "confidence" in reason


def test_rule1_fires_on_critical_severity_too() -> None:
    escalate, _ = should_escalate(
        make_triage(severity="critical"), make_diagnosis(confidence=0.49)
    )
    assert escalate is True


def test_rule1_does_not_fire_at_confidence_boundary() -> None:
    # The rule is strictly < 0.5; exactly 0.5 must not escalate.
    escalate, reason = should_escalate(
        make_triage(severity="high"), make_diagnosis(confidence=0.5)
    )
    assert (escalate, reason) == (False, "")


def test_rule1_does_not_fire_on_low_stakes_severity() -> None:
    # Same shaky diagnosis, but medium severity — a human isn't needed.
    escalate, reason = should_escalate(
        make_triage(severity="medium"), make_diagnosis(confidence=0.4)
    )
    assert (escalate, reason) == (False, "")


# ---------------------------------------------------------------------- #
# should_escalate — RULE 2: diagnostician stuck after retries
# ---------------------------------------------------------------------- #


def test_rule2_fires_when_still_starved_after_max_retries() -> None:
    escalate, reason = should_escalate(
        make_triage(severity="low"),
        make_diagnosis(confidence=0.9, needs_more_data=True),
        diagnosis_attempts=2,
    )
    assert escalate is True
    assert "needs_more_data" in reason


def test_rule2_does_not_fire_with_retry_budget_remaining() -> None:
    escalate, reason = should_escalate(
        make_triage(severity="low"),
        make_diagnosis(confidence=0.9, needs_more_data=True),
        diagnosis_attempts=1,
    )
    assert (escalate, reason) == (False, "")


def test_rule2_does_not_fire_when_data_needs_were_met() -> None:
    # Many attempts but needs_more_data resolved along the way: no escalation.
    escalate, reason = should_escalate(
        make_triage(severity="low"),
        make_diagnosis(confidence=0.9, needs_more_data=False),
        diagnosis_attempts=5,
    )
    assert (escalate, reason) == (False, "")


# ---------------------------------------------------------------------- #
# should_escalate — RULE 3: critical-risk plan, regardless of confidence
# ---------------------------------------------------------------------- #


def test_rule3_fires_on_critical_risk_even_at_max_confidence() -> None:
    escalate, reason = should_escalate(
        make_triage(severity="low"),
        make_diagnosis(confidence=0.99),
        plan=make_plan(risk_level="critical", confidence=0.99),
    )
    assert escalate is True
    assert "critical" in reason


def test_rule3_does_not_fire_on_high_risk() -> None:
    # 'high' is the adjacent risk category — the full-stop rule is only for
    # 'critical' (high-risk plans are still caught by the approval gate).
    escalate, reason = should_escalate(
        make_triage(severity="low"),
        make_diagnosis(confidence=0.9),
        plan=make_plan(risk_level="high"),
    )
    assert (escalate, reason) == (False, "")


def test_no_rule_fires_on_a_healthy_run() -> None:
    escalate, reason = should_escalate(
        make_triage(severity="high", confidence=0.9),
        make_diagnosis(confidence=0.85),
        plan=make_plan(risk_level="medium", confidence=0.9),
        diagnosis_attempts=1,
    )
    assert (escalate, reason) == (False, "")
