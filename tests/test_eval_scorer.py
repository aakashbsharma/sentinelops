"""Eval scorer unit tests — hermetic, NO LLM calls, NO real embeddings.

The eval HARNESS (eval/run_eval.py) is deliberately not exercised by pytest
(it costs real money); the scoring logic, however, is pure and must be
correct, so it's unit-tested here with the HashingEmbeddingProvider.
"""

import pytest

from app.agents.payloads import (
    DiagnosisPayload,
    RemediationPlanPayload,
    TriagePayload,
)
from app.memory.embeddings import HashingEmbeddingProvider
from eval.scorer import (
    Fixture,
    GroundTruth,
    RunRecord,
    aggregate,
    build_ideal_root_cause,
    score_incident,
    score_root_cause,
)

PROVIDER = HashingEmbeddingProvider()


def make_fixture(**truth_overrides) -> Fixture:
    truth = {
        "expected_severity": "high",
        "expected_category": "database",
        "expected_root_cause_keywords": [
            "database connection pool exhausted",
            "queries waiting to acquire a connection",
        ],
        "expected_risk_level": "medium",
        "should_require_approval": True,
        "is_actually_critical": False,
        **truth_overrides,
    }
    return Fixture(
        id="fx",
        title="t",
        raw_payload={},
        ground_truth=GroundTruth(**truth),
    )


def make_record(**overrides) -> RunRecord:
    defaults = dict(
        fixture_id="fx",
        incident_id="00000000-0000-0000-0000-000000000000",
        final_status="awaiting_approval",
        triage=TriagePayload(
            severity="high",
            category="database",
            confidence=0.9,
            reasoning="pool timeouts",
        ),
        diagnosis=DiagnosisPayload(
            root_cause=(
                "database connection pool exhausted; queries waiting to "
                "acquire a connection time out"
            ),
            evidence=["pool timeout errors"],
            contributing_factors=[],
            confidence=0.85,
        ),
        plan=RemediationPlanPayload(
            proposed_actions=[],
            risk_level="medium",
            requires_approval=True,
            rollback_plan="revert",
            confidence=0.9,
        ),
        auto_approved=False,
    )
    return RunRecord(**{**defaults, **overrides})


# ---------------------------------------------------------------------- #
# Root-cause similarity
# ---------------------------------------------------------------------- #


def test_ideal_root_cause_composition() -> None:
    text = build_ideal_root_cause(["a pool problem", "timeouts"])
    assert "a pool problem" in text and "timeouts" in text


@pytest.mark.asyncio
async def test_same_mechanism_scores_higher_than_different() -> None:
    keywords = [
        "database connection pool exhausted",
        "queries waiting to acquire a connection",
    ]
    same = await score_root_cause(
        "the database connection pool is exhausted and queries are waiting "
        "to acquire a connection",
        keywords,
        PROVIDER,
    )
    different = await score_root_cause(
        "a bad deployment shipped a broken configuration file to the fleet",
        keywords,
        PROVIDER,
    )
    assert same > different
    assert same > 0.5  # heavy token overlap must land clearly above noise


# ---------------------------------------------------------------------- #
# Per-incident scoring
# ---------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_correct_run_passes_all_checks() -> None:
    score = await score_incident(
        make_fixture(), make_record(), PROVIDER, threshold=0.5
    )
    assert score.severity_correct and score.category_correct
    assert score.root_cause_correct is True
    assert score.approval_gate_correct is True
    assert score.escalation_ok is None  # not a critical fixture
    assert not score.false_negative_critical
    assert score.passed


@pytest.mark.asyncio
async def test_wrong_severity_fails() -> None:
    record = make_record(
        triage=TriagePayload(
            severity="low", category="database", confidence=0.9, reasoning="meh"
        )
    )
    score = await score_incident(make_fixture(), record, PROVIDER, threshold=0.5)
    assert not score.severity_correct
    assert not score.passed


@pytest.mark.asyncio
async def test_missing_triage_counts_as_failure_not_skip() -> None:
    record = make_record(triage=None, final_status="escalated")
    score = await score_incident(make_fixture(), record, PROVIDER, threshold=0.5)
    assert score.severity_correct is False
    assert score.category_correct is False
    assert not score.passed


@pytest.mark.asyncio
async def test_approval_gate_mismatch_fails() -> None:
    record = make_record(
        plan=RemediationPlanPayload(
            proposed_actions=[],
            risk_level="low",
            requires_approval=False,  # ground truth demands approval
            rollback_plan="n/a",
            confidence=0.9,
        )
    )
    score = await score_incident(make_fixture(), record, PROVIDER, threshold=0.5)
    assert score.approval_gate_correct is False
    assert not score.passed


# ---------------------------------------------------------------------- #
# Escalation recall / zero-tolerance false negatives
# ---------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_critical_escalated_is_acceptable() -> None:
    fixture = make_fixture(is_actually_critical=True, expected_severity="critical")
    record = make_record(
        final_status="escalated",
        triage=TriagePayload(
            severity="critical", category="database", confidence=0.9, reasoning="r"
        ),
        diagnosis=None,  # escalated before a diagnosis existed — legitimate
        plan=None,
    )
    score = await score_incident(fixture, record, PROVIDER, threshold=0.5)
    assert score.escalation_ok is True
    assert not score.false_negative_critical
    assert score.root_cause_correct is None  # not applicable, not a failure
    assert score.passed


@pytest.mark.asyncio
async def test_critical_auto_remediated_on_shaky_diagnosis_is_hard_failure() -> None:
    fixture = make_fixture(is_actually_critical=True, expected_severity="critical")
    record = make_record(
        final_status="resolved",
        auto_approved=True,  # the code guardrail, not a human, approved it
        triage=TriagePayload(
            severity="critical", category="database", confidence=0.9, reasoning="r"
        ),
        diagnosis=DiagnosisPayload(
            root_cause="probably some kind of network blip, unclear",
            evidence=[],
            contributing_factors=[],
            confidence=0.3,  # a coin-flip diagnosis
        ),
    )
    score = await score_incident(fixture, record, PROVIDER, threshold=0.5)
    assert score.false_negative_critical
    assert score.escalation_ok is False
    assert not score.passed


@pytest.mark.asyncio
async def test_critical_triaged_low_and_handled_without_human_is_hard_failure() -> None:
    fixture = make_fixture(is_actually_critical=True, expected_severity="critical")
    record = make_record(
        final_status="resolved",
        auto_approved=True,
        triage=TriagePayload(
            severity="low",  # criticality missed at triage
            category="database",
            confidence=0.9,
            reasoning="looks minor",
        ),
    )
    score = await score_incident(fixture, record, PROVIDER, threshold=0.5)
    assert score.false_negative_critical
    assert not score.passed


@pytest.mark.asyncio
async def test_critical_resolved_correctly_with_confidence_is_acceptable() -> None:
    """Competence is allowed: a correct, confident resolution of a critical
    incident is not a false negative."""
    fixture = make_fixture(is_actually_critical=True, expected_severity="critical")
    record = make_record(
        final_status="resolved",
        auto_approved=True,
        triage=TriagePayload(
            severity="critical", category="database", confidence=0.95, reasoning="r"
        ),
    )
    score = await score_incident(fixture, record, PROVIDER, threshold=0.5)
    assert score.escalation_ok is True
    assert not score.false_negative_critical


# ---------------------------------------------------------------------- #
# Aggregation
# ---------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_aggregate_summary_counts() -> None:
    good = await score_incident(make_fixture(), make_record(), PROVIDER, threshold=0.5)
    bad = await score_incident(
        make_fixture(),
        make_record(
            triage=TriagePayload(
                severity="low", category="network", confidence=0.5, reasoning="r"
            )
        ),
        PROVIDER,
        threshold=0.5,
    )
    report = aggregate([good, bad])

    assert report.summary.total == 2
    assert report.summary.passed == 1
    assert report.summary.severity_accuracy == 50.0
    assert report.summary.category_accuracy == 50.0
    assert report.summary.false_negative_critical_count == 0
    assert report.summary.root_cause_avg_similarity is not None
