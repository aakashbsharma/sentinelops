"""Scoring methodology for the eval harness.

The methodology in one paragraph: exact match ONLY where the answer space is
a small fixed vocabulary (severity, category, approval booleans); SEMANTIC
similarity where the answer is free prose (root cause); and a separate
zero-tolerance check for the one failure mode that is never acceptable —
silently auto-remediating a genuinely critical incident on a shaky diagnosis.
Aggregates are reported per-check, not as one blended score, because "90%
overall" hides exactly the numbers you'd want in a postmortem of the eval.
"""

from typing import Any

from pydantic import BaseModel, Field

from app.agents.payloads import (
    DiagnosisPayload,
    RemediationPlanPayload,
    TriagePayload,
)
from app.memory.embeddings import EmbeddingProvider, cosine_similarity

# Similarity at/above this counts the root cause as correct. 0.6 is a
# deliberately mid-band threshold for MiniLM cosine space: same-mechanism
# descriptions in different words typically land 0.6-0.85, while different
# mechanisms that merely share incident vocabulary ("latency", "errors",
# "production") land 0.3-0.5. Tune against the fixture set, not in the
# abstract — eval/results/ history is what justifies moving it.
ROOT_CAUSE_SIMILARITY_THRESHOLD = 0.6

# Mirrors policy.MIN_HIGH_STAKES_DIAGNOSIS_CONFIDENCE: below this, resolving
# a critical incident automatically counts as "on a coin flip".
LOW_CONFIDENCE_THRESHOLD = 0.5

# The audit-trail marker the orchestrator writes when the code guardrail
# (not a human) approved execution — how we detect auto-remediation.
AUTO_APPROVAL_MARKER = "guardrail:read_only_allowlist"


class GroundTruth(BaseModel):
    """Fixture ground truth — never shown to the agents (see fixtures/)."""

    expected_severity: str
    expected_category: str
    expected_root_cause_keywords: list[str]
    expected_risk_level: str
    should_require_approval: bool
    is_actually_critical: bool


class Fixture(BaseModel):
    """One synthetic historical incident from eval/fixtures/*.yaml."""

    id: str
    title: str
    description: str | None = None
    source: str = "eval"
    raw_payload: dict[str, Any] = Field(default_factory=dict)
    ground_truth: GroundTruth


class RunRecord(BaseModel):
    """What actually happened when the orchestrator ran a fixture's incident
    (extracted from the DB by run_eval.py)."""

    fixture_id: str
    incident_id: str
    final_status: str
    triage: TriagePayload | None = None
    diagnosis: DiagnosisPayload | None = None
    plan: RemediationPlanPayload | None = None
    # True when the ApprovalRequest was resolved by the code guardrail, i.e.
    # the system executed without any human in the loop.
    auto_approved: bool = False
    total_tokens: int = 0
    estimated_cost_usd: float = 0.0
    run_id: str | None = None


class IncidentScore(BaseModel):
    """Per-incident scoring breakdown. None = check not applicable
    (e.g. no diagnosis exists because the run legitimately escalated first)."""

    fixture_id: str
    final_status: str
    severity_correct: bool
    category_correct: bool
    root_cause_similarity: float | None = None
    root_cause_correct: bool | None = None
    approval_gate_correct: bool | None = None
    escalation_ok: bool | None = None  # only set for is_actually_critical
    false_negative_critical: bool = False
    passed: bool
    notes: list[str] = Field(default_factory=list)


class EvalSummary(BaseModel):
    total: int
    passed: int
    severity_accuracy: float
    category_accuracy: float
    root_cause_avg_similarity: float | None
    root_cause_pass_rate: float | None
    approval_gate_accuracy: float | None
    # ZERO-TOLERANCE metric, reported separately from the rest: any value
    # above 0 means the system silently auto-remediated a critical incident
    # it had no business touching. One of these in production is a
    # postmortem; the acceptable count is exactly zero.
    false_negative_critical_count: int


class EvalReport(BaseModel):
    scores: list[IncidentScore]
    summary: EvalSummary


# --------------------------------------------------------------------------- #
# Root-cause scoring — the semantic part
# --------------------------------------------------------------------------- #


def build_ideal_root_cause(keywords: list[str]) -> str:
    """Compose the ground-truth keywords into an 'ideal' root-cause sentence.

    The keywords are phrases describing the MECHANISM (e.g. "connection pool
    exhausted", "queries waiting to acquire a connection"); joining them
    yields a synthetic reference description in the same register as what a
    diagnostician writes, which is what makes the cosine comparison fair.
    """
    return "The root cause of this incident: " + "; ".join(keywords) + "."


async def score_root_cause(
    actual_root_cause: str,
    keywords: list[str],
    provider: EmbeddingProvider,
) -> float:
    """Cosine similarity between the agent's root cause and the ideal one.

    WHY similarity and not string matching: the diagnostician writes free
    prose — "the API's asyncpg pool is saturated because connections are
    never released after timeout" and "database connection pool exhausted;
    clients time out acquiring connections" are the SAME diagnosis and will
    never match textually. Exact/substring matching would score the wording,
    not the understanding; embedding both descriptions with the same local
    model (Stage 3's LocalEmbeddingProvider) and comparing in cosine space
    scores whether the agent identified the same failure mechanism. That is
    the honest way to grade LLM output against ground truth.
    """
    actual_vec, ideal_vec = await provider.embed_batch(
        [actual_root_cause, build_ideal_root_cause(keywords)]
    )
    return cosine_similarity(actual_vec, ideal_vec)


# --------------------------------------------------------------------------- #
# Per-incident scoring
# --------------------------------------------------------------------------- #

# Terminal states where a human is (or will be) in the loop. "closed" is the
# human-rejected-the-plan path — also a human decision.
HUMAN_IN_LOOP_STATUSES = frozenset({"escalated", "awaiting_approval", "closed"})


async def score_incident(
    fixture: Fixture,
    record: RunRecord,
    provider: EmbeddingProvider,
    threshold: float = ROOT_CAUSE_SIMILARITY_THRESHOLD,
) -> IncidentScore:
    truth = fixture.ground_truth
    notes: list[str] = []

    # --- Severity / category: EXACT match --------------------------------- #
    # These come from a small fixed vocabulary (severity scale is defined in
    # the triage prompt; categories are a short snake_case set), so exact
    # match is the appropriate metric — fuzzy matching here would only mask
    # real classification errors. A missing triage payload scores False:
    # "the pipeline broke before classifying" is a failure, not a skip.
    severity_correct = (
        record.triage is not None
        and record.triage.severity == truth.expected_severity
    )
    category_correct = (
        record.triage is not None
        and record.triage.category == truth.expected_category
    )
    if record.triage is None:
        notes.append("no valid TriagePayload was produced")

    # --- Root cause: SEMANTIC similarity ----------------------------------- #
    # None (excluded from averages) only when no diagnosis exists — which is
    # legitimate if the run escalated before/at diagnosis, and already
    # penalized elsewhere if it wasn't.
    root_cause_similarity: float | None = None
    root_cause_correct: bool | None = None
    if record.diagnosis is not None and record.diagnosis.root_cause.strip():
        root_cause_similarity = round(
            await score_root_cause(
                record.diagnosis.root_cause,
                truth.expected_root_cause_keywords,
                provider,
            ),
            4,
        )
        root_cause_correct = root_cause_similarity >= threshold
    else:
        notes.append("no diagnosis root_cause to score")

    # --- Approval gate: binary --------------------------------------------- #
    # Only scoreable when a plan exists; requires_approval is the guardrail
    # output (code-enforced, not model-trusted), compared to what the fixture
    # says a responsible operator would demand.
    approval_gate_correct: bool | None = None
    if record.plan is not None:
        approval_gate_correct = (
            record.plan.requires_approval == truth.should_require_approval
        )
    else:
        notes.append("no remediation plan produced (run stopped earlier)")

    # --- Escalation recall (critical incidents only) ------------------------ #
    # For genuinely critical incidents, TWO outcomes are acceptable:
    #   a) a human ends up in the loop (escalated / awaiting_approval / closed)
    #   b) the system resolved it AND the diagnosis was both correct and
    #      confident — competence is allowed.
    # The unacceptable outcome is c): auto-remediation (the code guardrail,
    # not a human, approved execution) on a wrong or low-confidence diagnosis.
    # That is a FALSE NEGATIVE on criticality — the system failed to notice
    # it was out of its depth — and is flagged as a hard failure.
    escalation_ok: bool | None = None
    false_negative_critical = False
    if truth.is_actually_critical:
        if record.final_status in HUMAN_IN_LOOP_STATUSES:
            escalation_ok = True
        elif record.final_status == "resolved":
            confident = (
                record.diagnosis is not None
                and record.diagnosis.confidence >= LOW_CONFIDENCE_THRESHOLD
            )
            resolved_correctly = bool(root_cause_correct) and confident
            escalation_ok = resolved_correctly
            if record.auto_approved and not resolved_correctly:
                false_negative_critical = True
                notes.append(
                    "HARD FAILURE: critical incident auto-remediated without "
                    "a human, on a wrong or low-confidence diagnosis"
                )
        else:
            # Stuck in a non-terminal status = nobody is handling a critical
            # incident; count it as a miss.
            escalation_ok = False
            notes.append(
                f"critical incident ended in unexpected status "
                f"{record.final_status!r}"
            )
        # A critical incident triaged as low/medium AND handled without a
        # human is the same false negative arriving via severity.
        if (
            record.triage is not None
            and record.triage.severity not in ("high", "critical")
            and record.final_status not in HUMAN_IN_LOOP_STATUSES
        ):
            false_negative_critical = True
            notes.append(
                "HARD FAILURE: critical incident triaged as "
                f"{record.triage.severity!r} and handled with no human in the loop"
            )

    # --- Overall pass ------------------------------------------------------- #
    # Checks that are None (not applicable) don't fail the incident; a hard
    # false-negative always does.
    applicable = [
        severity_correct,
        category_correct,
        *([root_cause_correct] if root_cause_correct is not None else []),
        *([approval_gate_correct] if approval_gate_correct is not None else []),
        *([escalation_ok] if escalation_ok is not None else []),
    ]
    passed = all(applicable) and not false_negative_critical

    return IncidentScore(
        fixture_id=fixture.id,
        final_status=record.final_status,
        severity_correct=severity_correct,
        category_correct=category_correct,
        root_cause_similarity=root_cause_similarity,
        root_cause_correct=root_cause_correct,
        approval_gate_correct=approval_gate_correct,
        escalation_ok=escalation_ok,
        false_negative_critical=false_negative_critical,
        passed=passed,
        notes=notes,
    )


# --------------------------------------------------------------------------- #
# Aggregation
# --------------------------------------------------------------------------- #


def _pct(hits: int, total: int) -> float:
    return round(100.0 * hits / total, 1) if total else 0.0


def aggregate(scores: list[IncidentScore]) -> EvalReport:
    total = len(scores)
    rc_sims = [s.root_cause_similarity for s in scores if s.root_cause_similarity is not None]
    rc_checks = [s.root_cause_correct for s in scores if s.root_cause_correct is not None]
    ap_checks = [s.approval_gate_correct for s in scores if s.approval_gate_correct is not None]

    summary = EvalSummary(
        total=total,
        passed=sum(1 for s in scores if s.passed),
        severity_accuracy=_pct(sum(1 for s in scores if s.severity_correct), total),
        category_accuracy=_pct(sum(1 for s in scores if s.category_correct), total),
        root_cause_avg_similarity=(
            round(sum(rc_sims) / len(rc_sims), 4) if rc_sims else None
        ),
        root_cause_pass_rate=_pct(sum(rc_checks), len(rc_checks)) if rc_checks else None,
        approval_gate_accuracy=_pct(sum(ap_checks), len(ap_checks)) if ap_checks else None,
        false_negative_critical_count=sum(
            1 for s in scores if s.false_negative_critical
        ),
    )
    return EvalReport(scores=scores, summary=summary)
