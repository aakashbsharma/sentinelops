"""Eval harness runner — a DELIBERATE, MANUAL command.

    python -m eval.run_eval [--fixture ID] [--concurrency N] [--threshold X]

This drives the REAL Orchestrator with REAL Anthropic API calls (roughly
10-60k tokens per fixture depending on how chatty the agents get). It is
intentionally not wired into pytest or CI — running it is a decision, not a
side effect.

DB CHOICE (and why): a DEDICATED eval database (EVAL_DATABASE_URL, default =
DATABASE_URL with an `_eval` db name), with all tables dropped and recreated
fresh at the start of every run. Two reasons over reusing the dev DB or a
schema within it:
  1. eval incidents must not pollute dev data (or vice versa);
  2. more subtly, MemoryManager.write_outcome persists episodic memories for
     resolved incidents — on a shared/persistent DB, run N's answers would be
     retrieved as "similar past incidents" by run N+1's agents, silently
     inflating scores exactly as you iterate on prompts. Fresh tables per run
     keep every run cold and comparable.

Tools: the Orchestrator's default agent factory registers the IN-PROCESS mock
tools, seeded from each fixture's raw_payload["scenario"] — the same
deterministic synthetic world the MCP servers serve. No MCP servers, Redis,
or Celery needed; the only external dependencies are Postgres and the
Anthropic API.
"""

import argparse
import asyncio
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import UUID

import structlog
import yaml
from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncEngine, async_sessionmaker, create_async_engine

from app.agents.payloads import (
    DiagnosisPayload,
    RemediationPlanPayload,
    TriagePayload,
)
from app.core.config import get_settings
from app.core.database import Base
from app.core.logging import configure_logging
from app.models.agent_message import AgentMessage
from app.models.approval import ApprovalRequest
from app.models.incident import Incident
from app.observability.cost import compute_cost_summary
from app.orchestrator.core import Orchestrator
from eval.scorer import (
    AUTO_APPROVAL_MARKER,
    ROOT_CAUSE_SIMILARITY_THRESHOLD,
    EvalReport,
    Fixture,
    RunRecord,
    aggregate,
    score_incident,
)

import app.models  # noqa: F401  — register every table on Base.metadata

logger = structlog.get_logger(__name__)

FIXTURES_DIR = Path(__file__).parent / "fixtures"
RESULTS_DIR = Path(__file__).parent / "results"


# --------------------------------------------------------------------------- #
# Eval database
# --------------------------------------------------------------------------- #


def eval_database_url() -> str:
    url = os.environ.get("EVAL_DATABASE_URL")
    if url is None:
        # Default: same server as DATABASE_URL, database name + "_eval".
        base = get_settings().DATABASE_URL
        server, _, dbname = base.rpartition("/")
        url = f"{server}/{dbname.split('?')[0]}_eval"
    if not url.startswith("postgresql+asyncpg://"):
        raise SystemExit(
            "EVAL_DATABASE_URL must start with 'postgresql+asyncpg://'"
        )
    return url


async def _ensure_database_exists(url: str) -> None:
    """CREATE DATABASE if the eval db is missing (needs an admin connection
    to the default 'postgres' database — CREATE DATABASE can't run inside
    the target)."""
    import asyncpg

    server, _, dbname = url.removeprefix("postgresql+asyncpg://").rpartition("/")
    try:
        conn = await asyncpg.connect(dsn=f"postgresql://{server}/{dbname}")
        await conn.close()
        return  # already exists
    except asyncpg.InvalidCatalogNameError:
        pass
    admin = await asyncpg.connect(dsn=f"postgresql://{server}/postgres")
    try:
        await admin.execute(f'CREATE DATABASE "{dbname}"')
        print(f"created eval database {dbname!r}")
    finally:
        await admin.close()


async def make_eval_engine() -> AsyncEngine:
    url = eval_database_url()
    await _ensure_database_exists(url)
    engine = create_async_engine(url)
    async with engine.begin() as conn:
        # pgvector before create_all: memory_records has a vector column.
        await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        # Fresh tables every run — see module docstring (memory contamination).
        await conn.run_sync(Base.metadata.drop_all)
        await conn.run_sync(Base.metadata.create_all)
    return engine


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #


def load_fixtures(only: str | None = None) -> list[Fixture]:
    fixtures = []
    for path in sorted(FIXTURES_DIR.glob("*.yaml")):
        with open(path, encoding="utf-8") as f:
            fixtures.append(Fixture(**yaml.safe_load(f)))
    if only is not None:
        fixtures = [f for f in fixtures if f.id == only]
        if not fixtures:
            raise SystemExit(f"no fixture with id {only!r} in {FIXTURES_DIR}")
    return fixtures


# --------------------------------------------------------------------------- #
# Running one fixture through the real orchestrator
# --------------------------------------------------------------------------- #


def _latest_payload(
    messages: list[AgentMessage], role: str, model_cls: type
) -> Any | None:
    """Rebuild the newest valid typed payload a role produced (same
    reconstruction the orchestrator uses: confidence/needs_more_data live in
    first-class columns, the rest in content['payload'])."""
    for message in reversed(messages):
        if message.role != role:
            continue
        data = {
            **(message.content.get("payload") or {}),
            "confidence": message.confidence,
            "needs_more_data": message.needs_more_data,
        }
        try:
            return model_cls(**data)
        except Exception:  # noqa: BLE001 — invalid attempts are expected
            continue
    return None


async def run_fixture(
    fixture: Fixture,
    session_factory: async_sessionmaker,
    semaphore: asyncio.Semaphore,
) -> RunRecord:
    async with semaphore:
        print(f"[{fixture.id}] running ...", flush=True)
        async with session_factory() as session:
            incident = Incident(
                title=fixture.title,
                description=fixture.description,
                source=fixture.source,
                severity="unknown",
                status="open",
                # Ground truth deliberately NOT copied anywhere near here —
                # the agents see title/description/raw_payload only.
                raw_payload=fixture.raw_payload,
            )
            session.add(incident)
            await session.commit()
            incident_id: UUID = incident.id

            # Fresh Orchestrator per fixture; the default agent factory seeds
            # mock tools from raw_payload["scenario"]. run_incident never
            # raises for agent-level failures (it escalates), so one broken
            # fixture can't take down the whole eval.
            orchestrator = Orchestrator()
            await orchestrator.run_incident(incident_id, session)
            await session.commit()

        # Read back everything the run persisted, on a clean session.
        async with session_factory() as session:
            incident = await session.get(Incident, incident_id)
            messages = (
                (
                    await session.execute(
                        select(AgentMessage)
                        .where(AgentMessage.incident_id == incident_id)
                        .order_by(AgentMessage.created_at)
                    )
                )
                .scalars()
                .all()
            )
            approvals = (
                (
                    await session.execute(
                        select(ApprovalRequest).where(
                            ApprovalRequest.incident_id == incident_id
                        )
                    )
                )
                .scalars()
                .all()
            )

        cost = compute_cost_summary(messages)
        record = RunRecord(
            fixture_id=fixture.id,
            incident_id=str(incident_id),
            final_status=incident.status,
            triage=_latest_payload(messages, "triage", TriagePayload),
            diagnosis=_latest_payload(messages, "diagnostician", DiagnosisPayload),
            plan=_latest_payload(
                messages, "remediation_planner", RemediationPlanPayload
            ),
            auto_approved=any(
                a.status == "approved" and a.resolved_by == AUTO_APPROVAL_MARKER
                for a in approvals
            ),
            total_tokens=cost.total_tokens,
            estimated_cost_usd=cost.estimated_cost_usd,
            run_id=next(
                (m.run_id for m in messages if m.run_id is not None), None
            ),
        )
        print(
            f"[{fixture.id}] done: status={record.final_status} "
            f"tokens={record.total_tokens} (~${record.estimated_cost_usd})",
            flush=True,
        )
        return record


# --------------------------------------------------------------------------- #
# Report formatting (plain strings — deliberately unfancy)
# --------------------------------------------------------------------------- #


def _mark(value: bool | None) -> str:
    if value is None:
        return "-"
    return "PASS" if value else "FAIL"


def format_report(report: EvalReport, records: list[RunRecord]) -> str:
    by_id = {r.fixture_id: r for r in records}
    header = (
        f"{'fixture':<34} {'status':<18} {'sev':<5} {'cat':<5} "
        f"{'rc-sim':<7} {'appr':<5} {'esc':<5} {'FNcrit':<7} {'result':<7}"
    )
    lines = [header, "-" * len(header)]
    for s in report.scores:
        sim = f"{s.root_cause_similarity:.2f}" if s.root_cause_similarity is not None else "-"
        lines.append(
            f"{s.fixture_id:<34} {s.final_status:<18} "
            f"{_mark(s.severity_correct):<5} {_mark(s.category_correct):<5} "
            f"{sim:<7} {_mark(s.approval_gate_correct):<5} "
            f"{_mark(s.escalation_ok):<5} "
            f"{'YES!!' if s.false_negative_critical else 'no':<7} "
            f"{'PASS' if s.passed else 'FAIL':<7}"
        )
        for note in s.notes:
            lines.append(f"    note: {note}")

    m = report.summary
    total_cost = round(sum(r.estimated_cost_usd for r in records), 4)
    total_tokens = sum(r.total_tokens for r in records)
    rc_avg = (
        f"{m.root_cause_avg_similarity:.4f}"
        if m.root_cause_avg_similarity is not None
        else "n/a"
    )
    rc_rate = (
        f"{m.root_cause_pass_rate}%" if m.root_cause_pass_rate is not None else "n/a"
    )
    ap_acc = (
        f"{m.approval_gate_accuracy}%"
        if m.approval_gate_accuracy is not None
        else "n/a"
    )
    lines += [
        "",
        "SUMMARY",
        f"  incidents:                {m.passed}/{m.total} passed",
        f"  severity accuracy:        {m.severity_accuracy}%",
        f"  category accuracy:        {m.category_accuracy}%",
        f"  root-cause similarity:    avg {rc_avg}  (pass rate {rc_rate} at "
        f"threshold {ROOT_CAUSE_SIMILARITY_THRESHOLD})",
        f"  approval-gate accuracy:   {ap_acc}",
        f"  tokens / est. cost:       {total_tokens} tokens (~${total_cost}, "
        "illustrative pricing)",
        "",
        "  ZERO-TOLERANCE CHECK — false-negative criticals: "
        f"{m.false_negative_critical_count}"
        + ("  (OK)" if m.false_negative_critical_count == 0 else "  ** FAILURE **"),
    ]
    if m.false_negative_critical_count > 0:
        lines.append(
            "  A critical incident was auto-remediated without a human on a "
            "wrong/low-confidence diagnosis. This is reported separately "
            "because no aggregate accuracy excuses it."
        )
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #


async def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the SentinelOps eval harness (REAL LLM calls, costs money)."
    )
    parser.add_argument("--fixture", help="run only the fixture with this id")
    parser.add_argument(
        "--concurrency",
        type=int,
        default=3,
        help="fixtures run in parallel (default 3; be kind to rate limits)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=ROOT_CAUSE_SIMILARITY_THRESHOLD,
        help="root-cause cosine similarity pass threshold",
    )
    args = parser.parse_args()

    configure_logging()
    if not get_settings().ANTHROPIC_API_KEY:
        print(
            "ANTHROPIC_API_KEY is not set — the eval drives real agents and "
            "cannot run without it.",
            file=sys.stderr,
        )
        return 2

    fixtures = load_fixtures(only=args.fixture)
    print(
        f"Running {len(fixtures)} fixture(s) through the REAL orchestrator.\n"
        "This makes real Anthropic API calls (ballpark: a few dollars for the "
        "full set at illustrative pricing). Ctrl+C now if that's not intended.\n"
    )

    engine = await make_eval_engine()
    try:
        session_factory = async_sessionmaker(engine, expire_on_commit=False)
        semaphore = asyncio.Semaphore(args.concurrency)
        records = await asyncio.gather(
            *(run_fixture(f, session_factory, semaphore) for f in fixtures)
        )
    finally:
        await engine.dispose()

    # Scoring: real LocalEmbeddingProvider — the same embedding space the
    # production memory system uses (and it's local, so scoring is free).
    from app.memory.embeddings import LocalEmbeddingProvider

    provider = LocalEmbeddingProvider()
    scores = [
        await score_incident(f, r, provider, threshold=args.threshold)
        for f, r in zip(fixtures, records)
    ]
    report = aggregate(scores)

    # Raw results to eval/results/{timestamp}.json so runs stay comparable
    # while iterating on prompts.
    RESULTS_DIR.mkdir(exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = RESULTS_DIR / f"{stamp}.json"
    out_path.write_text(
        json.dumps(
            {
                "timestamp": stamp,
                "threshold": args.threshold,
                "fixture_count": len(fixtures),
                "records": [r.model_dump(mode="json") for r in records],
                "scores": [s.model_dump(mode="json") for s in report.scores],
                "summary": report.summary.model_dump(mode="json"),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print(format_report(report, list(records)))
    print(f"\nraw results written to {out_path}")
    return 0 if report.summary.false_negative_critical_count == 0 else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
