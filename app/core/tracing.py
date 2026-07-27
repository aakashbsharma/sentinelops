"""Trace correlation for incident runs — the Stage 7 observability spine.

THE DESIGN DECISION: correlation ids (incident_id + run_id) propagate via
`contextvars.ContextVar`, not function parameters. `run_incident()` sets them
once at the top of a run; every nested `await` — orchestrator -> agent loop ->
tool registry -> Redis publisher — inherits them automatically, because
asyncio copies the ambient `Context` into every task and callback. Nothing in
the agent/tool call chain needed a signature change to become traceable.

Why a run_id AND an incident_id: an incident is not a run. An incident that
escalates and is later resumed via `resume_after_approval()` has TWO runs, and
"what did the resume do?" is a different question from "what happened to this
incident overall?". Every AgentStep, AgentMessage, and log line carries both,
so the trace panel and the eval harness can reconstruct either view.

Log lines get the ids for free: `incident_run()` also binds them into
structlog's contextvars, and the `merge_contextvars` processor (configured in
app/core/logging.py) merges them into every event dict emitted within the run.
"""

import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any

import structlog.contextvars

# Defaults are None, not "": "no active run" must be distinguishable from a
# run — AgentStep/AgentMessage stamping treats None as "leave unstamped".
incident_id_var: ContextVar[str | None] = ContextVar("incident_id", default=None)
run_id_var: ContextVar[str | None] = ContextVar("run_id", default=None)


def current_incident_id() -> str | None:
    return incident_id_var.get()


def current_run_id() -> str | None:
    return run_id_var.get()


@contextmanager
def incident_run(incident_id: Any, run_id: str | None = None) -> Iterator[str]:
    """Establish the correlation context for one orchestrator run.

    One run_id per call — `run_incident()` and `resume_after_approval()` each
    open their own, so a resumed incident's second act is distinguishable from
    its first. Exit always restores the previous values (token-based reset),
    so nested/sequential runs in one process can't bleed ids into each other.
    """
    rid = run_id or str(uuid.uuid4())
    iid = str(incident_id)

    incident_token = incident_id_var.set(iid)
    run_token = run_id_var.set(rid)
    # Same ids into structlog's own contextvars so merge_contextvars stamps
    # every log line in the run without any logger.bind() plumbing.
    structlog_tokens = structlog.contextvars.bind_contextvars(
        incident_id=iid, run_id=rid
    )
    try:
        yield rid
    finally:
        structlog.contextvars.reset_contextvars(**structlog_tokens)
        incident_id_var.reset(incident_token)
        run_id_var.reset(run_token)


# --------------------------------------------------------------------------- #
# OpenTelemetry spans (minimal, optional)
# --------------------------------------------------------------------------- #
# Console exporter, not OTLP: zero extra infrastructure to demo spans. Behind
# OTEL_ENABLED (default False) so tests and normal dev output stay clean.
# Swapping to a real backend later = replace the exporter in _get_tracer().

_tracer: Any = None


def _get_tracer() -> Any:
    global _tracer
    if _tracer is None:
        from opentelemetry import trace
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import (
            ConsoleSpanExporter,
            SimpleSpanProcessor,
        )

        provider = TracerProvider()
        # SimpleSpanProcessor (synchronous export) over BatchSpanProcessor:
        # console output should appear immediately, and span volume here is
        # a handful per incident — batching buys nothing.
        provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))
        trace.set_tracer_provider(provider)
        _tracer = trace.get_tracer("sentinelops")
    return _tracer


@contextmanager
def span(name: str, **attributes: Any) -> Iterator[Any]:
    """Open an OTel span carrying the current correlation ids as attributes.

    No-op (yields None) when OTEL_ENABLED is off, so call sites never need
    their own feature check.
    """
    from app.core.config import get_settings

    if not get_settings().OTEL_ENABLED:
        yield None
        return

    attrs = {k: v for k, v in attributes.items() if v is not None}
    if (iid := current_incident_id()) is not None:
        attrs["incident_id"] = iid
    if (rid := current_run_id()) is not None:
        attrs["run_id"] = rid

    with _get_tracer().start_as_current_span(name, attributes=attrs) as otel_span:
        yield otel_span
