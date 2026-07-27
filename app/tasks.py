"""Celery tasks wrapping the Orchestrator.

This is what makes incident handling ASYNC relative to the API request that
created the incident: POST /incidents returns 202 immediately and a worker
picks the incident up from here. Same pattern as the memory-consolidation
task: each task run creates its own engine/session, because Celery workers
are separate processes and must not share the API's connection pool.

MCP note (Stage 6): the same reasoning applies to MCP client sessions — they
are bound to an event loop, and each task invocation is its own
asyncio.run(), so the worker connects to the MCP servers per run and closes
the sessions before the loop exits. With USE_MCP_TOOLS=False the
Orchestrator's Stage 5 default (in-process mock tools, scenario taken from
incident.raw_payload) is used unchanged.
"""

import asyncio
from collections.abc import Awaitable, Callable
from contextlib import AsyncExitStack
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.celery_app import celery_app
from app.orchestrator.core import Orchestrator


async def _make_orchestrator(stack: AsyncExitStack) -> Orchestrator:
    from app.core.config import get_settings
    from app.core.mcp_clients import MCPClientManager, registry_agent_factory

    if not get_settings().USE_MCP_TOOLS:
        return Orchestrator()  # Stage 5 default: in-process mock tools

    manager = MCPClientManager()
    stack.push_async_callback(manager.shutdown)
    registry = await manager.startup()  # degrades gracefully, never raises
    return Orchestrator(agent_factory=registry_agent_factory(registry))


def _run_with_own_session(
    fn: Callable[[Orchestrator, AsyncSession], Awaitable[None]],
) -> None:
    async def _run() -> None:
        from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

        from app.core.config import get_settings

        engine = create_async_engine(get_settings().DATABASE_URL)
        factory = async_sessionmaker(engine, expire_on_commit=False)
        async with AsyncExitStack() as stack:
            stack.push_async_callback(engine.dispose)
            orchestrator = await _make_orchestrator(stack)
            async with factory() as session:
                await fn(orchestrator, session)
                await session.commit()

    asyncio.run(_run())


@celery_app.task(name="orchestrator.run_incident")
def orchestrate_incident_task(incident_id: str) -> None:
    _run_with_own_session(
        lambda orch, session: orch.run_incident(UUID(incident_id), session)
    )


@celery_app.task(name="orchestrator.resume_incident")
def resume_incident_task(incident_id: str, approved: bool, resolved_by: str) -> None:
    _run_with_own_session(
        lambda orch, session: orch.resume_after_approval(
            UUID(incident_id), approved, resolved_by, session
        )
    )
