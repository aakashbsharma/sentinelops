"""WebSocket live trace: /ws/incidents/{id}/trace.

Protocol (what Stage 9's React panel consumes):
  1. On connect, the full existing step history is replayed from the
     persisted AgentMessage rows, oldest first.
  2. Then new AgentSteps stream live as agents produce them — the Celery
     worker publishes each completed step to Redis pub/sub
     (app/observability/trace_bus.py) and this handler forwards it.

Every frame is one JSON object of the same shape, history or live:
  {step_number, agent_name, thought, action, observation, reflection,
   timestamp, run_id}
"""

import asyncio
from uuid import UUID

import structlog
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from sqlalchemy import select

from app.core.config import get_settings
from app.core.database import async_session_factory
from app.models.agent_message import AgentMessage
from app.observability.trace_bus import trace_channel

logger = structlog.get_logger(__name__)

router = APIRouter(tags=["trace"])


async def _load_history(incident_id: UUID) -> list[dict]:
    """Flatten persisted AgentMessage.content["steps"] into wire frames.

    A dedicated short-lived session: the WebSocket then stays open for
    minutes, and holding a pooled DB connection hostage that whole time
    would starve the API under a handful of concurrent panels.
    """
    async with async_session_factory() as db:
        messages = (
            (
                await db.execute(
                    select(AgentMessage)
                    .where(AgentMessage.incident_id == incident_id)
                    .order_by(AgentMessage.created_at)
                )
            )
            .scalars()
            .all()
        )

    frames: list[dict] = []
    for message in messages:
        # Orchestrator decision messages carry no steps; .get() skips them.
        for step in (message.content or {}).get("steps", []):
            frames.append(
                {
                    "step_number": step.get("step_number"),
                    "agent_name": message.agent_name,
                    "thought": step.get("thought"),
                    "action": step.get("action"),
                    "observation": step.get("observation"),
                    "reflection": step.get("reflection"),
                    "timestamp": step.get("timestamp"),
                    # Steps persisted before Stage 7 lack a stamped run_id;
                    # fall back to the message's column.
                    "run_id": step.get("run_id") or message.run_id,
                }
            )
    return frames


async def _forward_live(websocket: WebSocket, incident_id: UUID) -> None:
    """Subscribe to the incident's Redis channel and forward every frame."""
    import redis.asyncio as aioredis

    client = aioredis.from_url(get_settings().REDIS_URL, decode_responses=True)
    pubsub = client.pubsub()
    try:
        await pubsub.subscribe(trace_channel(str(incident_id)))
        async for message in pubsub.listen():
            if message["type"] == "message":
                # Payload is already the JSON wire frame (trace_bus builds it).
                await websocket.send_text(message["data"])
    finally:
        await pubsub.aclose()
        await client.aclose()


@router.websocket("/ws/incidents/{incident_id}/trace")
async def incident_trace(websocket: WebSocket, incident_id: UUID) -> None:
    # Minimal lifecycle (per spec: one connection per client, no fan-out
    # manager needed — Redis pub/sub already handles multi-subscriber).
    await websocket.accept()
    log = logger.bind(incident_id=str(incident_id))
    log.info("trace_ws_connected")

    for frame in await _load_history(incident_id):
        await websocket.send_json(frame)

    # Two concurrent jobs: forwarding Redis -> client, and reading from the
    # client purely to notice disconnects (receive() raising is how FastAPI
    # signals them). Whichever ends first tears the other down.
    forwarder = asyncio.create_task(_forward_live(websocket, incident_id))
    try:
        while True:
            await websocket.receive_text()  # ignore inbound content
    except WebSocketDisconnect:
        log.info("trace_ws_disconnected")
    except Exception:  # noqa: BLE001 — e.g. Redis down mid-stream
        log.warning("trace_ws_error", exc_info=True)
    finally:
        forwarder.cancel()
        try:
            await forwarder
        except (asyncio.CancelledError, Exception):  # noqa: BLE001
            pass
