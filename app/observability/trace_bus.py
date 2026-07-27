"""Live trace bus: agent steps -> Redis pub/sub -> WebSocket clients.

Why a bus at all: agents run inside CELERY WORKER processes, WebSocket
clients connect to the FASTAPI process. Redis pub/sub (already deployed as
the Celery broker) is the bridge — the worker publishes each completed
AgentStep to a channel keyed by incident_id, and the WebSocket handler
(app/api/routes/ws_trace.py) subscribes and forwards.

The publisher is deliberately fire-and-forget: live streaming is a nicety,
incident handling is the job. A down/slow Redis must degrade to "no live
panel", never to a failed agent loop — every error here is swallowed
(logged at debug) and short socket timeouts cap the worst-case stall.
"""

import json
import time

import structlog

from app.agents.schemas import AgentStep
from app.core.tracing import current_incident_id

logger = structlog.get_logger(__name__)

_CHANNEL_PREFIX = "sentinelops:trace:"

# Circuit breaker: with Redis down, every connect attempt eats the full
# socket_connect_timeout (measured ~1s on Windows — no fast refusal), which
# would add a second to EVERY agent step. One failure trips the breaker and
# publishing goes silent for a cooldown instead of stalling the loop.
_BREAKER_COOLDOWN_S = 30.0
_breaker_open_until: float = 0.0


def trace_channel(incident_id: str) -> str:
    return f"{_CHANNEL_PREFIX}{incident_id}"


def step_payload(agent_name: str, step: AgentStep) -> dict:
    """The wire shape Stage 9's React trace panel consumes.

    The same shape is produced by the WebSocket handler when it replays
    history from persisted AgentMessages — clients never see two formats.
    """
    return {
        "step_number": step.step_number,
        "agent_name": agent_name,
        "thought": step.thought,
        "action": step.action.model_dump(mode="json") if step.action else None,
        "observation": step.observation,
        "reflection": step.reflection,
        "timestamp": step.timestamp.isoformat(),
        "run_id": step.run_id,
    }


async def publish_step(agent_name: str, step: AgentStep) -> None:
    """Publish one completed AgentStep to the incident's trace channel.

    No-op outside an incident run (incident_id contextvar unset — e.g. unit
    tests driving an agent directly). Never raises.
    """
    global _breaker_open_until

    incident_id = current_incident_id()
    if incident_id is None:
        return
    if time.monotonic() < _breaker_open_until:
        return

    try:
        import redis.asyncio as aioredis

        from app.core.config import get_settings

        # Connect-per-publish, not a cached client: Celery tasks each run in
        # their own asyncio.run() loop, so a module-level client would be
        # bound to a dead loop on the next task. At a handful of steps per
        # incident, connection setup cost is irrelevant.
        client = aioredis.from_url(
            get_settings().REDIS_URL,
            socket_connect_timeout=1.0,
            socket_timeout=1.0,
        )
        try:
            await client.publish(
                trace_channel(incident_id),
                json.dumps(step_payload(agent_name, step), default=str),
            )
        finally:
            await client.aclose()
    except Exception:  # noqa: BLE001 — streaming must never break a run
        _breaker_open_until = time.monotonic() + _BREAKER_COOLDOWN_S
        logger.debug(
            "trace_publish_failed",
            incident_id=incident_id,
            step_number=step.step_number,
            breaker_cooldown_s=_BREAKER_COOLDOWN_S,
        )
