"""MCP client lifecycle: connect at startup, discover tools, register adapters.

Used from two places, because agents run in two kinds of process:
  - FastAPI lifespan (app/main.py): connect on startup, close on shutdown.
  - Celery workers (app/tasks.py): a fresh connection per orchestration run —
    MCP sessions are bound to an event loop, and each Celery task gets its
    own asyncio.run(), so API-process sessions can't be shared with workers.

GRACEFUL DEGRADATION: if a given MCP server is unreachable at connect time,
we log an error and register an UnavailableTool stub for each of its tool
names instead of crashing. Why this matters: the tool servers are
dependencies of AGENT REASONING, not of incident INTAKE. If runbook-mcp is
down, agents should still triage and diagnose (observing "search_runbooks
temporarily unavailable" and routing around it, exactly like any other
failed ToolResult) — one dead sidecar must not take down the entire
incident-response front door.

Stub names come from MCP_SERVERS below. That's a deliberate, small
duplication: when a server is down we cannot list_tools() it, so the client
must know what SHOULD have been there in order to degrade per-tool. When the
server is up, its list_tools() is authoritative and this map is ignored.
"""

import asyncio
from contextlib import AsyncExitStack
from typing import Any

import httpx
import structlog

from app.agents.tools.base import ToolRegistry
from app.agents.tools.mcp_adapter import MCPToolAdapter, UnavailableTool
from app.core.config import get_settings

logger = structlog.get_logger(__name__)

# server key -> (settings attribute holding its URL, tool names it serves).
MCP_SERVERS: dict[str, tuple[str, list[str]]] = {
    "logs-metrics-mcp": ("LOGS_METRICS_MCP_URL", ["query_logs", "query_metrics"]),
    "infra-mcp": ("INFRA_MCP_URL", ["restart_service", "scale_deployment"]),
    "runbook-mcp": ("RUNBOOK_MCP_URL", ["search_runbooks"]),
}


class _MCPServerConnection:
    """One server's transport + session, owned by a dedicated asyncio task.

    WHY A DEDICATED TASK: the MCP streamable-HTTP transport is built on anyio
    task groups, whose cancel scopes MUST be entered and exited in the same
    task, in strict LIFO order. Holding the transport open across function
    boundaries in a shared AsyncExitStack (enter in startup(), exit in
    shutdown()) violates that and dies with "attempted to exit cancel scope
    in a different task". So each connection runs its entire context-manager
    chain inside one task that then parks on a close event; connect() and
    aclose() just signal it. The ClientSession itself is safe to use from
    other tasks — it communicates over memory streams.
    """

    def __init__(self, server_name: str, url: str, connect_timeout_s: float) -> None:
        self.server_name = server_name
        self.url = url
        self.connect_timeout_s = connect_timeout_s
        self.session: Any = None
        self.tools: list[Any] = []
        self._ready = asyncio.Event()
        self._close = asyncio.Event()
        self._error: BaseException | None = None
        self._task: asyncio.Task | None = None

    async def connect(self) -> list[Any]:
        """Start the connection task; return the discovered tools or raise."""
        self._task = asyncio.create_task(
            self._run(), name=f"mcp-connection-{self.server_name}"
        )
        try:
            # The httpx connect timeout should fire first; this outer wait is
            # the belt-and-braces bound on the whole handshake.
            await asyncio.wait_for(
                self._ready.wait(), timeout=self.connect_timeout_s + 5.0
            )
        except TimeoutError:
            await self.aclose()
            raise TimeoutError(
                f"MCP handshake with {self.url} exceeded "
                f"{self.connect_timeout_s + 5.0:.0f}s"
            ) from None
        if self._error is not None:
            await self.aclose()
            raise self._error
        return self.tools

    async def _run(self) -> None:
        try:
            from mcp import ClientSession
            from mcp.client.streamable_http import streamable_http_client

            async with AsyncExitStack() as stack:
                # async with AsyncExitStack() as stack:
                read, write, _ = await stack.enter_async_context(
                    streamable_http_client(self.url)
                )
                session = await stack.enter_async_context(ClientSession(read, write))
                await session.initialize()
                self.tools = (await session.list_tools()).tools
                self.session = session
                self._ready.set()
                # Park here, holding every context open, until shutdown.
                await self._close.wait()
        except asyncio.CancelledError:
            raise
        except BaseException as exc:  # noqa: BLE001 — includes anyio ExceptionGroups
            self._error = exc
            self._ready.set()

    async def aclose(self) -> None:
        if self._task is None or self._task.done():
            return
        self._close.set()
        try:
            # wait_for cancels the task if it doesn't unwind in time; either
            # way the scopes exit inside their own task, which is the point.
            await asyncio.wait_for(self._task, timeout=10.0)
        except (TimeoutError, asyncio.CancelledError):
            pass
        except Exception:  # noqa: BLE001 — shutdown must not propagate
            logger.warning("mcp_connection_close_failed", server=self.server_name)


class MCPClientManager:
    """Owns the MCP client connections and the ToolRegistry built from them."""

    def __init__(self) -> None:
        self._connections: list[_MCPServerConnection] = []
        self.registry = ToolRegistry()
        self.degraded_servers: list[str] = []

    async def startup(self) -> ToolRegistry:
        """Build the registry: MCP adapters when USE_MCP_TOOLS, in-process
        mocks otherwise. Never raises for an unreachable server."""
        settings = get_settings()
        if not settings.USE_MCP_TOOLS:
            self._register_mock_tools()
            logger.info("tool_registry_ready", mode="in-process mocks")
            return self.registry

        for server_name, (url_attr, tool_names) in MCP_SERVERS.items():
            url = getattr(settings, url_attr)
            try:
                await self._connect_and_register(server_name, url)
            except Exception as exc:  # noqa: BLE001 — degrade, never crash startup
                logger.error(
                    "mcp_server_unreachable",
                    server=server_name,
                    url=url,
                    error=f"{type(exc).__name__}: {exc}",
                )
                self.degraded_servers.append(server_name)
                for tool_name in tool_names:
                    self.registry.register(UnavailableTool(tool_name, server_name))
        logger.info(
            "tool_registry_ready",
            mode="mcp",
            tools=[t["name"] for t in self.registry.list_tools()],
            degraded=self.degraded_servers,
        )
        return self.registry

    async def shutdown(self) -> None:
        """Close every client session/transport cleanly (reverse order)."""
        for connection in reversed(self._connections):
            await connection.aclose()
        self._connections.clear()

    # ------------------------------------------------------------------ #

    async def _connect_and_register(self, server_name: str, url: str) -> None:
        settings = get_settings()
        connection = _MCPServerConnection(
            server_name, url, settings.MCP_CONNECT_TIMEOUT_S
        )
        tools = await connection.connect()  # raises on failure (self-cleans)
        self._connections.append(connection)

        # Register every discovered tool under the name the SERVER declares —
        # the same names Stage 4's mocks used, which is the whole contract.
        # Schemas/descriptions come from list_tools(), never hardcoded twice.
        for tool in tools:
            self.registry.register(
                MCPToolAdapter(
                    connection.session,
                    name=tool.name,
                    description=tool.description,
                    input_json_schema=tool.inputSchema,
                )
            )
        logger.info(
            "mcp_server_connected",
            server=server_name,
            url=url,
            tools=[t.name for t in tools],
        )

    def _register_mock_tools(self, scenario: str | None = None) -> None:
        from app.agents.tools.mock_infra import (
            QueryLogsTool,
            QueryMetricsTool,
            RestartServiceTool,
            ScaleDeploymentTool,
        )

        self.registry.register(QueryLogsTool(scenario=scenario))
        self.registry.register(QueryMetricsTool(scenario=scenario))
        self.registry.register(RestartServiceTool())
        self.registry.register(ScaleDeploymentTool())


def registry_agent_factory(registry: ToolRegistry) -> Any:
    """An Orchestrator agent_factory whose agents all share one (MCP-backed)
    ToolRegistry. Mirrors the Stage 5 default factory's LoopConfig wiring;
    only the tool source differs — which is exactly the Stage 2 promise."""
    from app.agents.registry import get_agent_class
    from app.agents.schemas import LoopConfig
    from app.models.incident import Incident

    def factory(role: str, incident: Incident) -> Any:
        settings = get_settings()
        config = LoopConfig(
            max_iterations=settings.MAX_AGENT_ITERATIONS,
            token_budget=settings.AGENT_TOKEN_BUDGET,
        )
        return get_agent_class(role)(config, registry)

    return factory
