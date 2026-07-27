"""MCPToolAdapter + graceful degradation — unit tests, no real MCP server.

The load-bearing claim verified here: MCP failures flow through Stage 2's
UNCHANGED ToolRegistry.execute() error path — typed exception in, failed
ToolResult out, nothing crashes. That's what let Stage 6 swap the tool
backend without touching a single agent.
"""

from types import SimpleNamespace
from typing import Any

import pytest

from app.agents.schemas import ToolCall
from app.agents.tools.base import ToolRegistry
from app.agents.tools.exceptions import (
    MCPConnectionError,
    MCPToolInvocationError,
    MCPToolUnavailableError,
)
from app.agents.tools.mcp_adapter import (
    MCPToolAdapter,
    UnavailableTool,
    model_from_json_schema,
)
from app.core.config import get_settings
from app.core.mcp_clients import MCP_SERVERS, MCPClientManager

QUERY_LOGS_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "service_name": {"type": "string"},
        "time_range": {"type": "string", "default": "15m"},
    },
    "required": ["service_name"],
}


def result_with(structured: Any = None, text: str | None = None, is_error: bool = False):
    content = (
        [SimpleNamespace(type="text", text=text)] if text is not None else []
    )
    return SimpleNamespace(
        structuredContent=structured, content=content, isError=is_error
    )


class FakeSession:
    """Stands in for mcp.ClientSession — records calls, returns a scripted
    result or raises a scripted exception."""

    def __init__(self, result: Any = None, exc: Exception | None = None) -> None:
        self.result = result
        self.exc = exc
        self.calls: list[tuple[str, dict[str, Any]]] = []

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> Any:
        self.calls.append((name, arguments))
        if self.exc is not None:
            raise self.exc
        return self.result


def make_adapter(session: FakeSession) -> MCPToolAdapter:
    return MCPToolAdapter(
        session,
        name="query_logs",
        description="Query recent application logs for a service.",
        input_json_schema=QUERY_LOGS_SCHEMA,
    )


# ---------------------------------------------------------------------- #
# Result parsing
# ---------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_adapter_returns_structured_content_as_plain_dict() -> None:
    payload = {"service": "api", "lines": ["INFO ok"]}
    session = FakeSession(result=result_with(structured=payload))
    adapter = make_adapter(session)

    output = await adapter.run(service_name="api", time_range="15m")

    assert output == payload  # the same plain dict the Stage 4 mocks returned
    assert session.calls == [
        ("query_logs", {"service_name": "api", "time_range": "15m"})
    ]


@pytest.mark.asyncio
async def test_adapter_unwraps_fastmcp_scalar_result_envelope() -> None:
    # FastMCP wraps non-object return values as {"result": value}.
    session = FakeSession(result=result_with(structured={"result": 42}))
    assert await make_adapter(session).run(service_name="api") == 42


@pytest.mark.asyncio
async def test_adapter_falls_back_to_json_text_content() -> None:
    session = FakeSession(
        result=result_with(structured=None, text='{"service": "api"}')
    )
    assert await make_adapter(session).run(service_name="api") == {"service": "api"}


# ---------------------------------------------------------------------- #
# Error contract through the UNCHANGED Stage 2 registry
# ---------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_tool_side_error_raises_typed_exception() -> None:
    session = FakeSession(result=result_with(text="boom", is_error=True))
    with pytest.raises(MCPToolInvocationError, match="boom"):
        await make_adapter(session).run(service_name="api")


@pytest.mark.asyncio
async def test_connection_error_raises_typed_exception() -> None:
    session = FakeSession(exc=ConnectionError("server went away"))
    with pytest.raises(MCPConnectionError, match="server went away"):
        await make_adapter(session).run(service_name="api")


@pytest.mark.asyncio
async def test_registry_converts_mcp_failure_to_failed_tool_result() -> None:
    """The Stage 2 verification: ToolRegistry.execute() (untouched since
    Stage 2) turns the adapter's typed exception into a failed ToolResult
    the agent loop observes — no crash, no special-casing for MCP."""
    registry = ToolRegistry()
    registry.register(make_adapter(FakeSession(exc=ConnectionError("refused"))))

    result = await registry.execute(
        ToolCall(tool_name="query_logs", arguments={"service_name": "api"})
    )

    assert result.success is False
    assert "MCPConnectionError" in (result.error or "")
    assert "refused" in (result.error or "")


@pytest.mark.asyncio
async def test_registry_validates_arguments_before_any_network_call() -> None:
    session = FakeSession(result=result_with(structured={}))
    registry = ToolRegistry()
    registry.register(make_adapter(session))

    result = await registry.execute(
        ToolCall(tool_name="query_logs", arguments={})  # missing service_name
    )

    assert result.success is False
    assert "Invalid arguments" in (result.error or "")
    assert session.calls == []  # the bad call never crossed the network


def test_model_from_json_schema_reconstructs_defaults() -> None:
    model = model_from_json_schema("X", QUERY_LOGS_SCHEMA)
    parsed = model(service_name="api")
    assert parsed.time_range == "15m"  # server-declared default survives


# ---------------------------------------------------------------------- #
# Graceful degradation
# ---------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_unreachable_server_registers_stub_not_crash(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Simulated connection failure for every server: startup() must complete,
    register an UnavailableTool per expected tool name, and those stubs must
    produce failed-but-non-crashing ToolResults."""
    monkeypatch.setattr(get_settings(), "USE_MCP_TOOLS", True)

    async def refuse_connection(self: MCPClientManager, server: str, url: str) -> None:
        raise ConnectionError(f"connection refused: {url}")

    monkeypatch.setattr(
        MCPClientManager, "_connect_and_register", refuse_connection
    )

    manager = MCPClientManager()
    registry = await manager.startup()  # must NOT raise

    expected_names = [name for _, names in MCP_SERVERS.values() for name in names]
    assert manager.degraded_servers == list(MCP_SERVERS)
    registered = {spec["name"] for spec in registry.list_tools()}
    assert registered == set(expected_names)

    # A stub call degrades to a failed ToolResult — incident intake survives.
    result = await registry.execute(
        ToolCall(tool_name="query_logs", arguments={"service_name": "api"})
    )
    assert result.success is False
    assert "temporarily unavailable" in (result.error or "")

    await manager.shutdown()


@pytest.mark.asyncio
async def test_unavailable_stub_raises_typed_exception() -> None:
    stub = UnavailableTool("query_logs", "logs-metrics-mcp")
    with pytest.raises(MCPToolUnavailableError, match="temporarily unavailable"):
        await stub.run(service_name="api")


@pytest.mark.asyncio
async def test_use_mcp_tools_false_registers_in_process_mocks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The USE_MCP_TOOLS=False fallback: same four tool names, zero network."""
    monkeypatch.setattr(get_settings(), "USE_MCP_TOOLS", False)

    manager = MCPClientManager()
    registry = await manager.startup()

    names = {spec["name"] for spec in registry.list_tools()}
    assert names == {"query_logs", "query_metrics", "restart_service", "scale_deployment"}

    result = await registry.execute(
        ToolCall(tool_name="query_logs", arguments={"service_name": "api"})
    )
    assert result.success is True
    await manager.shutdown()
