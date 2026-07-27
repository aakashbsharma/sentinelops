"""End-to-end MCP integration test — the one that proves the wiring is real.

Starts the actual logs_metrics_server as a uvicorn subprocess, connects a
real MCP client over streamable HTTP, discovers tools via list_tools(),
wraps them as MCPToolAdapters in the full Stage 2 ToolRegistry, and executes
query_logs through registry.execute() — the exact code path an agent's loop
takes. Excluded from the default run (see pytest.ini); run with:

    pytest -m integration tests/test_mcp_integration.py -o addopts=""
"""

import os
import subprocess
import sys
import time
from collections.abc import Iterator

import httpx
import pytest

from app.agents.schemas import ToolCall
from app.agents.tools.base import ToolRegistry
from app.agents.tools.mcp_adapter import MCPToolAdapter
from mcp_servers.logs_metrics_server.generators import generate_logs

pytestmark = pytest.mark.integration

PORT = 8151
BASE_URL = f"http://127.0.0.1:{PORT}"
SCENARIO = "memory_leak"


@pytest.fixture(scope="module")
def logs_metrics_server() -> Iterator[str]:
    """Run the real server exactly as production would: its own uvicorn
    process, scenario injected via env — not a test double of any kind."""
    env = {**os.environ, "SIMULATED_SCENARIO": SCENARIO}
    proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "mcp_servers.logs_metrics_server.server:app",
            "--host",
            "127.0.0.1",
            "--port",
            str(PORT),
        ],
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        deadline = time.monotonic() + 30
        while True:
            try:
                if httpx.get(f"{BASE_URL}/health", timeout=1).status_code == 200:
                    break
            except httpx.HTTPError:
                pass
            if proc.poll() is not None:
                raise RuntimeError("logs_metrics_server exited during startup")
            if time.monotonic() > deadline:
                raise TimeoutError("logs_metrics_server did not become healthy")
            time.sleep(0.2)
        yield f"{BASE_URL}/mcp"
    finally:
        proc.kill()
        proc.wait(timeout=10)


@pytest.mark.asyncio
async def test_query_logs_end_to_end_through_registry(
    logs_metrics_server: str,
) -> None:
    from mcp import ClientSession
    from mcp.client.streamable_http import streamable_http_client

    async with streamable_http_client(logs_metrics_server) as (read, write, _):
        async with ClientSession(read, write) as session:
            await session.initialize()

            # Discovery, not hardcoding: schemas come from the server.
            listed = await session.list_tools()
            names = {t.name for t in listed.tools}
            assert names == {"query_logs", "query_metrics"}

            registry = ToolRegistry()
            for tool in listed.tools:
                registry.register(
                    MCPToolAdapter(
                        session,
                        name=tool.name,
                        description=tool.description,
                        input_json_schema=tool.inputSchema,
                    )
                )

            # The exact call path an agent loop takes: ToolCall -> registry.
            result = await registry.execute(
                ToolCall(
                    tool_name="query_logs",
                    arguments={"service_name": "payment-service", "time_range": "15m"},
                )
            )

            assert result.success is True
            assert any("OOMKilled" in line for line in result.output["lines"])

            # Byte-for-byte identical to the in-process generator — the mock
            # path and the MCP path serve the same deterministic world.
            assert result.output == generate_logs(
                "payment-service", "15m", scenario=SCENARIO
            )

            # And a validation failure is stopped client-side, same as mocks.
            bad = await registry.execute(
                ToolCall(tool_name="query_logs", arguments={})
            )
            assert bad.success is False
            assert "Invalid arguments" in (bad.error or "")
