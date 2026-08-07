"""Logs & metrics MCP server — simulates the observability stack.

Run standalone:  uvicorn mcp_servers.logs_metrics_server.server:app --port 8101

TRANSPORT CHOICE — streamable HTTP, not stdio: stdio couples an MCP server's
lifecycle to a parent process that spawns it as a subprocess, which does not
fit a long-running containerized FastAPI service (the app would have to fork
and babysit its own tool servers, and they couldn't be scaled, health-checked
or restarted independently). Streamable HTTP makes each server a plain HTTP
endpoint (/mcp) — its own container, its own port, standard Docker Compose
healthchecks and depends_on, standard reverse proxies. We run it stateless
with JSON responses: every tool here is a simple request/response call, so
per-session server state would only add reconnection failure modes.

WHICH SCENARIO IS ACTIVE: the simulated world's state comes from the
SIMULATED_SCENARIO env var (e.g. connection_pool_exhaustion, memory_leak,
bad_deploy, retry_storm; unset = healthy). It is deliberately NOT a tool
parameter — the agent must not be able to choose its own evidence.
"""

import os
from typing import Any

from mcp.server.fastmcp import FastMCP
from mcp.server.transport_security import TransportSecuritySettings
from starlette.requests import Request
from starlette.responses import JSONResponse

from mcp_servers.logs_metrics_server.generators import generate_logs, generate_metrics

mcp = FastMCP(
    "sentinelops-logs-metrics",
    stateless_http=True,
    json_response=True,
    transport_security=TransportSecuritySettings(
        enable_dns_rebinding_protection=True,
        allowed_hosts=[
            "localhost:*",
            "127.0.0.1:*",
            "logs-metrics-mcp:*",
        ],
        allowed_origins=[
            "http://localhost:*",
            "http://127.0.0.1:*",
        ],
    ),
)


def _scenario() -> str | None:
    # Read per-call (not at import) so tests/compose can change the simulated
    # world without restarting the process.
    return os.environ.get("SIMULATED_SCENARIO") or None


# NOTE: these docstrings and typed signatures ARE the MCP contract — the
# client lists them at startup and injects them into the agents' LLM prompts.
# Keep them accurate; a misleading description here misleads every agent.


@mcp.tool()
async def query_logs(service_name: str, time_range: str = "15m") -> dict[str, Any]:
    """Query recent application logs for a service. Returns log lines from
    the requested time range (e.g. 15m, 1h, 24h)."""
    return generate_logs(service_name, time_range, scenario=_scenario())


@mcp.tool()
async def query_metrics(
    service_name: str, metric: str = "all", time_range: str = "15m"
) -> dict[str, Any]:
    """Query time-series metrics (cpu, memory, latency_p99, error_rate — or
    'all') for a service. Returns 12 data points over the time range."""
    return generate_metrics(service_name, metric, time_range, scenario=_scenario())


@mcp.custom_route("/health", methods=["GET"])
async def health(_: Request) -> JSONResponse:
    """Plain HTTP healthcheck for Docker Compose / load balancers."""
    return JSONResponse({"status": "ok", "server": "logs-metrics-mcp"})


# ASGI app for uvicorn; the MCP endpoint lives at /mcp.
app = mcp.streamable_http_app()
