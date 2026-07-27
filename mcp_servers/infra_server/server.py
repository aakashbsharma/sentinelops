"""Infrastructure-actions MCP server — the MUTATING tools.

Run standalone:  uvicorn mcp_servers.infra_server.server:app --port 8102

Same streamable-HTTP-not-stdio reasoning as logs_metrics_server; see the
transport comment there. Same dry_run semantics as Stage 4's mocks: the
dry_run branch is the real, transferable logic; the execute branch logs
loudly and pretends (no real infra in this project). The app-side Executor
still forces dry_run from EXECUTOR_ALLOW_REAL_EXECUTION regardless of what
arrives here — defense in depth, not this server's trust in its callers.
"""

import logging
from typing import Any

from mcp.server.fastmcp import FastMCP
from pydantic import Field
from starlette.requests import Request
from starlette.responses import JSONResponse

logger = logging.getLogger("infra_mcp")

mcp = FastMCP("sentinelops-infra", stateless_http=True, json_response=True)


@mcp.tool()
async def restart_service(
    service_name: str,
    dry_run: bool = Field(
        default=True,
        description="If true, describe what would happen without doing it.",
    ),
) -> dict[str, Any]:
    """Perform a rolling restart of a service's pods. MUTATING ACTION —
    supports dry_run."""
    if dry_run:
        return {
            "dry_run": True,
            "would_do": f"rolling restart of {service_name} (3 pods, one at a time)",
            "expected_downtime": "none (rolling)",
        }
    logger.warning(
        "SIMULATED EXECUTION: restart_service service=%s (would have executed in real infra)",
        service_name,
    )
    return {
        "dry_run": False,
        "executed": True,
        "detail": f"[SIMULATED] rolling restart of {service_name} completed",
    }


@mcp.tool()
async def scale_deployment(
    deployment: str,
    replicas: int = Field(ge=0, le=50),
    dry_run: bool = Field(
        default=True,
        description="If true, describe what would happen without doing it.",
    ),
) -> dict[str, Any]:
    """Scale a deployment to N replicas. MUTATING ACTION — supports dry_run."""
    if dry_run:
        return {
            "dry_run": True,
            "would_do": f"scale {deployment} to {replicas} replicas",
        }
    logger.warning(
        "SIMULATED EXECUTION: scale_deployment deployment=%s replicas=%s "
        "(would have executed in real infra)",
        deployment,
        replicas,
    )
    return {
        "dry_run": False,
        "executed": True,
        "detail": f"[SIMULATED] {deployment} scaled to {replicas} replicas",
    }


@mcp.custom_route("/health", methods=["GET"])
async def health(_: Request) -> JSONResponse:
    return JSONResponse({"status": "ok", "server": "infra-mcp"})


app = mcp.streamable_http_app()
