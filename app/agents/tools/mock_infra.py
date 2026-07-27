"""Mock infrastructure tools — the in-process fallback (USE_MCP_TOOLS=False).

Stage 6 moved the deterministic synthetic data generation to
mcp_servers/logs_metrics_server/generators.py, where the real MCP servers
serve it from. These Tool classes now delegate there, so the in-process mocks
and the MCP path produce byte-for-byte identical evidence — the unit suite
(which runs with USE_MCP_TOOLS=False, no network) exercises the same
synthetic world the MCP demo path does.

Same `Tool` interface the MCP adapters implement, so swapping mocks for real
backends is a registration change, not a rewrite.
"""

from typing import Any

import structlog
from pydantic import BaseModel, Field

from app.agents.tools.base import Tool
from mcp_servers.logs_metrics_server.generators import generate_logs, generate_metrics

logger = structlog.get_logger(__name__)


class QueryLogsInput(BaseModel):
    service_name: str
    time_range: str = Field(default="15m", description="e.g. 15m, 1h, 24h")


class QueryLogsTool(Tool):
    name = "query_logs"
    description = (
        "Query recent application logs for a service. Returns log lines from "
        "the requested time range."
    )
    input_schema = QueryLogsInput

    def __init__(self, scenario: str | None = None) -> None:
        self.scenario = scenario

    async def run(self, **kwargs: Any) -> Any:
        return generate_logs(
            kwargs["service_name"],
            kwargs.get("time_range", "15m"),
            scenario=self.scenario,
        )


class QueryMetricsInput(BaseModel):
    service_name: str
    metric: str = Field(
        default="all", description="cpu | memory | latency_p99 | error_rate | all"
    )
    time_range: str = Field(default="15m")


class QueryMetricsTool(Tool):
    name = "query_metrics"
    description = (
        "Query time-series metrics (cpu, memory, latency_p99, error_rate) "
        "for a service. Returns 12 data points over the time range."
    )
    input_schema = QueryMetricsInput

    def __init__(self, scenario: str | None = None) -> None:
        self.scenario = scenario

    async def run(self, **kwargs: Any) -> Any:
        return generate_metrics(
            kwargs["service_name"],
            kwargs.get("metric", "all"),
            kwargs.get("time_range", "15m"),
            scenario=self.scenario,
        )


class RestartServiceInput(BaseModel):
    service_name: str
    dry_run: bool = Field(
        default=True,
        description="If true, describe what would happen without doing it.",
    )


class RestartServiceTool(Tool):
    name = "restart_service"
    description = (
        "Perform a rolling restart of a service's pods. MUTATING ACTION — "
        "supports dry_run."
    )
    input_schema = RestartServiceInput

    async def run(self, **kwargs: Any) -> Any:
        service = kwargs["service_name"]
        # The dry_run branch is the REAL logic that transfers to the MCP
        # infra server — only the "execute" arm is simulated here.
        if kwargs.get("dry_run", True):
            return {
                "dry_run": True,
                "would_do": f"rolling restart of {service} (3 pods, one at a time)",
                "expected_downtime": "none (rolling)",
            }
        # No real infra in this project: log loudly and pretend.
        logger.warning(
            "simulated_execution",
            tool=self.name,
            service=service,
            note="would have executed in real infra",
        )
        return {
            "dry_run": False,
            "executed": True,
            "detail": f"[SIMULATED] rolling restart of {service} completed",
        }


class ScaleDeploymentInput(BaseModel):
    deployment: str
    replicas: int = Field(ge=0, le=50)
    dry_run: bool = Field(default=True)


class ScaleDeploymentTool(Tool):
    name = "scale_deployment"
    description = (
        "Scale a deployment to N replicas. MUTATING ACTION — supports dry_run."
    )
    input_schema = ScaleDeploymentInput

    async def run(self, **kwargs: Any) -> Any:
        deployment = kwargs["deployment"]
        replicas = kwargs["replicas"]
        if kwargs.get("dry_run", True):
            return {
                "dry_run": True,
                "would_do": f"scale {deployment} to {replicas} replicas",
            }
        logger.warning(
            "simulated_execution",
            tool=self.name,
            deployment=deployment,
            replicas=replicas,
            note="would have executed in real infra",
        )
        return {
            "dry_run": False,
            "executed": True,
            "detail": f"[SIMULATED] {deployment} scaled to {replicas} replicas",
        }
