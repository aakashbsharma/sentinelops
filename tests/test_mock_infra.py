"""Determinism and dry-run tests for the mock infrastructure tools."""

import pytest

from app.agents.tools.mock_infra import (
    QueryLogsTool,
    QueryMetricsTool,
    RestartServiceTool,
)


@pytest.mark.asyncio
async def test_logs_deterministic_for_same_inputs() -> None:
    """Same incident (scenario + args) must always surface the same evidence —
    required for reproducible evals in Stage 8."""
    tool_a = QueryLogsTool(scenario="memory_leak")
    tool_b = QueryLogsTool(scenario="memory_leak")

    first = await tool_a.run(service_name="payment-service", time_range="15m")
    second = await tool_b.run(service_name="payment-service", time_range="15m")

    assert first == second
    assert any("OOMKilled" in line for line in first["lines"])  # scenario evidence present


@pytest.mark.asyncio
async def test_logs_differ_across_services_and_scenarios() -> None:
    tool = QueryLogsTool(scenario="bad_deploy")
    a = await tool.run(service_name="payment-service", time_range="15m")
    b = await tool.run(service_name="search-service", time_range="15m")
    assert a["lines"] != b["lines"]

    healthy = QueryLogsTool(scenario=None)
    c = await healthy.run(service_name="payment-service", time_range="15m")
    assert not any("KeyError" in line for line in c["lines"])


@pytest.mark.asyncio
async def test_metrics_show_scenario_pathology() -> None:
    tool = QueryMetricsTool(scenario="connection_pool_exhaustion")
    result = await tool.run(service_name="api", metric="all", time_range="15m")

    latency = result["series"]["latency_p99"]
    assert latency[-1] > latency[0] * 5  # ramps toward pathological
    cpu = result["series"]["cpu"]
    assert max(cpu) < 50  # unaffected metric stays near baseline


@pytest.mark.asyncio
async def test_restart_dry_run_vs_execute_branches() -> None:
    tool = RestartServiceTool()

    dry = await tool.run(service_name="payment-service", dry_run=True)
    assert dry["dry_run"] is True
    assert "would_do" in dry
    assert "executed" not in dry

    wet = await tool.run(service_name="payment-service", dry_run=False)
    assert wet["dry_run"] is False
    assert wet["executed"] is True
    assert "SIMULATED" in wet["detail"]
