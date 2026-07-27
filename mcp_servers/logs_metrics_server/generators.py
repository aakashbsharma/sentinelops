"""Deterministic synthetic log/metric generation (moved from Stage 4's
app/agents/tools/mock_infra.py — this is now the single source of truth; the
in-process mock tools delegate here so USE_MCP_TOOLS=False produces byte-for-
byte the same evidence as the real MCP servers).

DETERMINISM: outputs are generated from an md5-seeded RNG keyed on the
arguments plus an optional scenario hint. The same incident always surfaces
the same "evidence" — required for reproducible tests and for the Stage 8
eval harness, where a flaky mock would make accuracy metrics meaningless.
"""

import hashlib
import random
from typing import Any


def _rng(*parts: Any) -> random.Random:
    seed_material = "|".join(str(p) for p in parts)
    seed = int.from_bytes(hashlib.md5(seed_material.encode()).digest()[:8], "little")
    return random.Random(seed)


# Known incident scenarios and their tell-tale evidence. The simulated infra
# "contains" the signal an agent should find; which scenario is active comes
# from the caller (mock tools: incident.raw_payload; MCP server: the
# SIMULATED_SCENARIO env var — the state of the simulated world).
SCENARIO_LOG_SIGNATURES: dict[str, list[str]] = {
    "connection_pool_exhaustion": [
        "ERROR TimeoutError: could not acquire connection from pool after 30.0s",
        "WARNING sqlalchemy.pool QueuePool limit of size 10 overflow 20 reached",
        "ERROR asyncpg.exceptions.TooManyConnectionsError: remaining connection slots are reserved",
    ],
    "memory_leak": [
        "WARNING container memory usage at 94% of limit",
        "ERROR OOMKilled: container exceeded memory limit (2Gi)",
        "WARNING gc pause of 2100ms detected in retry buffer worker",
    ],
    "bad_deploy": [
        "ERROR KeyError: 'PAYMENT_GATEWAY_URL' during startup config load",
        "ERROR 500 Internal Server Error on POST /api/charge",
        "WARNING readiness probe failing since deploy 2026-07-15T09:14Z",
    ],
    "retry_storm": [
        "WARNING downstream card-processor returned 503, scheduling retry 8/10",
        "ERROR retry budget exhausted for processor gateway",
        "WARNING CPU throttling detected: 97% utilization for 300s",
    ],
}

_BASELINE_LINES = [
    "INFO request completed path=/api/health status=200 duration_ms=3",
    "INFO request completed path=/api/charge status=200 duration_ms=142",
    "INFO worker heartbeat ok queue_depth=0",
    "INFO request completed path=/api/refund status=200 duration_ms=201",
]

# Which metric goes pathological per scenario (baseline: all healthy).
# metric: (healthy_baseline, pathological_peak)
SCENARIO_METRIC_SHAPES: dict[str, dict[str, tuple[float, float]]] = {
    "connection_pool_exhaustion": {"latency_p99": (180.0, 4800.0), "error_rate": (0.2, 14.0)},
    "memory_leak": {"memory": (48.0, 96.0), "error_rate": (0.2, 6.0)},
    "bad_deploy": {"error_rate": (0.2, 22.0)},
    "retry_storm": {"cpu": (35.0, 97.0), "latency_p99": (180.0, 2600.0)},
}

METRIC_BASELINES: dict[str, float] = {
    "cpu": 35.0,
    "memory": 48.0,
    "latency_p99": 180.0,
    "error_rate": 0.2,
}


def generate_logs(
    service_name: str, time_range: str = "15m", scenario: str | None = None
) -> dict[str, Any]:
    rng = _rng("logs", service_name, time_range, scenario)

    signature = SCENARIO_LOG_SIGNATURES.get(scenario or "", [])
    lines: list[str] = []
    for i in range(12):
        # Deterministically interleave scenario evidence into baseline
        # noise (roughly every third line when a scenario is active).
        if signature and i % 3 == 1:
            lines.append(f"[{service_name}] {signature[(i // 3) % len(signature)]}")
        else:
            lines.append(f"[{service_name}] {rng.choice(_BASELINE_LINES)}")
    return {"service": service_name, "time_range": time_range, "lines": lines}


def generate_metrics(
    service_name: str,
    metric: str = "all",
    time_range: str = "15m",
    scenario: str | None = None,
) -> dict[str, Any]:
    shapes = SCENARIO_METRIC_SHAPES.get(scenario or "", {})

    metrics = list(METRIC_BASELINES) if metric == "all" else [metric]
    series: dict[str, list[float]] = {}
    for name in metrics:
        rng = _rng("metrics", service_name, name, time_range, scenario)
        baseline = METRIC_BASELINES.get(name, 1.0)
        if name in shapes:
            start, peak = shapes[name]
            # Ramp from healthy to pathological across the window.
            series[name] = [
                round(start + (peak - start) * (i / 11) + rng.uniform(-2, 2), 2)
                for i in range(12)
            ]
        else:
            series[name] = [
                round(baseline + rng.uniform(-baseline * 0.1, baseline * 0.1), 2)
                for _ in range(12)
            ]
    return {"service": service_name, "time_range": time_range, "series": series}
