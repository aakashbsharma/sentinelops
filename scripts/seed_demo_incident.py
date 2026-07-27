"""Seed one realistic demo incident via the API.

    python scripts/seed_demo_incident.py [--api-url http://localhost:8000]

Plain HTTP against POST /incidents — exactly what a real alerting webhook
would do. The raw_payload's `scenario` key seeds the simulated world's
evidence (see mcp_servers/logs_metrics_server/generators.py), so the agents
will find real pool-exhaustion signals in the logs/metrics they query, and
the dashboard has something meaningful to show immediately.
"""

import argparse
import sys

import httpx

DEMO_INCIDENT = {
    "title": "checkout-api: database connection timeouts spiking",
    "description": (
        "PagerDuty alert: checkout-api p99 latency degraded from ~180ms to "
        ">4s over the last 15 minutes. Application logs show database "
        "timeout errors. Checkout conversion dropping. On-call paged."
    ),
    "source": "demo-seed",
    "severity": "unknown",  # provisional — the Triage agent overwrites this
    "raw_payload": {
        "scenario": "connection_pool_exhaustion",
        "alert_name": "CheckoutApiDbTimeouts",
        "service": "checkout-api",
        "environment": "production",
        "fired_at": "2026-07-16T14:02:00Z",
        "metric_snapshot": {
            "latency_p99_ms": 4310,
            "error_rate_pct": 11.8,
            "requests_per_s": 240,
        },
    },
}


def main() -> int:
    parser = argparse.ArgumentParser(description="Create one demo incident via the API.")
    parser.add_argument("--api-url", default="http://localhost:8000")
    args = parser.parse_args()

    try:
        response = httpx.post(
            f"{args.api_url}/incidents", json=DEMO_INCIDENT, timeout=10.0
        )
        response.raise_for_status()
    except httpx.HTTPError as exc:
        print(f"Failed to create incident: {exc}", file=sys.stderr)
        print(
            "Is the API running (uvicorn app.main:app) and reachable at "
            f"{args.api_url}?",
            file=sys.stderr,
        )
        return 1

    body = response.json()
    print(f"Created incident {body['id']} (status={body['status']})")
    print(f"  API:       {args.api_url}/incidents/{body['id']}")
    print(f"  Dashboard: http://localhost:5173 (select the incident)")
    print(
        "A Celery worker must be running for the agents to pick it up; "
        "the trace panel will stream their steps live."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
