"""Application configuration via pydantic-settings.

All runtime configuration is env-driven (12-factor). `get_settings()` is the
single cached entry point — import it, never instantiate `Settings` directly.
"""

from functools import lru_cache
from typing import Literal

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
    )

    DATABASE_URL: str = (
        "postgresql+asyncpg://sentinelops:sentinelops@localhost:5432/sentinelops"
    )
    REDIS_URL: str = "redis://localhost:6379/0"
    ENVIRONMENT: Literal["dev", "staging", "prod"] = "dev"
    LOG_LEVEL: str = "INFO"
    ANTHROPIC_API_KEY: str | None = None
    GROQ_API_KEY: str = "llama-3.3-70b-versatile"
    # Agent guardrails — enforced by the loop engine from Stage 2 onward.
    MAX_AGENT_ITERATIONS: int = 8
    AGENT_TOKEN_BUDGET: int = 50_000

    # Orchestrator guardrails (Stage 5).
    # How many times a stage may be re-run when its agent reports
    # needs_more_data (a self-loop in the incident state machine).
    MAX_RETRY_ATTEMPTS: int = 2
    # Incident-level safety cap on TOTAL agent invocations, separate from each
    # agent's own max_iterations: a triage<->diagnose retry loop compounded
    # with per-agent iterations could otherwise make one incident arbitrarily
    # expensive. Exceeding this force-escalates to a human.
    MAX_ORCHESTRATION_ROUNDS: int = 6

    # MCP tool servers (Stage 6). Each is an independent uvicorn process
    # speaking MCP streamable HTTP; the /mcp path is the protocol endpoint.
    LOGS_METRICS_MCP_URL: str = "http://localhost:8101/mcp"
    INFRA_MCP_URL: str = "http://localhost:8102/mcp"
    RUNBOOK_MCP_URL: str = "http://localhost:8103/mcp"
    # False = register Stage 4's in-process mock tools instead of connecting
    # to MCP servers. This is what keeps the unit suite fast and hermetic
    # (no network, no subprocesses) while dev/prod and the integration tests
    # use the real MCP path.
    USE_MCP_TOOLS: bool = True
    # Startup budget per MCP server before declaring it unreachable and
    # registering degradation stubs.
    MCP_CONNECT_TIMEOUT_S: float = 10.0

    # Observability (Stage 7): approximate blended $/token per model, used by
    # the per-incident cost summary. ILLUSTRATIVE ONLY, not billing-accurate:
    # real pricing distinguishes input/output/cached tokens and changes over
    # time; a single blended constant is enough to answer "roughly what did
    # this agent run cost?" — which is the question the trace panel asks.
    # "default" is the fallback for messages that predate model stamping.
    MODEL_COST_PER_TOKEN: dict[str, float] = {
        "claude-sonnet-4-6": 6e-6,
        "default": 6e-6,
    }

    # Observability (Stage 7). OpenTelemetry spans around agent runs and tool
    # executions, exported to the console — off by default so tests and normal
    # dev output stay clean. Correlation ids (incident_id/run_id) flow via
    # contextvars regardless of this flag; this only controls span export.
    OTEL_ENABLED: bool = False

    # THE production execution switch (see ExecutorAgent._resolve_dry_run):
    # False = every remediation runs as dry_run, even when human-approved.
    # Flipping this to True is the single, auditable change that lets the
    # executor touch (simulated) infrastructure for real.
    EXECUTOR_ALLOW_REAL_EXECUTION: bool = False

    @field_validator("DATABASE_URL")
    @classmethod
    def _require_asyncpg_driver(cls, v: str) -> str:
        if not v.startswith("postgresql+asyncpg://"):
            raise ValueError(
                "DATABASE_URL must start with 'postgresql+asyncpg://' — "
                "SentinelOps is fully async and does not support sync drivers."
            )
        return v


@lru_cache
def get_settings() -> Settings:
    return Settings()
