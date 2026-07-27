from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import health, incidents, ws_trace
from app.core.config import get_settings
from app.core.logging import configure_logging, get_logger
from app.core.mcp_clients import MCPClientManager


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Connect to the MCP tool servers on startup, close cleanly on shutdown.

    MCPClientManager.startup() never raises for an unreachable server — it
    registers degradation stubs instead, so incident intake stays up even if
    every tool server is down. The registry lands on app.state for anything
    that runs agents in-process (and Stage 7's observability endpoints);
    Celery workers build their own per-run connections (see app/tasks.py).
    """
    manager = MCPClientManager()
    app.state.mcp_manager = manager
    app.state.tool_registry = await manager.startup()
    try:
        yield
    finally:
        await manager.shutdown()


def create_app() -> FastAPI:
    configure_logging()
    settings = get_settings()
    log = get_logger("sentinelops.startup")

    app = FastAPI(
        title="SentinelOps",
        description="Autonomous multi-agent AI incident response platform",
        version="0.1.0",
        lifespan=lifespan,
    )

    # Allow the local React dashboard (Stage 9) during development.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "http://localhost:3000",
            "http://localhost:5173",
            "http://127.0.0.1:3000",
            "http://127.0.0.1:5173",
        ],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(health.router)
    app.include_router(incidents.router)
    app.include_router(ws_trace.router)

    log.info(
        "app_created",
        environment=settings.ENVIRONMENT,
        use_mcp_tools=settings.USE_MCP_TOOLS,
    )
    return app


app = create_app()
