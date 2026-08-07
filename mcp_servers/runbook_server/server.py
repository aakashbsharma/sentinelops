"""Runbook-search MCP server — MCP over an INTERNAL capability.

Run standalone:  uvicorn mcp_servers.runbook_server.server:app --port 8103

The other two servers simulate external systems; this one is the interesting
case: it wraps app.memory.semantic.SemanticMemoryStore — the same pgvector-
backed store the MemoryManager uses — and exposes it as an MCP tool. That's
the pattern for making an internal capability available to ANY MCP client
(other services, an operator's IDE, a different agent platform) without those
clients importing this codebase. It therefore deliberately imports from app/
(unlike the other servers) and needs DATABASE_URL, hence its depends_on
Postgres in compose.

Same streamable-HTTP-not-stdio reasoning as logs_metrics_server; see the
transport comment there.
"""

from typing import Any

from mcp.server.fastmcp import FastMCP
from mcp.server.transport_security import TransportSecuritySettings
from pydantic import Field
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from starlette.requests import Request
from starlette.responses import JSONResponse

from app.core.config import get_settings
from app.memory.embeddings import LocalEmbeddingProvider
from app.memory.semantic import SemanticMemoryStore

mcp = FastMCP(
    "sentinelops-runbook",
    stateless_http=True,
    json_response=True,
    transport_security=TransportSecuritySettings(
        enable_dns_rebinding_protection=True,
        allowed_hosts=[
            "localhost:*",
            "127.0.0.1:*",
            "runbook-mcp:*",
        ],
        allowed_origins=[
            "http://localhost:*",
            "http://127.0.0.1:*",
        ],
    ),
)
# This server is its own process: own engine/pool, never shared with the API.
# Created lazily so importing this module (e.g. to inspect the ASGI app)
# doesn't require a reachable database.
_engine = None
_session_factory = None
_store = SemanticMemoryStore(LocalEmbeddingProvider())


def _get_session_factory() -> async_sessionmaker:
    global _engine, _session_factory
    if _session_factory is None:
        _engine = create_async_engine(get_settings().DATABASE_URL, pool_pre_ping=True)
        _session_factory = async_sessionmaker(_engine, expire_on_commit=False)
    return _session_factory


@mcp.tool()
async def search_runbooks(
    query: str,
    k: int = Field(default=3, ge=1, le=10, description="Max excerpts to return."),
) -> dict[str, Any]:
    """Search the operations runbook library (semantic search over runbook
    excerpts). Returns the most relevant excerpts with similarity scores;
    excerpts below the relevance floor are omitted rather than padded in."""
    factory = _get_session_factory()
    async with factory() as session:
        results = await _store.retrieve(query, session, k=k)
    return {
        "query": query,
        "results": [
            {
                "content": record.content,
                "similarity": round(similarity, 3),
                "source_file": (record.metadata_ or {}).get("source_file"),
            }
            for record, similarity in results
        ],
    }


@mcp.custom_route("/health", methods=["GET"])
async def health(_: Request) -> JSONResponse:
    return JSONResponse({"status": "ok", "server": "runbook-mcp"})


app = mcp.streamable_http_app()
