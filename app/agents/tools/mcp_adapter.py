"""MCPToolAdapter — a remote MCP tool behind the local `Tool` interface.

The whole point of Stage 2's abstraction was that agents depend on `Tool` /
`ToolRegistry`, never on where a tool runs. This adapter is the proof: it
makes a tool living in another process (over MCP streamable HTTP) look
exactly like Stage 4's in-process mocks, so no agent code changes.

Error contract (verified by tests, not assumed): every failure raises an
MCPToolError subclass, and `ToolRegistry.execute()` already converts any
exception from `Tool.run()` into a failed ToolResult — so Stage 2's
exception handling needed zero changes for this swap.
"""

import json
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, create_model

from app.agents.tools.base import Tool
from app.agents.tools.exceptions import (
    MCPConnectionError,
    MCPToolError,
    MCPToolInvocationError,
    MCPToolUnavailableError,
)

_JSON_TYPE_TO_PY: dict[str, type] = {
    "string": str,
    "integer": int,
    "number": float,
    "boolean": bool,
    "object": dict,
    "array": list,
}


def _py_type_for(prop: dict[str, Any]) -> Any:
    """Best-effort JSON-schema-property -> Python type for client-side
    validation. Server-side validation remains authoritative; this only has
    to catch the obvious LLM mistakes (wrong field name, string-for-int)
    before they cross the network."""
    if "type" in prop:
        return _JSON_TYPE_TO_PY.get(prop["type"], Any)
    for variant in prop.get("anyOf", []):
        if variant.get("type") not in (None, "null"):
            return _JSON_TYPE_TO_PY.get(variant["type"], Any)
    return Any


def model_from_json_schema(model_name: str, schema: dict[str, Any]) -> type[BaseModel]:
    """Reconstruct a Pydantic model from an MCP tool's input JSON schema, so
    ToolRegistry.execute() can keep validating arguments before run() — the
    LLM stays an untrusted argument source regardless of where the tool lives.
    """
    required = set(schema.get("required", []))
    fields: dict[str, Any] = {}
    for prop_name, prop in (schema.get("properties") or {}).items():
        py_type = _py_type_for(prop)
        description = prop.get("description")
        if prop_name in required:
            fields[prop_name] = (py_type, Field(..., description=description))
        else:
            fields[prop_name] = (
                py_type | None,
                Field(default=prop.get("default"), description=description),
            )
    return create_model(model_name, **fields)


class MCPToolAdapter(Tool):
    """One remote MCP tool, presented through the local Tool interface.

    name/description/input schema come from the client's list_tools() at
    startup — the server's declarations are the single source of truth,
    never hardcoded twice.
    """

    def __init__(
        self,
        session: Any,  # mcp.ClientSession (Any so unit tests can pass fakes)
        *,
        name: str,
        description: str | None,
        input_json_schema: dict[str, Any] | None,
    ) -> None:
        self._session = session
        self.name = name
        self.description = description or f"Remote MCP tool {name!r}"
        self._server_schema = input_json_schema or {"type": "object", "properties": {}}
        self.input_schema = model_from_json_schema(
            f"{name.title().replace('_', '')}McpInput", self._server_schema
        )

    def spec(self) -> dict[str, Any]:
        # Prefer the server's schema verbatim for the LLM prompt: it carries
        # the server's own field descriptions/constraints, which the
        # reconstructed validation model only approximates.
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self._server_schema,
        }

    async def run(self, **kwargs: Any) -> Any:
        try:
            result = await self._session.call_tool(self.name, arguments=kwargs)
        except MCPToolError:
            raise
        except Exception as exc:  # transport/session failure
            raise MCPConnectionError(
                f"MCP call to {self.name!r} failed: {type(exc).__name__}: {exc}"
            ) from exc

        if getattr(result, "isError", False):
            # Tool-side exception, reported in-band per the MCP spec.
            raise MCPToolInvocationError(
                f"MCP tool {self.name!r} reported an error: {self._text_of(result)}"
            )
        return self._parse_result(result)

    # ------------------------------------------------------------------ #
    # Result parsing: MCP content -> the plain Python value the rest of
    # the codebase expects (the same dicts the Stage 4 mocks returned).
    # ------------------------------------------------------------------ #

    def _parse_result(self, result: Any) -> Any:
        structured = getattr(result, "structuredContent", None)
        if structured is not None:
            # FastMCP wraps non-object return values as {"result": value}.
            if set(structured) == {"result"}:
                return structured["result"]
            return structured

        text = self._text_of(result)
        try:
            return json.loads(text)
        except (json.JSONDecodeError, TypeError):
            return text

    @staticmethod
    def _text_of(result: Any) -> str:
        parts = [
            getattr(block, "text", "")
            for block in getattr(result, "content", []) or []
            if getattr(block, "type", None) == "text"
        ]
        return "\n".join(p for p in parts if p)


class _AnyInput(BaseModel):
    """Accept anything — an unavailable tool can't know its real schema."""

    model_config = ConfigDict(extra="allow")


class UnavailableTool(Tool):
    """Degradation stub registered when an MCP server is unreachable at
    startup. Calling it raises MCPToolUnavailableError, which
    ToolRegistry.execute() converts into ToolResult(success=False, ...) —
    the agent observes "tool temporarily unavailable" and can route around
    it, instead of the whole app refusing to start over one dead dependency.
    """

    input_schema = _AnyInput

    def __init__(self, name: str, server_name: str) -> None:
        self.name = name
        self.server_name = server_name
        self.description = (
            f"{name} (TEMPORARILY UNAVAILABLE — the {server_name} backend "
            "could not be reached; calls will fail until it recovers)"
        )

    async def run(self, **kwargs: Any) -> Any:
        raise MCPToolUnavailableError(
            f"tool temporarily unavailable: MCP server {self.server_name!r} "
            "was unreachable at startup"
        )
