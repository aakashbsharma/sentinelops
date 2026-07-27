class ToolError(Exception):
    """Base class for tool-layer errors."""


class ToolNotFoundError(ToolError):
    """Raised by ToolRegistry.get() for an unregistered tool name.

    Note: `ToolRegistry.execute()` never propagates this — it converts it to a
    failed ToolResult so the agent loop can observe the mistake and recover
    (the LLM may simply have hallucinated a tool name).
    """


class ToolAlreadyRegisteredError(ToolError):
    """Raised when two tools are registered under the same name."""


class MCPToolError(ToolError):
    """Base class for MCP-backed tool failures (Stage 6).

    Subclasses of ToolError on purpose: `ToolRegistry.execute()` already
    converts any exception from `Tool.run()` into a failed ToolResult, so
    MCP failures flow through the exact same Stage 2 path as every other
    tool failure — observed by the agent, never crashing the loop.
    """


class MCPConnectionError(MCPToolError):
    """Transport-level failure: server unreachable, session dropped, timeout."""


class MCPToolInvocationError(MCPToolError):
    """The MCP server executed the tool and reported an error (isError=True)."""


class MCPToolUnavailableError(MCPToolError):
    """Raised by the degradation stub registered when an MCP server was
    unreachable at startup (see app/core/mcp_clients.py)."""
