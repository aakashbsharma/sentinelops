"""Tool abstraction + registry.

Design rule: `ToolRegistry.execute()` NEVER raises. Every failure mode —
unknown tool, invalid arguments, timeout, tool bug — becomes a failed
`ToolResult`. A tool failure is information the agent should reason about
in its next REFLECT/PLAN, not an exception that unwinds the loop.
"""

import asyncio
import time
from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel, ValidationError

from app.agents.schemas import ToolCall, ToolResult
from app.agents.tools.exceptions import ToolAlreadyRegisteredError, ToolNotFoundError
from app.core import tracing


class Tool(ABC):
    """Base class for all tools.

    Subclasses set `name`, `description`, and `input_schema` (a Pydantic model
    class used to validate arguments before `run` is ever called — the LLM is
    an untrusted argument source), and implement async `run`.
    """

    name: str
    description: str
    input_schema: type[BaseModel]

    @abstractmethod
    async def run(self, **kwargs: Any) -> Any:
        """Execute the tool with already-validated arguments."""

    def spec(self) -> dict[str, Any]:
        """Serializable description for injection into an agent's prompt
        (matches the shape Anthropic tool-use expects)."""
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_schema.model_json_schema(),
        }


class ToolRegistry:
    def __init__(self, default_timeout_s: float = 30.0) -> None:
        self._tools: dict[str, Tool] = {}
        self.default_timeout_s = default_timeout_s

    def register(self, tool: Tool) -> None:
        if tool.name in self._tools:
            raise ToolAlreadyRegisteredError(f"Tool {tool.name!r} already registered")
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool:
        try:
            return self._tools[name]
        except KeyError:
            raise ToolNotFoundError(f"No tool registered under {name!r}") from None

    def list_tools(self) -> list[dict[str, Any]]:
        return [tool.spec() for tool in self._tools.values()]

    async def execute(
        self, tool_call: ToolCall, timeout_s: float | None = None
    ) -> ToolResult:
        """Execute a ToolCall and always return a ToolResult (never raises)."""
        # Span carries incident_id/run_id from the ambient contextvars — this
        # method never receives them explicitly (see app/core/tracing.py).
        with tracing.span("tool.execute", tool_name=tool_call.tool_name) as otel_span:
            result = await self._execute_inner(tool_call, timeout_s)
            if otel_span is not None:
                otel_span.set_attribute("tool.success", result.success)
                otel_span.set_attribute("tool.latency_ms", result.latency_ms)
            return result

    async def _execute_inner(
        self, tool_call: ToolCall, timeout_s: float | None = None
    ) -> ToolResult:
        start = time.perf_counter()

        def _failed(error: str) -> ToolResult:
            return ToolResult(
                call_id=tool_call.call_id,
                success=False,
                error=error,
                latency_ms=(time.perf_counter() - start) * 1000,
            )

        # The LLM may hallucinate a tool name — that's an observation for the
        # agent ("tool does not exist"), not a crash.
        try:
            tool = self.get(tool_call.tool_name)
        except ToolNotFoundError as exc:
            return _failed(str(exc))

        # Validate before running: the LLM is an untrusted argument source and
        # a typo'd field must not reach real infrastructure code.
        try:
            validated = tool.input_schema(**tool_call.arguments)
        except ValidationError as exc:
            return _failed(f"Invalid arguments for {tool.name!r}: {exc}")

        # Timeout: a hung tool (stuck network call, deadlocked mock) would
        # otherwise freeze the whole agent loop — and the incident it's
        # supposed to be resolving.
        try:
            output = await asyncio.wait_for(
                tool.run(**validated.model_dump()),
                timeout=timeout_s if timeout_s is not None else self.default_timeout_s,
            )
        except asyncio.TimeoutError:
            return _failed(f"Tool {tool.name!r} timed out")
        except Exception as exc:  # noqa: BLE001 — deliberate: tool bugs become data
            return _failed(f"Tool {tool.name!r} raised {type(exc).__name__}: {exc}")

        return ToolResult(
            call_id=tool_call.call_id,
            success=True,
            output=output,
            latency_ms=(time.perf_counter() - start) * 1000,
        )


class EchoInput(BaseModel):
    message: str


class EchoTool(Tool):
    """Trivial tool for exercising the loop machinery in tests.

    Real tools (query_logs, search_runbooks, restart_service, ...) arrive in
    Stage 6 as MCP-backed implementations of the same `Tool` interface.
    """

    name = "echo"
    description = "Returns the message it was given. For testing only."
    input_schema = EchoInput

    async def run(self, **kwargs: Any) -> Any:
        return {"echo": kwargs["message"]}
