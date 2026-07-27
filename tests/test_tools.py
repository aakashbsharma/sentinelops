"""Tests for the Tool abstraction and ToolRegistry failure handling."""

import asyncio
from typing import Any

import pytest
from pydantic import BaseModel

from app.agents.schemas import ToolCall
from app.agents.tools.base import EchoTool, Tool, ToolRegistry
from app.agents.tools.exceptions import ToolAlreadyRegisteredError, ToolNotFoundError


class SleepInput(BaseModel):
    seconds: float


class SleepTool(Tool):
    name = "sleep"
    description = "Sleeps for N seconds."
    input_schema = SleepInput

    async def run(self, **kwargs: Any) -> Any:
        await asyncio.sleep(kwargs["seconds"])
        return "woke up"


@pytest.mark.asyncio
async def test_echo_tool_roundtrip() -> None:
    registry = ToolRegistry()
    registry.register(EchoTool())

    result = await registry.execute(
        ToolCall(tool_name="echo", arguments={"message": "hello"})
    )

    assert result.success is True
    assert result.output == {"echo": "hello"}
    assert result.latency_ms >= 0


@pytest.mark.asyncio
async def test_invalid_arguments_fail_without_raising() -> None:
    registry = ToolRegistry()
    registry.register(EchoTool())

    result = await registry.execute(
        ToolCall(tool_name="echo", arguments={"wrong_field": 42})
    )

    assert result.success is False
    assert result.error is not None
    assert "Invalid arguments" in result.error


@pytest.mark.asyncio
async def test_timeout_fails_without_raising() -> None:
    registry = ToolRegistry(default_timeout_s=0.05)
    registry.register(SleepTool())

    result = await registry.execute(
        ToolCall(tool_name="sleep", arguments={"seconds": 5})
    )

    assert result.success is False
    assert result.error is not None
    assert "timed out" in result.error


def test_duplicate_registration_rejected() -> None:
    registry = ToolRegistry()
    registry.register(EchoTool())
    with pytest.raises(ToolAlreadyRegisteredError):
        registry.register(EchoTool())


def test_get_unknown_tool_raises() -> None:
    with pytest.raises(ToolNotFoundError):
        ToolRegistry().get("nope")


def test_list_tools_exposes_prompt_ready_specs() -> None:
    registry = ToolRegistry()
    registry.register(EchoTool())

    specs = registry.list_tools()

    assert len(specs) == 1
    assert specs[0]["name"] == "echo"
    assert "description" in specs[0]
    assert specs[0]["input_schema"]["type"] == "object"
    assert "message" in specs[0]["input_schema"]["properties"]
