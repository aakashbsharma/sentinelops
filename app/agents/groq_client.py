"""Groq client wiring — mirrors llm.py's interface exactly so it's a
drop-in substitute. Same function names, same return shape:
{"content": [...], "stop_reason": ..., "usage": {...}}.

Nothing downstream (AgentLoop, tests, ToolRegistry) needs to know which
provider is active — only the import at the call site changes.
"""

import json
from functools import lru_cache
from typing import Any

from openai import (
    APIConnectionError,
    APITimeoutError,
    InternalServerError,
    RateLimitError,
)
from openai import AsyncOpenAI
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from app.core.config import get_settings


class LLMConfigurationError(RuntimeError):
    """Raised when the LLM cannot be used due to missing configuration."""


@lru_cache
def get_groq_client() -> AsyncOpenAI:
    """Cached client factory — mirrors get_anthropic_client()."""
    settings = get_settings()
    if not settings.GROQ_API_KEY:
        raise LLMConfigurationError(
            "GROQ_API_KEY is not set. Add it to .env. "
            "Unit tests don't need it — they inject scripted plan/reflect steps."
        )
    return AsyncOpenAI(
        api_key=settings.GROQ_API_KEY,
        base_url="https://api.groq.com/openai/v1",
    )


_RETRYABLE_ERRORS = (
    RateLimitError,
    APITimeoutError,
    APIConnectionError,
    InternalServerError,
)


@retry(
    reraise=True,
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=8),
    retry=retry_if_exception_type(_RETRYABLE_ERRORS),
)
async def _send_with_retry(client: AsyncOpenAI, **kwargs: Any) -> Any:
    return await client.chat.completions.create(**kwargs)


def _to_openai_tools(anthropic_tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Reshape Anthropic-format tool schemas -> OpenAI-format.

    Anthropic: {"name", "description", "input_schema"}
    OpenAI:    {"type": "function", "function": {"name", "description", "parameters"}}

    ToolRegistry.list_tools() emits Anthropic format (per llm.py's comment) —
    this function is the only place that needs to know the difference.
    """
    reshaped = []
    for tool in anthropic_tools:
        reshaped.append(
            {
                "type": "function",
                "function": {
                    "name": tool["name"],
                    "description": tool.get("description", ""),
                    "parameters": tool.get("input_schema", {"type": "object", "properties": {}}),
                },
            }
        )
    return reshaped


async def call_groq(
    client: AsyncOpenAI,
    *,
    model: str,
    system: str,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
    max_tokens: int = 2048,
) -> dict[str, Any]:
    """Call Groq's chat completions API and normalize to the SAME plain-dict
    shape call_anthropic() returns. This is what makes the swap invisible
    to AgentLoop.
    """
    # OpenAI-style: system is a message, not a separate top-level param.
    full_messages = [{"role": "system", "content": system}, *messages]

    kwargs: dict[str, Any] = {
        "model": model,
        "messages": full_messages,
        "max_tokens": max_tokens,
    }
    if tools:
        kwargs["tools"] = _to_openai_tools(tools)

    response = await _send_with_retry(client, **kwargs)
    choice = response.choices[0]
    msg = choice.message

    content: list[dict[str, Any]] = []

    if msg.content:
        content.append({"type": "text", "text": msg.content})

    if msg.tool_calls:
        for tc in msg.tool_calls:
            # Groq/OpenAI gives arguments as a JSON string — must parse.
            try:
                parsed_input = json.loads(tc.function.arguments)
            except json.JSONDecodeError:
                # Malformed args from the model — surface as empty dict rather
                # than crashing the loop; the agent's own validation will
                # reject an incomplete tool call same as it would elsewhere.
                parsed_input = {}
            content.append(
                {
                    "type": "tool_use",
                    "id": tc.id,
                    "name": tc.function.name,
                    "input": parsed_input,
                }
            )

    # Normalize OpenAI's stop reasons to Anthropic's vocabulary so
    # AgentLoop's existing stop_reason checks keep working unmodified.
    stop_reason_map = {
        "stop": "end_turn",
        "tool_calls": "tool_use",
        "length": "max_tokens",
    }
    stop_reason = stop_reason_map.get(choice.finish_reason, choice.finish_reason)

    return {
        "content": content,
        "stop_reason": stop_reason,
        "usage": {
            "input_tokens": response.usage.prompt_tokens,
            "output_tokens": response.usage.completion_tokens,
        },
    }