"""Anthropic client wiring — the only module that talks to the API.

Retry policy: transient failures (rate limits, timeouts, connection drops,
5xx) are retried 3x with exponential backoff. Auth and validation errors are
NOT retried — retrying a bad API key or a malformed request just burns three
attempts to learn the same thing; fail fast so the misconfiguration is
visible immediately.
"""

from functools import lru_cache
from typing import Any

import anthropic
from anthropic import AsyncAnthropic
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
def get_anthropic_client() -> AsyncAnthropic:
    """Cached client factory.

    Call this once at startup (the Stage 5 orchestrator does) so a missing
    key fails the boot with a clear message, instead of surfacing as a
    confusing mid-loop 'error' AgentOutcome on the first real incident.
    """
    settings = get_settings()
    if not settings.ANTHROPIC_API_KEY:
        raise LLMConfigurationError(
            "ANTHROPIC_API_KEY is not set. Add it to .env (see .env.example). "
            "Unit tests don't need it — they inject scripted plan/reflect steps."
        )
    return AsyncAnthropic(api_key=settings.ANTHROPIC_API_KEY)


# Transient by nature; safe to retry.
_RETRYABLE_ERRORS = (
    anthropic.RateLimitError,
    anthropic.APITimeoutError,
    anthropic.APIConnectionError,
    anthropic.InternalServerError,
)


@retry(
    reraise=True,
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=8),
    retry=retry_if_exception_type(_RETRYABLE_ERRORS),
)
async def _send_with_retry(client: AsyncAnthropic, **kwargs: Any) -> Any:
    return await client.messages.create(**kwargs)


async def call_anthropic(
    client: AsyncAnthropic,
    *,
    model: str,
    system: str,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
    max_tokens: int = 2048,
) -> dict[str, Any]:
    """Call the Messages API and normalize the response to the plain-dict
    shape AgentLoop's parsers consume:
    {"content": [...text/tool_use blocks...], "stop_reason": ..., "usage": {...}}.

    Keeping the SDK types contained here means the loop (and its tests) never
    depend on anthropic objects.
    """
    kwargs: dict[str, Any] = {
        "model": model,
        "system": system,
        "messages": messages,
        "max_tokens": max_tokens,
    }
    if tools:
        # ToolRegistry.list_tools() already emits Anthropic's tool format:
        # {"name", "description", "input_schema" (JSON Schema)}.
        kwargs["tools"] = tools

    response = await _send_with_retry(client, **kwargs)

    content: list[dict[str, Any]] = []
    for block in response.content:
        if block.type == "text":
            content.append({"type": "text", "text": block.text})
        elif block.type == "tool_use":
            content.append(
                {
                    "type": "tool_use",
                    "id": block.id,
                    "name": block.name,
                    "input": block.input,
                }
            )

    return {
        "content": content,
        "stop_reason": response.stop_reason,
        # Real token counts from the API — replaces the chars/4 estimate for
        # budget enforcement whenever a real call was made.
        "usage": {
            "input_tokens": response.usage.input_tokens,
            "output_tokens": response.usage.output_tokens,
        },
    }
