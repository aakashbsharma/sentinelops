"""Helpers for extracting typed payloads from LLM answer text (internal).

LLMs are asked to emit a single JSON object, but in practice wrap it in
markdown fences or prose. These helpers tolerate that without ever raising —
a parse failure returns None and the caller decides what to do (usually:
reject the final answer and let the loop retry).
"""

import json
import re
from typing import Any, TypeVar

from pydantic import BaseModel, ValidationError

from app.agents.schemas import AgentStep

T = TypeVar("T", bound=BaseModel)

_FENCED_JSON = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)


def extract_json_object(text: str) -> dict[str, Any] | None:
    """Best-effort extraction of one JSON object from LLM text."""
    match = _FENCED_JSON.search(text)
    if match:
        try:
            data = json.loads(match.group(1))
            if isinstance(data, dict):
                return data
        except json.JSONDecodeError:
            pass

    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end > start:
        try:
            data = json.loads(text[start : end + 1])
            if isinstance(data, dict):
                return data
        except json.JSONDecodeError:
            pass
    return None


def parse_payload(text: str, model_cls: type[T]) -> T | None:
    """Extract a JSON object from text and validate it as model_cls."""
    data = extract_json_object(text)
    if data is None:
        return None
    try:
        return model_cls(**data)
    except ValidationError:
        return None


def extract_latest_payload(steps: list[AgentStep], model_cls: type[T]) -> T | None:
    """Scan the trace newest-first for a thought containing a valid payload.

    Newest-first matters: an agent may emit a draft payload early, gather
    more evidence, and emit a corrected one — the correction wins.
    """
    for step in reversed(steps):
        payload = parse_payload(step.thought, model_cls)
        if payload is not None:
            return payload
    return None
