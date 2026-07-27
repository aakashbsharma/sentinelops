"""Per-incident cost aggregation.

DESIGN: cost is DERIVED ON READ from the persisted AgentMessage rows, not
accumulated into a running-total column. The messages already carry exact
per-invocation token counts (`content["total_tokens_used"]`, from the API's
usage field) and the model that priced them, so aggregating at read time
keeps one source of truth, spans run/resume boundaries for free, and needs
no migration. At this read volume the arithmetic is negligible; if incidents
ever carried thousands of messages, materialize the total then.

Dollar figures come from MODEL_COST_PER_TOKEN in config — an illustrative
blended $/token constant, NOT billing-accurate (see the comment there).
"""

from collections.abc import Iterable

from pydantic import BaseModel, Field

from app.core.config import get_settings
from app.models.agent_message import AgentMessage


class AgentCostBreakdown(BaseModel):
    """Tokens and estimated cost attributed to one agent across all its runs."""

    total_tokens: int = 0
    estimated_cost_usd: float = 0.0
    invocations: int = 0


class CostSummary(BaseModel):
    """What this incident's agent loops cost, approximately."""

    total_tokens: int = 0
    estimated_cost_usd: float = 0.0
    per_agent_breakdown: dict[str, AgentCostBreakdown] = Field(default_factory=dict)


def _rate_for(model: str | None) -> float:
    rates = get_settings().MODEL_COST_PER_TOKEN
    if model is not None and model in rates:
        return rates[model]
    return rates.get("default", 0.0)


def compute_cost_summary(messages: Iterable[AgentMessage]) -> CostSummary:
    """Aggregate token usage from an incident's AgentMessages.

    Orchestrator decision messages carry no token usage (routing is
    deterministic code, not an LLM) — `content.get(...)` returning 0 skips
    them naturally.
    """
    summary = CostSummary()
    for message in messages:
        content = message.content or {}
        tokens = int(content.get("total_tokens_used", 0) or 0)
        if tokens <= 0:
            continue
        cost = tokens * _rate_for(content.get("model"))

        agent = summary.per_agent_breakdown.setdefault(
            message.agent_name, AgentCostBreakdown()
        )
        agent.total_tokens += tokens
        agent.estimated_cost_usd = round(agent.estimated_cost_usd + cost, 6)
        agent.invocations += 1

        summary.total_tokens += tokens
        summary.estimated_cost_usd = round(summary.estimated_cost_usd + cost, 6)
    return summary
