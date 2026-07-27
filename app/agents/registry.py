"""Role-name -> agent-class mapping for dynamic instantiation (Stage 5)."""

from app.agents.base import AgentLoop
from app.agents.diagnostician import DiagnosticianAgent
from app.agents.executor import ExecutorAgent
from app.agents.remediation_planner import RemediationPlannerAgent
from app.agents.triage import TriageAgent

AGENT_REGISTRY: dict[str, type[AgentLoop]] = {
    "triage": TriageAgent,
    "diagnostician": DiagnosticianAgent,
    "remediation_planner": RemediationPlannerAgent,
    "executor": ExecutorAgent,
}


def get_agent_class(role: str) -> type[AgentLoop]:
    try:
        return AGENT_REGISTRY[role]
    except KeyError:
        raise KeyError(
            f"Unknown agent role {role!r}. Known roles: {sorted(AGENT_REGISTRY)}"
        ) from None
