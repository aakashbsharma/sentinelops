"""The hand-rolled agentic loop: PLAN -> ACT -> OBSERVE -> REFLECT -> (replan | terminate).

This is deliberately a state machine with enumerated exit states, not a
`while True` around an LLM call. Every run() exits through exactly one of:

    completed              — the agent finished its job (or answered directly)
    needs_more_data        — finished, but flagged that it lacks information
    max_iterations_reached — iteration guardrail fired
    budget_exceeded        — token-budget guardrail fired
    error                  — unhandled exception, converted to data

Subclasses (Stage 4: Triage / Diagnostician / Planner / Executor) provide the
persona, the domain-specific termination test, and the payload extraction.
The control flow itself lives here, once, and is domain-agnostic.
"""

import json
from abc import ABC, abstractmethod
from typing import Any

import structlog

from app.agents.schemas import (
    AgentOutcome,
    AgentStep,
    LoopConfig,
    PlanDecision,
    Reflection,
    ToolCall,
    ToolResult,
)
from app.agents.tools.base import ToolRegistry
from app.observability.trace_bus import publish_step

logger = structlog.get_logger(__name__)


class AgentLoop(ABC):
    """Abstract base for all agents. Subclass and implement the three hooks;
    never override run() — the guardrails live there."""

    #: Subclasses set this; it lands in logs and (later) AgentMessage rows.
    agent_name: str = "agent"

    def __init__(self, config: LoopConfig, tool_registry: ToolRegistry) -> None:
        self.config = config
        self.tools = tool_registry

    # ------------------------------------------------------------------ #
    # Subclass hooks (Stage 4 fills these per agent)
    # ------------------------------------------------------------------ #

    @abstractmethod
    async def build_system_prompt(self, context: dict[str, Any]) -> str:
        """Subclass defines its role/persona/instructions."""

    @abstractmethod
    async def should_terminate(
        self, steps: list[AgentStep], context: dict[str, Any]
    ) -> bool:
        """Domain-specific early termination — e.g. Triage stops once severity
        is assigned with high confidence. Checked after every iteration."""

    @abstractmethod
    def parse_final_output(
        self, steps: list[AgentStep], context: dict[str, Any]
    ) -> dict[str, Any]:
        """Extract the agent's structured payload from the trace.

        Reserved keys the loop lifts onto the AgentOutcome (and removes from
        payload): "summary" (str), "confidence" (float 0-1),
        "needs_more_data" (bool).
        """

    async def accept_final_answer(
        self,
        decision: PlanDecision,
        steps: list[AgentStep],
        context: dict[str, Any],
    ) -> bool:
        """Optional hook: subclasses can REJECT a final answer (e.g. one that
        doesn't parse into their required payload schema). A rejection is
        recorded as an observation so the next PLAN sees why its answer was
        refused and can fix its own formatting — instead of the loop silently
        accepting garbage as 'completed'. Default: accept everything."""
        return True

    # ------------------------------------------------------------------ #
    # The loop — concrete, shared by every agent
    # ------------------------------------------------------------------ #

    async def run(self, context: dict[str, Any]) -> AgentOutcome:
        """Execute the plan/act/observe/reflect loop with all guardrails."""
        steps: list[AgentStep] = []
        total_tokens = 0
        log = logger.bind(agent=self.agent_name)

        # GUARDRAIL: the whole loop is wrapped so an unhandled bug (bad parse,
        # LLM outage, programming error in a subclass hook) becomes an
        # AgentOutcome(status="error") instead of an exception that takes down
        # the orchestrator. One misbehaving agent must never kill the incident
        # response pipeline that other agents are still working on.
        try:
            # GUARDRAIL: bounded iteration. An LLM that keeps asking for "one
            # more tool call" (a common failure mode on ambiguous incidents)
            # would otherwise loop forever, burning tokens and blocking the
            # incident. `for` over a range — not `while True` — makes the
            # bound structural, not conditional.
            for step_number in range(1, self.config.max_iterations + 1):
                # GUARDRAIL: token budget, checked BEFORE each planning call.
                # Each iteration re-sends the growing trace, so cost per step
                # increases superlinearly with step count — unbounded agent
                # loops are an unbounded bill. We stop before spending, not
                # after.
                if total_tokens >= self.config.token_budget:
                    log.warning(
                        "token_budget_exceeded",
                        total_tokens=total_tokens,
                        budget=self.config.token_budget,
                    )
                    return self._finalize("budget_exceeded", steps, context, total_tokens)

                # --- PLAN ---------------------------------------------- #
                # One LLM call: given persona + trace + tools, choose either
                # a tool call or a final answer.
                decision, plan_tokens = await self._plan_step(context, steps)
                total_tokens += plan_tokens

                step = AgentStep(step_number=step_number, thought=decision.thought)

                # --- ACT + OBSERVE -------------------------------------- #
                if decision.tool_call is not None:
                    step.action = decision.tool_call
                    # execute() never raises: unknown tool, bad arguments,
                    # timeout, or a tool bug all come back as a failed
                    # ToolResult. The failure is recorded as the observation
                    # so the next PLAN sees it and can route around it.
                    result = await self.tools.execute(
                        decision.tool_call, timeout_s=self.config.tool_timeout_s
                    )
                    step.observation = self._render_tool_result(result)
                    log.info(
                        "tool_executed",
                        step=step_number,
                        tool=decision.tool_call.tool_name,
                        success=result.success,
                        latency_ms=round(result.latency_ms, 1),
                    )

                # --- REFLECT -------------------------------------------- #
                # A second, cheap LLM call that asks: "given what you just
                # observed, is the plan still on track — and do you have
                # enough information?" This is the mechanism that lets an
                # agent catch its own mistake (e.g. a log query that returned
                # nothing useful) instead of barreling forward and building
                # conclusions on bad data. Costs one extra call per step;
                # config.reflect_every_step turns it off for cheap agents.
                if self.config.reflect_every_step:
                    reflection, reflect_tokens = await self._reflect_step(
                        context, steps, step
                    )
                    total_tokens += reflect_tokens
                    step.reflection = reflection.critique
                    # Always assign (not just set-on-True): the flag tracks
                    # the LATEST reflection — an early "need more data" that
                    # a later tool call satisfied must not stick forever.
                    context["_needs_more_data"] = reflection.needs_more_data

                steps.append(step)
                # Live trace (Stage 7): each COMPLETED step goes out on the
                # Redis trace bus as it happens — publishing only when the
                # whole run finishes would stream in stage-sized bursts.
                # Fire-and-forget; a down Redis never touches the loop.
                await publish_step(self.agent_name, step)

                # --- TERMINATE? ------------------------------------------ #
                # Two ways out per iteration: the LLM produced a final answer
                # (no tool call) that the subclass ACCEPTS, or the subclass's
                # domain test says "done" (e.g. severity assigned with high
                # confidence). Explicit termination conditions are what make
                # this a state machine rather than a hopeful while-loop.
                if decision.is_final:
                    if await self.accept_final_answer(decision, steps, context):
                        log.info("terminated_final_answer", step=step_number)
                        return self._finalize("completed", steps, context, total_tokens)
                    # Rejected final answer: record why and keep looping so
                    # the next PLAN can correct its output format.
                    step.observation = (
                        "[final answer rejected] Output did not parse into the "
                        "required payload schema described in your instructions. "
                        "Re-emit your answer as a single valid JSON object."
                    )
                    log.info("final_answer_rejected", step=step_number)
                elif await self.should_terminate(steps, context):
                    log.info("terminated_by_subclass", step=step_number)
                    return self._finalize("completed", steps, context, total_tokens)

            # Loop ran out of iterations without a termination signal.
            log.warning("max_iterations_reached", iterations=self.config.max_iterations)
            return self._finalize("max_iterations_reached", steps, context, total_tokens)

        except Exception as exc:  # noqa: BLE001 — deliberate: see guardrail note above
            log.exception("agent_loop_error")
            return AgentOutcome(
                status="error",
                summary=f"Agent loop failed: {type(exc).__name__}: {exc}",
                confidence=0.0,
                needs_more_data=False,
                payload={"exception_type": type(exc).__name__},
                steps=steps,  # partial trace preserved for debugging
                total_tokens_used=total_tokens,
            )

    # ------------------------------------------------------------------ #
    # Loop internals
    # ------------------------------------------------------------------ #

    def _finalize(
        self,
        status: str,
        steps: list[AgentStep],
        context: dict[str, Any],
        total_tokens: int,
    ) -> AgentOutcome:
        """Assemble the AgentOutcome for any non-error exit path.

        parse_final_output() runs even for guardrail exits so a partially
        complete agent still reports whatever it managed to establish.
        """
        payload = self.parse_final_output(steps, context)
        summary = str(payload.pop("summary", f"{self.agent_name} finished: {status}"))
        confidence = float(payload.pop("confidence", 0.0))
        needs_more_data = bool(
            payload.pop("needs_more_data", context.get("_needs_more_data", False))
        )

        # A clean completion that still lacks data gets its own status so the
        # orchestrator (Stage 5) can route for enrichment instead of trusting
        # a low-information answer.
        if status == "completed" and needs_more_data:
            status = "needs_more_data"

        return AgentOutcome(
            status=status,  # type: ignore[arg-type]
            summary=summary,
            confidence=max(0.0, min(1.0, confidence)),
            needs_more_data=needs_more_data,
            payload=payload,
            steps=steps,
            total_tokens_used=total_tokens,
        )

    async def _plan_step(
        self, context: dict[str, Any], prior_steps: list[AgentStep]
    ) -> tuple[PlanDecision, int]:
        """One PLAN call: build the prompt, call the LLM, parse the decision.

        Returns (decision, tokens_spent) — exact counts from the API's usage
        field when available, chars/4 estimate as fallback.
        """
        system = await self.build_system_prompt(context)
        messages = self._render_trace_as_messages(context, prior_steps)
        tools = self.tools.list_tools()

        response = await self._call_llm(system=system, messages=messages, tools=tools)
        decision = self._parse_plan_response(response)

        tokens = self._tokens_from_response(response)
        if tokens == 0:
            tokens = (
                self._estimate_tokens(system)
                + self._estimate_tokens(json.dumps(messages, default=str))
                + self._estimate_tokens(decision.thought)
            )
        return decision, tokens

    async def _reflect_step(
        self,
        context: dict[str, Any],
        prior_steps: list[AgentStep],
        current_step: AgentStep,
    ) -> tuple[Reflection, int]:
        """One REFLECT call: ask the agent to self-critique the last observation."""
        system = await self.build_system_prompt(context)
        messages = self._render_trace_as_messages(context, [*prior_steps, current_step])
        messages.append(
            {
                "role": "user",
                "content": (
                    "Reflect on your last step. Given this observation, is your "
                    "plan still on track? Do you have enough information, or do "
                    "you need another tool call? Reply with JSON: "
                    '{"on_track": true|false, "critique": "<1-3 sentences>", '
                    '"needs_more_data": true|false}'
                ),
            }
        )
        # Reflection deliberately gets NO tools: it must evaluate, not act.
        # It's also capped at fewer output tokens than planning — a critique
        # is a couple of sentences, not an essay.
        response = await self._call_llm(
            system=system, messages=messages, tools=[], max_tokens=512
        )
        reflection = self._parse_reflection_response(response)

        tokens = self._tokens_from_response(response)
        if tokens == 0:
            tokens = self._estimate_tokens(
                json.dumps(messages, default=str)
            ) + self._estimate_tokens(reflection.critique)
        return reflection, tokens

    async def _call_llm(
        self,
        *,
        system: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        max_tokens: int = 2048,
    ) -> dict[str, Any]:
        """The single LLM boundary: Anthropic Messages API with native tool
        use. Returns the normalized dict shape the parsers consume (see
        app/agents/llm.py). Unit tests never reach this — they inject
        scripted _plan_step/_reflect_step results instead."""
        from app.agents.llm import call_anthropic, get_anthropic_client
        from app.agents.groq_client import call_groq, get_groq_client

        #client = get_anthropic_client()  # raises LLMConfigurationError if no key
        # return await call_anthropic(
        #     client,
        #     model=self.config.model,
        #     system=system,
        #     messages=messages,
        #     tools=tools,
        #     max_tokens=max_tokens,
        # )

        client = get_groq_client()
        return await call_groq(
            client,
            model=self.config.model,
            system=system,
            messages=messages,
            tools=tools,
            max_tokens=max_tokens,
        )

    @staticmethod
    def _tokens_from_response(response: dict[str, Any]) -> int:
        """Exact usage from the API response; 0 if absent (caller falls back
        to the estimate)."""
        usage = response.get("usage") or {}
        return int(usage.get("input_tokens", 0)) + int(usage.get("output_tokens", 0))

    # ------------------------------------------------------------------ #
    # Parsing / rendering helpers (real plumbing, LLM-format-aware)
    # ------------------------------------------------------------------ #

    def _parse_plan_response(self, response: dict[str, Any]) -> PlanDecision:
        """Parse an Anthropic-shaped response into a PlanDecision.

        text blocks -> thought / final answer; the first tool_use block ->
        ToolCall. No tool_use block means the agent answered directly.
        """
        thought_parts: list[str] = []
        tool_call: ToolCall | None = None

        for block in response.get("content", []):
            if block.get("type") == "text":
                thought_parts.append(block.get("text", ""))
            elif block.get("type") == "tool_use" and tool_call is None:
                tool_call = ToolCall(
                    tool_name=block["name"],
                    arguments=block.get("input", {}),
                    call_id=block.get("id", ToolCall(tool_name=block["name"]).call_id),
                )

        thought = "\n".join(part for part in thought_parts if part).strip()
        return PlanDecision(
            thought=thought or "(no reasoning text returned)",
            tool_call=tool_call,
            final_answer=thought if tool_call is None else None,
        )

    def _parse_reflection_response(self, response: dict[str, Any]) -> Reflection:
        """Parse the reflection JSON; fall back to raw text if the LLM didn't
        comply with the format (never crash the loop over a malformed reflection)."""
        text = "".join(
            block.get("text", "")
            for block in response.get("content", [])
            if block.get("type") == "text"
        ).strip()

        try:
            data = json.loads(text)
            return Reflection(
                on_track=bool(data.get("on_track", True)),
                critique=str(data.get("critique", text)),
                needs_more_data=bool(data.get("needs_more_data", False)),
            )
        except (json.JSONDecodeError, TypeError):
            return Reflection(critique=text or "(empty reflection)")

    def _render_trace_as_messages(
        self, context: dict[str, Any], steps: list[AgentStep]
    ) -> list[dict[str, Any]]:
        """Serialize the running trace into chat messages for the next LLM call.

        The trace is the agent's working memory within a run: every PLAN sees
        everything that happened before it.
        """
        messages: list[dict[str, Any]] = [
            {
                "role": "user",
                "content": f"Task context:\n{json.dumps(self._public_context(context), default=str, indent=2)}",
            }
        ]
        for step in steps:
            messages.append({"role": "assistant", "content": step.thought})
            if step.action is not None:
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            f"[tool result for {step.action.tool_name}] "
                            f"{step.observation or '(no result)'}"
                        ),
                    }
                )
            if step.reflection:
                messages.append(
                    {"role": "assistant", "content": f"(reflection) {step.reflection}"}
                )
        return messages

    @staticmethod
    def _public_context(context: dict[str, Any]) -> dict[str, Any]:
        """Strip loop-internal keys (underscore-prefixed) before prompting."""
        return {k: v for k, v in context.items() if not k.startswith("_")}

    @staticmethod
    def _render_tool_result(result: ToolResult) -> str:
        if result.success:
            return json.dumps(
                {"success": True, "output": result.output}, default=str
            )
        return json.dumps({"success": False, "error": result.error})

    @staticmethod
    def _estimate_tokens(text: str) -> int:
        """Crude chars/4 heuristic — good enough for budget enforcement until
        _call_llm returns exact usage from the API. Deliberately conservative:
        overcounting trips the budget earlier, which fails safe (cheaper)."""
        return max(1, len(text) // 4)
