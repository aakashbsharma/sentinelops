/**
 * Live trace panel — the proof that the agents are really reasoning.
 *
 * Renders every AgentStep from /ws/incidents/{id}/trace as a vertical
 * timeline: the agent's PLAN (thought), the tool it chose (action), what
 * came back (observation), and its self-critique (reflection). History is
 * replayed on connect; new steps appear live while the orchestrator runs.
 */
import {
  BrainIcon,
  ChevronRightIcon,
  MessageSquareQuoteIcon,
  WrenchIcon,
} from "lucide-react"
import { useEffect, useMemo, useRef } from "react"

import { Badge } from "@/components/ui/badge"
import { useTraceSocket, type SocketStatus } from "@/hooks/useTraceSocket"
import type { TraceStep } from "@/lib/api"
import { cn } from "@/lib/utils"

// One accent per agent so a glance shows the baton passing between them.
const AGENT_STYLES: Record<string, { dot: string; label: string }> = {
  triage: { dot: "bg-sky-500", label: "text-sky-700 dark:text-sky-300" },
  diagnostician: {
    dot: "bg-indigo-500",
    label: "text-indigo-700 dark:text-indigo-300",
  },
  remediation_planner: {
    dot: "bg-violet-500",
    label: "text-violet-700 dark:text-violet-300",
  },
  executor: {
    dot: "bg-emerald-500",
    label: "text-emerald-700 dark:text-emerald-300",
  },
}
const DEFAULT_STYLE = { dot: "bg-muted-foreground", label: "text-foreground" }

function agentStyle(name: string) {
  return AGENT_STYLES[name] ?? DEFAULT_STYLE
}

/** Observations are serialized ToolResults: {"success": bool, ...}. */
function parseObservation(
  raw: string
): { success: boolean | null; pretty: string } {
  try {
    const parsed = JSON.parse(raw)
    return {
      success: typeof parsed.success === "boolean" ? parsed.success : null,
      pretty: JSON.stringify(parsed.output ?? parsed.error ?? parsed, null, 2),
    }
  } catch {
    return { success: null, pretty: raw }
  }
}

function StatusPill({ status }: { status: SocketStatus }) {
  const config: Record<SocketStatus, { className: string; text: string }> = {
    live: {
      className:
        "bg-emerald-100 text-emerald-900 dark:bg-emerald-950 dark:text-emerald-200",
      text: "live",
    },
    connecting: {
      className: "bg-amber-100 text-amber-900 dark:bg-amber-950 dark:text-amber-200",
      text: "connecting",
    },
    closed: { className: "bg-muted text-muted-foreground", text: "offline" },
  }
  const { className, text } = config[status]
  return (
    <Badge variant="secondary" className={className}>
      {status === "live" && (
        <span className="relative flex size-1.5">
          <span className="absolute inline-flex size-full animate-ping rounded-full bg-current opacity-60" />
          <span className="relative inline-flex size-1.5 rounded-full bg-current" />
        </span>
      )}
      {text}
    </Badge>
  )
}

function StepEntry({ step }: { step: TraceStep }) {
  const style = agentStyle(step.agent_name)
  const observation =
    step.observation !== null ? parseObservation(step.observation) : null

  return (
    <li className="relative pb-6 pl-6 last:pb-2">
      {/* timeline rail + agent-colored dot */}
      <span className="absolute top-1.5 left-0 flex size-3 items-center justify-center">
        <span className={cn("size-2.5 rounded-full", style.dot)} />
      </span>
      <span className="absolute top-5 bottom-0 left-[5px] w-px bg-border" />

      <div className="flex flex-wrap items-baseline gap-x-2 gap-y-0.5">
        <span className={cn("text-sm font-semibold", style.label)}>
          {step.agent_name.replaceAll("_", " ")}
        </span>
        <span className="text-xs text-muted-foreground">
          step {step.step_number}
          {" · "}
          {new Date(step.timestamp).toLocaleTimeString()}
        </span>
      </div>

      {/* PLAN */}
      <p className="mt-1 text-sm leading-relaxed whitespace-pre-wrap">
        {step.thought}
      </p>

      {/* ACT */}
      {step.action && (
        <div className="mt-2 flex flex-wrap items-center gap-1.5 text-xs">
          <WrenchIcon className="size-3.5 text-muted-foreground" />
          <code className="rounded bg-muted px-1.5 py-0.5 font-medium">
            {step.action.tool_name}
          </code>
          <code className="max-w-full truncate text-muted-foreground">
            {JSON.stringify(step.action.arguments)}
          </code>
        </div>
      )}

      {/* OBSERVE — collapsible, tool output can be long */}
      {observation && (
        <details className="group mt-2 rounded-lg border bg-muted/40 text-xs">
          <summary className="flex cursor-pointer items-center gap-1.5 px-2.5 py-1.5 select-none">
            <ChevronRightIcon className="size-3.5 transition-transform group-open:rotate-90" />
            <span className="font-medium">observation</span>
            {observation.success !== null && (
              <span
                className={cn(
                  "rounded px-1 py-px font-medium",
                  observation.success
                    ? "bg-emerald-100 text-emerald-900 dark:bg-emerald-950 dark:text-emerald-200"
                    : "bg-red-100 text-red-900 dark:bg-red-950 dark:text-red-200"
                )}
              >
                {observation.success ? "success" : "failed"}
              </span>
            )}
          </summary>
          <pre className="max-h-56 overflow-auto border-t px-2.5 py-2 whitespace-pre-wrap">
            {observation.pretty}
          </pre>
        </details>
      )}

      {/* REFLECT */}
      {step.reflection && (
        <div className="mt-2 flex gap-1.5 text-xs text-muted-foreground italic">
          <MessageSquareQuoteIcon className="mt-0.5 size-3.5 shrink-0" />
          <p className="leading-relaxed">{step.reflection}</p>
        </div>
      )}
    </li>
  )
}

export function TracePanel({ incidentId }: { incidentId: string }) {
  const { steps, status } = useTraceSocket(incidentId)
  const bottomRef = useRef<HTMLDivElement>(null)
  const container = useRef<HTMLDivElement>(null)

  // Group consecutive steps by run so a resumed incident reads as two acts.
  const runs = useMemo(() => {
    const groups: { runId: string | null; steps: TraceStep[] }[] = []
    for (const step of steps) {
      const last = groups[groups.length - 1]
      if (last && last.runId === step.run_id) last.steps.push(step)
      else groups.push({ runId: step.run_id, steps: [step] })
    }
    return groups
  }, [steps])

  // Follow the stream only when the user is already near the bottom —
  // never yank the scroll away from someone reading an earlier step.
  useEffect(() => {
    const el = container.current
    if (!el) return
    const nearBottom = el.scrollHeight - el.scrollTop - el.clientHeight < 120
    if (nearBottom) bottomRef.current?.scrollIntoView({ behavior: "smooth" })
  }, [steps])

  return (
    <div className="flex h-full min-h-0 flex-col">
      <div className="flex items-center justify-between gap-2 border-b px-4 py-2.5">
        <div className="flex items-center gap-2">
          <BrainIcon className="size-4 text-muted-foreground" />
          <h2 className="text-sm font-semibold">Agent trace</h2>
        </div>
        <StatusPill status={status} />
      </div>

      <div ref={container} className="min-h-0 flex-1 overflow-y-auto px-4 py-3">
        {steps.length === 0 ? (
          <p className="py-8 text-center text-sm text-muted-foreground">
            {status === "connecting"
              ? "Connecting to trace stream…"
              : "No agent steps yet — they will appear here live as the agents reason."}
          </p>
        ) : (
          runs.map((run, i) => (
            <section key={run.runId ?? i}>
              {runs.length > 1 && (
                <p className="mb-2 text-xs font-medium tracking-wide text-muted-foreground uppercase">
                  run {i + 1}
                  {run.runId && (
                    <span className="ml-1 normal-case">({run.runId.slice(0, 8)})</span>
                  )}
                </p>
              )}
              <ol>
                {run.steps.map((step) => (
                  <StepEntry
                    key={`${step.run_id}|${step.agent_name}|${step.step_number}|${step.timestamp}`}
                    step={step}
                  />
                ))}
              </ol>
            </section>
          ))
        )}
        <div ref={bottomRef} />
      </div>
    </div>
  )
}
