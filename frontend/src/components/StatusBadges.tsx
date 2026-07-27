/**
 * Severity/status badges shared by the feed and the detail panel.
 * Colors are deliberate: severity is "how bad", status is "where in the
 * pipeline" — two different visual languages (saturation vs hue).
 */
import { Badge } from "@/components/ui/badge"
import { cn } from "@/lib/utils"

const SEVERITY_CLASSES: Record<string, string> = {
  critical: "bg-red-600 text-white dark:bg-red-500",
  high: "bg-orange-500 text-white dark:bg-orange-400 dark:text-black",
  medium: "bg-amber-400 text-black",
  low: "bg-emerald-200 text-emerald-950 dark:bg-emerald-900 dark:text-emerald-100",
  unknown: "bg-muted text-muted-foreground",
}

export function SeverityBadge({ severity }: { severity: string }) {
  return (
    <Badge
      variant="secondary"
      className={cn(
        "uppercase tracking-wide",
        SEVERITY_CLASSES[severity] ?? SEVERITY_CLASSES.unknown
      )}
    >
      {severity}
    </Badge>
  )
}

const STATUS_CLASSES: Record<string, string> = {
  open: "bg-sky-100 text-sky-900 dark:bg-sky-950 dark:text-sky-200",
  triaging: "bg-sky-100 text-sky-900 dark:bg-sky-950 dark:text-sky-200",
  diagnosing: "bg-indigo-100 text-indigo-900 dark:bg-indigo-950 dark:text-indigo-200",
  awaiting_approval: "bg-amber-100 text-amber-900 dark:bg-amber-950 dark:text-amber-200",
  remediating: "bg-violet-100 text-violet-900 dark:bg-violet-950 dark:text-violet-200",
  resolved: "bg-emerald-100 text-emerald-900 dark:bg-emerald-950 dark:text-emerald-200",
  escalated: "bg-red-100 text-red-900 dark:bg-red-950 dark:text-red-200",
  closed: "bg-muted text-muted-foreground",
}

/** Statuses where agents are actively working — shown with a pulse dot. */
const ACTIVE_STATUSES = new Set(["open", "triaging", "diagnosing", "remediating"])

export function StatusBadge({ status }: { status: string }) {
  return (
    <Badge
      variant="secondary"
      className={cn(STATUS_CLASSES[status] ?? "bg-muted text-muted-foreground")}
    >
      {ACTIVE_STATUSES.has(status) && (
        <span className="relative flex size-1.5">
          <span className="absolute inline-flex size-full animate-ping rounded-full bg-current opacity-60" />
          <span className="relative inline-flex size-1.5 rounded-full bg-current" />
        </span>
      )}
      {status.replaceAll("_", " ")}
    </Badge>
  )
}
