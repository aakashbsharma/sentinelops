/**
 * Approval dialog — the human-in-the-loop gate made visible.
 *
 * Shown when an incident is awaiting_approval: the planner PROPOSED actions,
 * the code guardrail decided a human must sign off, and nothing executes
 * until this dialog's Approve button (or the API equivalent) is pressed.
 * Rejecting closes the incident without executing anything.
 */
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query"
import { ShieldAlertIcon, UndoIcon } from "lucide-react"
import { useState } from "react"

import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "@/components/ui/dialog"
import { fetchPendingApproval, submitApproval } from "@/lib/api"
import { cn } from "@/lib/utils"

const RISK_CLASSES: Record<string, string> = {
  low: "bg-emerald-100 text-emerald-900 dark:bg-emerald-950 dark:text-emerald-200",
  medium: "bg-amber-100 text-amber-900 dark:bg-amber-950 dark:text-amber-200",
  high: "bg-orange-500 text-white",
  critical: "bg-red-600 text-white",
}

function RiskBadge({ level }: { level: string }) {
  return (
    <Badge
      variant="secondary"
      className={cn("uppercase tracking-wide", RISK_CLASSES[level] ?? "")}
    >
      {level} risk
    </Badge>
  )
}

export function ApprovalDialog({ incidentId }: { incidentId: string }) {
  const [open, setOpen] = useState(false)
  const [resolvedBy, setResolvedBy] = useState("operator@dashboard")
  const queryClient = useQueryClient()

  const { data: approval, isPending } = useQuery({
    queryKey: ["approval", incidentId],
    queryFn: () => fetchPendingApproval(incidentId),
    retry: false, // a 404 just means nothing is pending anymore
  })

  const decide = useMutation({
    mutationFn: (approved: boolean) =>
      submitApproval(incidentId, approved, resolvedBy.trim() || "operator@dashboard"),
    onSuccess: () => {
      setOpen(false)
      // The decision changes the incident's status and (on approve) starts
      // the executor — refresh everything that watches this incident.
      void queryClient.invalidateQueries({ queryKey: ["incident", incidentId] })
      void queryClient.invalidateQueries({ queryKey: ["incidents"] })
      void queryClient.invalidateQueries({ queryKey: ["approval", incidentId] })
    },
  })

  if (isPending || !approval) return null
  const plan = approval.proposed_action

  return (
    <Dialog open={open} onOpenChange={setOpen}>
      <DialogTrigger
        render={
          <Button className="w-full" variant="destructive">
            <ShieldAlertIcon />
            Review proposed remediation
          </Button>
        }
      />
      <DialogContent className="sm:max-w-lg">
        <DialogHeader>
          <DialogTitle className="flex items-center gap-2">
            Human approval required
            <RiskBadge level={approval.risk_level} />
          </DialogTitle>
          <DialogDescription>
            The remediation planner proposed the actions below, and the
            guardrail requires a human decision before anything executes.
            Approving hands the plan to the executor; rejecting closes the
            incident without touching infrastructure.
          </DialogDescription>
        </DialogHeader>

        <div className="flex max-h-80 flex-col gap-3 overflow-y-auto">
          {plan.proposed_actions.map((action, i) => (
            <div key={i} className="rounded-lg border p-3 text-sm">
              <div className="flex flex-wrap items-center gap-1.5">
                <Badge variant="outline">{action.action_type}</Badge>
                <code className="rounded bg-muted px-1.5 py-0.5 text-xs font-medium">
                  {action.tool_name}
                </code>
                <span className="text-xs text-muted-foreground">
                  → {action.target}
                </span>
              </div>
              {Object.keys(action.parameters).length > 0 && (
                <pre className="mt-2 overflow-x-auto rounded bg-muted/60 px-2 py-1.5 text-xs">
                  {JSON.stringify(action.parameters, null, 2)}
                </pre>
              )}
              <p className="mt-2 text-xs leading-relaxed text-muted-foreground">
                {action.rationale}
              </p>
            </div>
          ))}

          {plan.rollback_plan && (
            <div className="flex gap-2 rounded-lg border border-dashed p-3 text-xs text-muted-foreground">
              <UndoIcon className="mt-0.5 size-3.5 shrink-0" />
              <p>
                <span className="font-medium text-foreground">Rollback: </span>
                {plan.rollback_plan}
              </p>
            </div>
          )}

          <label className="flex flex-col gap-1 text-xs">
            <span className="font-medium">Deciding as</span>
            <input
              value={resolvedBy}
              onChange={(e) => setResolvedBy(e.target.value)}
              className="h-8 rounded-lg border bg-background px-2.5 text-sm outline-none focus-visible:ring-3 focus-visible:ring-ring/50"
            />
          </label>

          {decide.isError && (
            <p className="text-xs text-destructive">
              Decision failed — the incident may have changed state. Refresh and
              try again.
            </p>
          )}
        </div>

        <DialogFooter>
          <Button
            variant="outline"
            disabled={decide.isPending}
            onClick={() => decide.mutate(false)}
          >
            Reject &amp; close
          </Button>
          <Button
            disabled={decide.isPending}
            onClick={() => decide.mutate(true)}
          >
            {decide.isPending ? "Submitting…" : "Approve & execute"}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  )
}
