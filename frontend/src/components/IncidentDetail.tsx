/**
 * Side panel for a selected incident: summary header (severity, status,
 * cost), the approval gate when one is pending, and the live trace panel.
 */
import { useQuery } from "@tanstack/react-query"

import { ApprovalDialog } from "@/components/ApprovalDialog"
import { SeverityBadge, StatusBadge } from "@/components/StatusBadges"
import { TracePanel } from "@/components/TracePanel"
import { fetchIncident } from "@/lib/api"

export function IncidentDetail({ incidentId }: { incidentId: string }) {
  const { data: incident } = useQuery({
    queryKey: ["incident", incidentId],
    queryFn: () => fetchIncident(incidentId),
    // Status and cost move while the orchestrator works; keep them fresh.
    refetchInterval: 5000,
  })

  if (!incident) {
    return (
      <p className="p-4 text-sm text-muted-foreground">Loading incident…</p>
    )
  }

  const cost = incident.cost_summary

  return (
    <div className="flex h-full min-h-0 flex-col">
      <div className="flex flex-col gap-2 border-b px-4 py-3">
        <h1 className="text-sm leading-snug font-semibold">{incident.title}</h1>
        <div className="flex flex-wrap items-center gap-1.5">
          <SeverityBadge severity={incident.severity} />
          <StatusBadge status={incident.status} />
          {cost.total_tokens > 0 && (
            <span
              className="text-xs text-muted-foreground"
              title="Illustrative pricing — not billing-accurate"
            >
              {cost.total_tokens.toLocaleString()} tokens · ~$
              {cost.estimated_cost_usd.toFixed(4)}
            </span>
          )}
        </div>
        {incident.description && (
          <p className="text-xs leading-relaxed text-muted-foreground">
            {incident.description}
          </p>
        )}
        {incident.status === "awaiting_approval" && (
          <ApprovalDialog incidentId={incidentId} />
        )}
      </div>

      {/* key: remount per incident so the socket + step state start fresh */}
      <TracePanel key={incidentId} incidentId={incidentId} />
    </div>
  )
}
