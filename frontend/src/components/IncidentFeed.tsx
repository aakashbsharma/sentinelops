/**
 * Incident feed: polls GET /incidents every 5s via react-query.
 *
 * Polling over WebSocket here on purpose: the trace socket is scoped to ONE
 * incident's steps and has no incident-level lifecycle events, so reusing it
 * for the feed would mean inventing a new server-side event stream — a 5s
 * refetchInterval is one line and plenty fresh for a triage feed.
 */
import { useQuery } from "@tanstack/react-query"
import { formatDistanceToNow } from "date-fns"
import { useState } from "react"

import { SeverityBadge, StatusBadge } from "@/components/StatusBadges"
import { Button } from "@/components/ui/button"
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table"
import { fetchIncidents } from "@/lib/api"
import { cn } from "@/lib/utils"

const PAGE_SIZE = 20

interface Props {
  selectedId: string | null
  onSelect: (id: string) => void
}

export function IncidentFeed({ selectedId, onSelect }: Props) {
  // Simplest pagination that works: grow the limit, single query. At demo
  // data volumes re-fetching the grown window is irrelevant.
  const [limit, setLimit] = useState(PAGE_SIZE)
  const { data, isPending, isError } = useQuery({
    queryKey: ["incidents", limit],
    queryFn: () => fetchIncidents(limit),
    refetchInterval: 5000,
  })

  if (isPending) {
    return <p className="p-4 text-sm text-muted-foreground">Loading incidents…</p>
  }
  if (isError) {
    return (
      <p className="p-4 text-sm text-destructive">
        Could not reach the API — is the backend running?
      </p>
    )
  }
  if (data.items.length === 0) {
    return (
      <p className="p-4 text-sm text-muted-foreground">
        No incidents yet. Seed one with{" "}
        <code className="rounded bg-muted px-1 py-0.5">
          python scripts/seed_demo_incident.py
        </code>
      </p>
    )
  }

  return (
    <div className="flex flex-col gap-2">
      <Table>
        <TableHeader>
          <TableRow>
            <TableHead>Incident</TableHead>
            <TableHead>Severity</TableHead>
            <TableHead>Status</TableHead>
            <TableHead className="text-right">Created</TableHead>
          </TableRow>
        </TableHeader>
        <TableBody>
          {data.items.map((incident) => (
            <TableRow
              key={incident.id}
              onClick={() => onSelect(incident.id)}
              className={cn(
                "cursor-pointer",
                incident.id === selectedId && "bg-muted"
              )}
            >
              <TableCell className="max-w-96 truncate font-medium">
                {incident.title}
              </TableCell>
              <TableCell>
                <SeverityBadge severity={incident.severity} />
              </TableCell>
              <TableCell>
                <StatusBadge status={incident.status} />
              </TableCell>
              <TableCell className="text-right text-muted-foreground">
                {formatDistanceToNow(new Date(incident.created_at), {
                  addSuffix: true,
                })}
              </TableCell>
            </TableRow>
          ))}
        </TableBody>
      </Table>
      {data.total > data.items.length && limit < 100 && (
        <Button
          variant="outline"
          size="sm"
          className="self-center"
          onClick={() => setLimit((l) => Math.min(l + PAGE_SIZE, 100))} // API caps limit at 100
        >
          Load more ({data.items.length} of {data.total})
        </Button>
      )}
    </div>
  )
}
