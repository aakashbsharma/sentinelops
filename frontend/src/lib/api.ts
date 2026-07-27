/**
 * API client + types mirroring the FastAPI schemas (app/schemas/incidents.py)
 * and the Stage 7 WebSocket trace frame (app/observability/trace_bus.py).
 */
import axios from "axios"

export const API_URL: string =
  import.meta.env.VITE_API_URL ?? "http://localhost:8000"
export const WS_URL: string =
  import.meta.env.VITE_WS_URL ?? API_URL.replace(/^http/, "ws")

export const api = axios.create({ baseURL: API_URL })

// ---------------------------------------------------------------- types --- //

export interface Incident {
  id: string
  title: string
  description: string | null
  source: string
  severity: string
  status: string
  raw_payload: Record<string, unknown> | null
  created_at: string
  updated_at: string
  resolved_at: string | null
}

export interface AgentMessage {
  id: string
  agent_name: string
  role: string
  run_id: string | null
  content: Record<string, unknown>
  confidence: number | null
  needs_more_data: boolean
  created_at: string
}

export interface AgentCostBreakdown {
  total_tokens: number
  estimated_cost_usd: number
  invocations: number
}

export interface CostSummary {
  total_tokens: number
  estimated_cost_usd: number
  per_agent_breakdown: Record<string, AgentCostBreakdown>
}

export interface IncidentDetail extends Incident {
  messages: AgentMessage[]
  cost_summary: CostSummary
}

export interface IncidentList {
  items: Incident[]
  total: number
  limit: number
  offset: number
}

export interface ProposedAction {
  action_type: string
  target: string
  tool_name: string
  parameters: Record<string, unknown>
  rationale: string
}

/** proposed_action on an ApprovalRequest is a serialized RemediationPlanPayload. */
export interface RemediationPlan {
  proposed_actions: ProposedAction[]
  risk_level: string
  requires_approval: boolean
  rollback_plan: string
  confidence: number
}

export interface ApprovalRequest {
  id: string
  incident_id: string
  proposed_action: RemediationPlan
  risk_level: string
  status: string
  requested_at: string
  resolved_at: string | null
  resolved_by: string | null
}

/** One frame from /ws/incidents/{id}/trace — history replay and live share it. */
export interface TraceStep {
  step_number: number
  agent_name: string
  thought: string
  action: {
    tool_name: string
    arguments: Record<string, unknown>
    call_id: string
  } | null
  observation: string | null
  reflection: string | null
  timestamp: string
  run_id: string | null
}

// ------------------------------------------------------------- requests --- //

export async function fetchIncidents(limit: number): Promise<IncidentList> {
  const { data } = await api.get<IncidentList>("/incidents", {
    params: { limit },
  })
  return data
}

export async function fetchIncident(id: string): Promise<IncidentDetail> {
  const { data } = await api.get<IncidentDetail>(`/incidents/${id}`)
  return data
}

export async function fetchPendingApproval(
  incidentId: string
): Promise<ApprovalRequest> {
  const { data } = await api.get<ApprovalRequest>(
    `/incidents/${incidentId}/approval`
  )
  return data
}

export async function submitApproval(
  incidentId: string,
  approved: boolean,
  resolvedBy: string
): Promise<void> {
  await api.post(`/incidents/${incidentId}/approve`, {
    approved,
    resolved_by: resolvedBy,
  })
}

export function traceSocketUrl(incidentId: string): string {
  return `${WS_URL}/ws/incidents/${incidentId}/trace`
}
