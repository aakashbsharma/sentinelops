/**
 * WebSocket hook for /ws/incidents/{id}/trace.
 *
 * Protocol (Stage 7): on connect the server replays the incident's full step
 * history, then streams new AgentSteps live. Frames are deduped by a
 * composite key so a reconnect (which replays history again) or React
 * StrictMode's double-mount never duplicates timeline entries.
 */
import { useEffect, useRef, useState } from "react"

import { traceSocketUrl, type TraceStep } from "@/lib/api"

export type SocketStatus = "connecting" | "live" | "closed"

const MAX_RETRIES = 5

function frameKey(step: TraceStep): string {
  return `${step.run_id}|${step.agent_name}|${step.step_number}|${step.timestamp}`
}

/**
 * NOTE: state is NOT reset when incidentId changes — callers must remount
 * the consuming component per incident (key={incidentId}), which lets React
 * throw the old socket + steps away wholesale instead of us resetting state
 * inside an effect.
 */
export function useTraceSocket(incidentId: string | null) {
  const [steps, setSteps] = useState<TraceStep[]>([])
  const [status, setStatus] = useState<SocketStatus>("connecting")
  const seen = useRef<Set<string>>(new Set())

  useEffect(() => {
    if (incidentId === null) return

    let socket: WebSocket | null = null
    let retries = 0
    let retryTimer: number | undefined
    let disposed = false

    const connect = () => {
      socket = new WebSocket(traceSocketUrl(incidentId))

      socket.onopen = () => {
        retries = 0
        setStatus("live")
      }

      socket.onmessage = (event) => {
        let frame: TraceStep
        try {
          frame = JSON.parse(event.data as string)
        } catch {
          return // never let one malformed frame kill the stream
        }
        const key = frameKey(frame)
        if (seen.current.has(key)) return
        seen.current.add(key)
        setSteps((prev) => [...prev, frame])
      }

      socket.onclose = () => {
        if (disposed) return
        setStatus("closed")
        if (retries < MAX_RETRIES) {
          retries += 1
          setStatus("connecting")
          retryTimer = window.setTimeout(connect, 1000 * retries)
        }
      }
    }

    connect()
    return () => {
      disposed = true
      window.clearTimeout(retryTimer)
      socket?.close()
    }
  }, [incidentId])

  return { steps, status }
}
