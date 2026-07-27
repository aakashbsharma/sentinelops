/**
 * Two views, zero routing library: the feed is always visible; selecting a
 * row opens the incident's detail (trace + approval) in a side panel. Local
 * state is all the navigation two views need.
 */
import { RadarIcon } from "lucide-react"
import { useState } from "react"

import { IncidentDetail } from "@/components/IncidentDetail"
import { IncidentFeed } from "@/components/IncidentFeed"

function App() {
  const [selectedId, setSelectedId] = useState<string | null>(null)

  return (
    <div className="flex h-dvh flex-col">
      <header className="flex items-center gap-2 border-b px-4 py-2.5">
        <RadarIcon className="size-5" />
        <span className="font-heading text-sm font-semibold">SentinelOps</span>
        <span className="text-xs text-muted-foreground">
          autonomous incident response
        </span>
      </header>

      <main className="grid min-h-0 flex-1 lg:grid-cols-[minmax(0,1fr)_minmax(0,1fr)] xl:grid-cols-[minmax(0,5fr)_minmax(0,4fr)]">
        <section className="min-h-0 overflow-y-auto border-r p-2">
          <IncidentFeed selectedId={selectedId} onSelect={setSelectedId} />
        </section>
        <aside className="hidden min-h-0 lg:block">
          {selectedId ? (
            <IncidentDetail incidentId={selectedId} />
          ) : (
            <div className="flex h-full items-center justify-center p-8 text-center text-sm text-muted-foreground">
              Select an incident to watch its agents reason in real time.
            </div>
          )}
        </aside>
        {/* Small screens: detail becomes an overlay instead of a column. */}
        {selectedId && (
          <div className="fixed inset-0 z-40 bg-background lg:hidden">
            <button
              className="w-full border-b px-4 py-2 text-left text-sm text-muted-foreground"
              onClick={() => setSelectedId(null)}
            >
              ← back to incidents
            </button>
            <div className="h-[calc(100dvh-2.5rem)]">
              <IncidentDetail incidentId={selectedId} />
            </div>
          </div>
        )}
      </main>
    </div>
  )
}

export default App
