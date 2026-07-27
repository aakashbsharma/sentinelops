# Runbook: Database connection pool exhaustion

## Symptoms

Application logs show "TimeoutError: could not acquire connection from pool"
or "remaining connection slots are reserved". API requests hang and then
fail with 500s. Database CPU is often NORMAL — the bottleneck is connection
slots, not query load. Connections to Postgres time out or queue.

## Likely causes

1. A slow query holding connections open (check `pg_stat_activity` for
   long-running transactions — anything over 30 seconds is suspect).
2. A code path leaking connections: sessions opened but never closed,
   usually after an unhandled exception skips cleanup.
3. Too many app replicas for the configured `max_connections` — each replica
   holds `pool_size` connections even when idle.

## Diagnosis steps

Query `pg_stat_activity` grouped by state and application_name. If most
connections are `idle in transaction`, it's a leak or a stuck transaction.
If they're `active` on the same query, it's a slow query. Count total
connections against `max_connections` to check for replica over-provisioning.

## Remediation

For stuck transactions: terminate the specific backend with
`pg_terminate_backend(pid)` — do NOT restart the database. For leaks: roll
back to the previous app version and file the leak. For over-provisioning:
reduce per-replica pool_size or add pgbouncer in transaction mode. A full
database restart is almost never the right first move; it drops every healthy
connection too.
