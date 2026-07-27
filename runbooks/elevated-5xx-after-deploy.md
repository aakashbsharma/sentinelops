# Runbook: Elevated 5xx rate after a deploy

## Symptoms

HTTP 5xx rate rises within 15 minutes of a deployment finishing. Errors are
concentrated on the newly deployed service. Rollout dashboards show the new
ReplicaSet serving traffic.

## Likely causes

1. Missing or wrong environment variable / secret in the new release —
   crashes at startup or on first request to the affected code path.
2. Schema drift: the new code expects a migration that wasn't run (or ran
   partially). Look for column-does-not-exist errors.
3. A dependency version bump with a breaking change that only manifests
   under production traffic patterns.

## Diagnosis steps

Correlate the 5xx start time against the deploy timestamp — if the error
rate started BEFORE the deploy, stop, this runbook doesn't apply. Check the
new pods' startup logs for config errors, then check for migration errors in
the app logs. Diff the release's dependency lockfile if config and schema
are clean.

## Remediation

Default action: roll back the deployment first, diagnose second — every
minute of elevated 5xx is user-facing damage, and a rollback is cheap and
reversible. The exception is a schema-coupled release where the migration
already ran; rolling back the code without reverting the migration can make
things worse. In that case, roll FORWARD with a fix or apply the missing
migration.
