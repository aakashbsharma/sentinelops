# Runbook: High CPU on payment-service

## Symptoms

CPU utilization above 85% on payment-service pods for more than 5 minutes.
Often accompanied by rising p99 latency on the /charge endpoint and an
increase in gateway timeouts from the checkout flow.

## Likely causes

1. Retry storm from a downstream card-processor outage — payment-service
   retries synchronously and burns CPU serializing requests.
2. A deploy that re-enabled verbose request logging (JSON serialization of
   full card metadata is CPU-heavy).
3. Legitimate traffic spike (flash sale) without a matching HPA scale-up.

## Diagnosis steps

Check the card-processor status page and the retry-rate metric
`payment.processor.retries` first. If retries are flat, diff the last deploy
for logging config changes. Compare request rate against the same hour last
week to rule traffic in or out.

## Remediation

For retry storms: enable the circuit breaker via the `PAYMENT_CB_ENABLED`
flag and scale the deployment to 6 replicas. For logging regressions: roll
back the deploy. For real traffic: scale to 8 replicas and raise the HPA max.
Historically, scaling without fixing a retry storm just burns more CPU —
fix the breaker first.
