# Theorem 5 audit on a strictly larger module population

| Metric | Value |
|---|---|
| Candidate modules | 107 |
| Modules with successful Dynamo warmup | 55 |
| Excluded — subprocess timeout (>240 s) | 35 |
| Excluded — warmup failed | 17 |
| Total in-contract recompiles | 72 |
| SHAPE/DTYPE/RANK recompiles | 0 |
| Out-of-catalogue SHAPE/DTYPE/RANK guards | 0 |
| Modules falsifying Theorem 5 | 0 / 55 |
| Falsifier rate | 0/0 (convention 0) |

By guard kind (aggregate over the 55 successful modules): INT = 72.

Hard constraints: subprocess-per-module isolation, 240 s wall-clock kill, 4-shard parallelism.
WON'T-CONVERT graph-break log lines are excluded from guard classification.
