# Latency regression benchmark (Criterion-style)

Calibration-normalized verification cost for each benchmark case. Each ratio is the case's median verification time divided by the anchor model's median time in the same run, so the value is independent of absolute machine speed. `make regression-bench-gate` fails CI when any case regresses by more than the committed tolerance.

| Case | Steps | Baseline ratio |
|------|-------|----------------|
| `stack_12` | 37 | 11.0147 |
| `stack_24` | 73 | 40.3260 |
| `stack_6` | 19 | 3.1549 |

Tolerance: 10 percent regression budget.
