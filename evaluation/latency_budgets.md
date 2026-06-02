# End-to-end verification latency budgets

Per-model latency budgets for the full `verify_model` pipeline (source parse, graph extraction, bounded model checking, Z3), grouped by size tier. The committed manifest is deterministic (budgets and extracted step counts only); measured wall-clock latency is enforced live by `make latency-budgets-gate`.

| Model | Tier | Steps | Budget (s) |
|-------|------|-------|------------|
| `large_stack_40` | large | 121 | 30.0 |
| `medium_stack_12` | medium | 37 | 12.0 |
| `small_cnn` | small | 8 | 3.0 |
| `small_mlp` | small | 4 | 3.0 |
