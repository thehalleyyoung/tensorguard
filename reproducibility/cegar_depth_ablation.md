# CEGAR refinement-depth ablation: precision/runtime trade-off (Step 119)

Sweep of the shape-CEGAR budget over **16** infeasible-contract bugs and **8** clean controls, every case validated against real PyTorch.

| depth | bugs detected | refined-contract diagnoses | clean false alarms | total CEGAR iterations |
| --- | --- | --- | --- | --- |
| 0 | 16/16 | 0/16 | 0/8 | 0 |
| 1 | 16/16 | 16/16 | 0/8 | 24 |
| 2 | 16/16 | 16/16 | 0/8 | 40 |
| 3 | 16/16 | 16/16 | 0/8 | 40 |
| 4 | 16/16 | 16/16 | 0/8 | 40 |
| 5 | 16/16 | 16/16 | 0/8 | 40 |
| 6 | 16/16 | 16/16 | 0/8 | 40 |

## Trade-off

- detection (recall) is depth-invariant at full and clean false alarms stay zero at every depth — CEGAR depth is a diagnosis knob, not a soundness knob: **True**
- contract-level diagnostic precision rises from 0 at depth 0 to full (16) at the refinement knee (depth 1), then plateaus: **True**
- work saturates at the convergence bound (depth 2, 40 total iterations); beyond it the budget is free but useless: **True**
