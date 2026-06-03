# Step 256 cost/latency Pareto curves

Sound-mode sweep over model size, operator coverage, solver budget, and abstention. The JSON/Markdown artifact is deterministic; latency is reported only in the hardware-normalized volatile companion `evaluation/pareto_latency.json`.

## Budget frontier

| budget | work units | mean coverage | abstention rate | refined diagnoses | frontier? |
| --- | --- | --- | --- | --- | --- |
| 0 | 205 | 0.8958 | 0.2500 | 0 | True |
| 1 | 410 | 0.8958 | 0.2500 | 1 | True |
| 3 | 422 | 0.8958 | 0.2500 | 1 | False |

## Model/budget points

| model | family | budget | params | coverage | verdict | CEGAR iters | abstention reason |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `stack_1` | supported_stack | 0 | 1056 | 1.0000 | SAFE | 0 | - |
| `stack_4` | supported_stack | 0 | 4224 | 1.0000 | SAFE | 0 | - |
| `stack_8` | supported_stack | 0 | 8448 | 1.0000 | SAFE | 0 | - |
| `stack_16` | supported_stack | 0 | 16896 | 1.0000 | SAFE | 0 | - |
| `stack_32` | supported_stack | 0 | 33792 | 1.0000 | SAFE | 0 | - |
| `cegar_conflict` | budget_sensitive_bug | 0 | 12820 | 1.0000 | UNSAFE | 0 | - |
| `heuristic_lstsq` | heuristic_operator_abstention | 0 | 72 | 0.5000 | UNKNOWN | 0 | heuristic-tagged operator(s) used: torch.linalg.lstsq |
| `data_dependent_item` | fragment_abstention | 0 | 72 | 0.6667 | UNKNOWN | 0 | static fragment violation(s): DATA_DEPENDENT_CONTROL_FLOW, TENSOR_TO_SCALAR |
| `stack_1` | supported_stack | 1 | 1056 | 1.0000 | SAFE | 1 | - |
| `stack_4` | supported_stack | 1 | 4224 | 1.0000 | SAFE | 1 | - |
| `stack_8` | supported_stack | 1 | 8448 | 1.0000 | SAFE | 1 | - |
| `stack_16` | supported_stack | 1 | 16896 | 1.0000 | SAFE | 1 | - |
| `stack_32` | supported_stack | 1 | 33792 | 1.0000 | SAFE | 1 | - |
| `cegar_conflict` | budget_sensitive_bug | 1 | 12820 | 1.0000 | UNSAFE | 1 | - |
| `heuristic_lstsq` | heuristic_operator_abstention | 1 | 72 | 0.5000 | UNKNOWN | 1 | heuristic-tagged operator(s) used: torch.linalg.lstsq |
| `data_dependent_item` | fragment_abstention | 1 | 72 | 0.6667 | UNKNOWN | 1 | static fragment violation(s): DATA_DEPENDENT_CONTROL_FLOW, TENSOR_TO_SCALAR |
| `stack_1` | supported_stack | 3 | 1056 | 1.0000 | SAFE | 1 | - |
| `stack_4` | supported_stack | 3 | 4224 | 1.0000 | SAFE | 1 | - |
| `stack_8` | supported_stack | 3 | 8448 | 1.0000 | SAFE | 1 | - |
| `stack_16` | supported_stack | 3 | 16896 | 1.0000 | SAFE | 1 | - |
| `stack_32` | supported_stack | 3 | 33792 | 1.0000 | SAFE | 1 | - |
| `cegar_conflict` | budget_sensitive_bug | 3 | 12820 | 1.0000 | UNSAFE | 2 | - |
| `heuristic_lstsq` | heuristic_operator_abstention | 3 | 72 | 0.5000 | UNKNOWN | 1 | heuristic-tagged operator(s) used: torch.linalg.lstsq |
| `data_dependent_item` | fragment_abstention | 3 | 72 | 0.6667 | UNKNOWN | 1 | static fragment violation(s): DATA_DEPENDENT_CONTROL_FLOW, TENSOR_TO_SCALAR |

The non-dominated budget set keeps the zero-refinement point (lowest structural work) and the first refinement point (contract-level CEGAR diagnosis), while the higher budget is dominated once convergence has already been reached.
