# Step 252 same-case head-to-head benchmark

Broad comparison on the exact **20** frozen LLM-baseline cases (10 buggy / 10 clean). The committed GPT-4.1-nano artifact is hash-checked and reused; no external LLM call is made.

## Headline metrics (NA counts as wrong)

| tool | TP | FP | TN | FN | NA | precision | recall | F1 | coverage |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `tensorguard_unified` | 10 | 0 | 10 | 0 | 0 | 1.000 | 1.000 | 1.000 | 1.000 |
| `pytea` | 0 | 9 | 1 | 10 | 19 | 0.000 | 0.000 | 0.000 | 0.050 |
| `pyright` | 2 | 0 | 10 | 8 | 0 | 1.000 | 0.200 | 0.333 | 1.000 |
| `jaxtyping_runtime` | 0 | 10 | 0 | 10 | 20 | 0.000 | 0.000 | 0.000 | 0.000 |
| `torchtyping_runtime` | 0 | 10 | 0 | 10 | 20 | 0.000 | 0.000 | 0.000 | 0.000 |
| `torch_export_guards` | 3 | 7 | 3 | 7 | 14 | 0.300 | 0.300 | 0.300 | 0.300 |
| `torch_dynamo_guards` | 0 | 10 | 0 | 10 | 20 | 0.000 | 0.000 | 0.000 | 0.000 |
| `runtime_forward_smoke` | 8 | 0 | 10 | 2 | 0 | 1.000 | 0.800 | 0.889 | 1.000 |
| `llm_gpt4_1_nano_frozen` | 8 | 1 | 9 | 2 | 0 | 0.889 | 0.800 | 0.842 | 1.000 |

## Module subset for export-style tools

| tool | TP | FP | TN | FN | NA | recall | coverage |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `tensorguard_unified` | 3 | 0 | 3 | 0 | 0 | 1.000 | 1.000 |
| `pytea` | 0 | 3 | 0 | 3 | 6 | 0.000 | 0.000 |
| `pyright` | 0 | 0 | 3 | 3 | 0 | 0.000 | 1.000 |
| `jaxtyping_runtime` | 0 | 3 | 0 | 3 | 6 | 0.000 | 0.000 |
| `torchtyping_runtime` | 0 | 3 | 0 | 3 | 6 | 0.000 | 0.000 |
| `torch_export_guards` | 3 | 0 | 3 | 0 | 0 | 1.000 | 1.000 |
| `torch_dynamo_guards` | 0 | 3 | 0 | 3 | 6 | 0.000 | 0.000 |
| `runtime_forward_smoke` | 3 | 0 | 3 | 0 | 0 | 1.000 | 1.000 |
| `llm_gpt4_1_nano_frozen` | 3 | 1 | 2 | 0 | 0 | 1.000 | 1.000 |

## Capability axes

| tool | static/no exec | needs inputs | needs annotations | code upload/API | live/frozen |
| --- | --- | --- | --- | --- | --- |
| `tensorguard_unified` | True | False | False | False | live_local |
| `pytea` | True | False | False | False | live_local_if_node_available |
| `pyright` | True | False | False | False | live_local_if_cli_available |
| `jaxtyping_runtime` | False | True | True | False | live_local_if_package_available |
| `torchtyping_runtime` | False | True | True | False | live_local_if_package_available |
| `torch_export_guards` | False | True | False | False | live_local |
| `torch_dynamo_guards` | False | True | False | False | live_local_if_supported_by_torch |
| `runtime_forward_smoke` | False | True | False | False | live_local |
| `llm_gpt4_1_nano_frozen` | True | False | False | True | frozen_prior_api_run |

## Reading the comparison honestly

- Pyright is a Python type checker, not a tensor-shape verifier; on this corpus it can catch Optional/None bugs but not Linear/Conv/reshape/broadcast tensor-shape mismatches.
- jaxtyping/torchtyping require user-authored shape annotations. The source cases are unannotated, so these runtime rows mostly collapse to executing the same deterministic entrypoint under a thin checked wrapper.
- torch.export and TorchDynamo only support nn.Module entries here; function-only cases are scored NA and summarized by the module subset.

LLM artifact: `experiments/llm_baseline_results.json` (sha256 `3dbc38de4390e75b29ad121fb688f5f38bf614d239c04b958cd2e6150448de9d`).
