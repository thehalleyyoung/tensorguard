# PR-history survival study (Step 254)

This is an offline, deterministic survival estimate over the frozen GitHub provenance corpus.  It asks whether TensorGuard's current runtime-signature-category check would have rejected the class of bug represented by a fix-linked PR/issue before the fix merged.  It does **not** replay historical pre-fix checkouts.

Corpus rows: **2704**; fix-linked rows: **332** (205 direct PR rows, 127 issue rows with candidate same-repo/category PR links).

## Category-level survival estimate

| population | rows | category-level caught | missed | catch rate |
| --- | ---: | ---: | ---: | ---: |
| all fix-linked | 332 | 332 | 0 | 1.000 |
| direct PR only | 205 | 205 | 0 | 1.000 |
| candidate issue links only | 127 | 127 | 0 | 1.000 |

## Runtime-signature replay rows

| category | fix-linked rows | direct PR | candidate | TensorGuard verdict | eager proof | dynamic prefix ops | graph steps |
| --- | ---: | ---: | ---: | --- | --- | ---: | ---: |
| `broadcast_mismatch` | 43 | 17 | 26 | UNSAFE | proven_live | 0 | 2 |
| `cat_stack_mismatch` | 20 | 17 | 3 | UNSAFE | proven_live | 0 | 2 |
| `conv_channel_mismatch` | 12 | 9 | 3 | UNSAFE | proven_live | 0 | 2 |
| `device_mismatch` | 22 | 17 | 5 | UNSAFE | cuda_qualified_not_executed_in_cpu_ci | 2 | 4 |
| `dim_out_of_range` | 130 | 87 | 43 | UNSAFE | proven_live | 0 | 2 |
| `dtype_device_input_mismatch` | 9 | 9 | 0 | UNSAFE | proven_live | 1 | 3 |
| `matmul_linear_mismatch` | 57 | 33 | 24 | UNSAFE | proven_live | 0 | 2 |
| `view_reshape_total_size` | 39 | 16 | 23 | UNSAFE | proven_live | 0 | 2 |

## Detection-depth and CI-cost proxy

The proxy is structural and deterministic, not a wall-clock benchmark.

- Static gate: 332 analyzer passes, 717 graph steps analyzed, 0 model executions.
- Dynamic forward baseline: 332 forward invocations, 417 concrete tensor constructions, 53 successful prefix ops before the failing op.
- Static gate avoids those concrete tensor constructions: 417.

## Scope notes

- No historical patches or source bodies are redistributed or replayed.
- Device eager replay is CUDA-qualified; source-level static replay still exercises the device-transfer contract.
- Dtype replay covers the same runtime-signature family with a source-visible non-floating input; the committed eager reproducer covers the sibling input/weight dtype mismatch.
- All cost figures are deterministic structural proxies, not wall-clock measurements.
