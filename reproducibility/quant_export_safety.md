# Quantization & Export Safety: static verdict vs real PyTorch

TensorGuard's `src/quant_export_checks.py` flags two deployment-time hazard classes that the float shape/device/dtype verifier does not target. Each row is cross-checked against the **live** behavior of real PyTorch (`torch.export` tracing; quantized-tensor arithmetic).

## Export safety (`analyze_export_safety` vs `torch.export.export`)

| case | static export hazard | live exports clean | consistent |
| --- | --- | --- | --- |
| `clean_linear` | False | True | True |
| `data_dependent_branch` | True | False | True |
| `tensor_to_scalar_item` | True | False | True |
| `data_dependent_loop` | True | False | True |

A case is **consistent** when the static analyzer flags an export hazard if and only if real `torch.export` fails to trace.

## Quantization placement (`analyze_quantization` vs quantized-tensor ops)

| case | static quant hazard | live raises | consistent |
| --- | --- | --- | --- |
| `quant_add_no_floatfunctional` | True | True | True |
| `quant_add_with_floatfunctional` | False | null | True |
| `missing_dequantstub` | True | null | True |
| `plain_float_model` | False | null | True |

All export cases consistent: **True**. All quant cases consistent: **True**.

`live raises = null` marks structural boundary hazards (e.g. a missing `DeQuantStub`) that have no single runtime op to trip; they are verified statically only.
