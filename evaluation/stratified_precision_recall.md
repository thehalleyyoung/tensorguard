# Stratified precision/recall with sample-size gates (Step 250)

This artifact post-processes the committed real-code predictions in `evaluation/confusion_matrices.json`. It does **not** re-run detectors, so PyTea/runtime availability cannot silently change the strata.

Wilson score confidence intervals are shown as `estimate [low, high]`; `(expl.)` marks a metric whose denominator is below the publication gate.

## Executable corpus: TensorGuard strata

### operator_family

| stratum | N | TP | FP | TN | FN | precision | recall | FPR | gate |
| --- | ---: | ---: | ---: | ---: | ---: | --- | --- | --- | --- |
| `attention_matmul` | 1 | 0 | 0 | 1 | 0 | -- | -- | 0.000 [0.000, 0.793] (expl.) | exploratory_only |
| `broadcast_concat` | 1 | 1 | 0 | 0 | 0 | 1.000 [0.206, 1.000] (expl.) | 1.000 [0.206, 1.000] (expl.) | -- | exploratory_only |
| `convolution` | 2 | 1 | 0 | 1 | 0 | 1.000 [0.206, 1.000] (expl.) | 1.000 [0.206, 1.000] (expl.) | 0.000 [0.000, 0.793] (expl.) | exploratory_only |
| `convolution_pool_linear` | 1 | 0 | 0 | 1 | 0 | -- | -- | 0.000 [0.000, 0.793] (expl.) | exploratory_only |
| `device_placement` | 1 | 1 | 0 | 0 | 0 | 1.000 [0.206, 1.000] (expl.) | 1.000 [0.206, 1.000] (expl.) | -- | exploratory_only |
| `gradient_flow` | 1 | 1 | 0 | 0 | 0 | 1.000 [0.206, 1.000] (expl.) | 1.000 [0.206, 1.000] (expl.) | -- | exploratory_only |
| `matmul_linear` | 4 | 3 | 0 | 1 | 0 | 1.000 [0.439, 1.000] (expl.) | 1.000 [0.439, 1.000] (expl.) | 0.000 [0.000, 0.793] (expl.) | exploratory_only |
| `normalization_convolution` | 1 | 0 | 0 | 1 | 0 | -- | -- | 0.000 [0.000, 0.793] (expl.) | exploratory_only |
| `normalization_linear` | 1 | 0 | 0 | 1 | 0 | -- | -- | 0.000 [0.000, 0.793] (expl.) | exploratory_only |
| `phase_dropout_linear` | 1 | 0 | 0 | 1 | 0 | -- | -- | 0.000 [0.000, 0.793] (expl.) | exploratory_only |
| `reshape_view` | 1 | 1 | 0 | 0 | 0 | 1.000 [0.206, 1.000] (expl.) | 1.000 [0.206, 1.000] (expl.) | -- | exploratory_only |
| `residual_add` | 1 | 0 | 0 | 1 | 0 | -- | -- | 0.000 [0.000, 0.793] (expl.) | exploratory_only |

### framework

| stratum | N | TP | FP | TN | FN | precision | recall | FPR | gate |
| --- | ---: | ---: | ---: | ---: | ---: | --- | --- | --- | --- |
| `pytorch_nn_module` | 16 | 8 | 0 | 8 | 0 | 1.000 [0.676, 1.000] | 1.000 [0.676, 1.000] | 0.000 [0.000, 0.324] | precision_recall_claimable |

### bug_class

| stratum | N | TP | FP | TN | FN | precision | recall | FPR | gate |
| --- | ---: | ---: | ---: | ---: | ---: | --- | --- | --- | --- |
| `broadcasting` | 1 | 1 | 0 | 0 | 0 | 1.000 [0.206, 1.000] (expl.) | 1.000 [0.206, 1.000] (expl.) | -- | exploratory_only |
| `clean` | 8 | 0 | 0 | 8 | 0 | -- | -- | 0.000 [0.000, 0.324] | exploratory_only |
| `conv_channel_mismatch` | 1 | 1 | 0 | 0 | 0 | 1.000 [0.206, 1.000] (expl.) | 1.000 [0.206, 1.000] (expl.) | -- | exploratory_only |
| `device_mismatch` | 1 | 1 | 0 | 0 | 0 | 1.000 [0.206, 1.000] (expl.) | 1.000 [0.206, 1.000] (expl.) | -- | exploratory_only |
| `gradient_detach` | 1 | 1 | 0 | 0 | 0 | 1.000 [0.206, 1.000] (expl.) | 1.000 [0.206, 1.000] (expl.) | -- | exploratory_only |
| `linear_inout_mismatch` | 3 | 3 | 0 | 0 | 0 | 1.000 [0.439, 1.000] (expl.) | 1.000 [0.439, 1.000] (expl.) | -- | exploratory_only |
| `view_reshape_total_size` | 1 | 1 | 0 | 0 | 0 | 1.000 [0.206, 1.000] (expl.) | 1.000 [0.206, 1.000] (expl.) | -- | exploratory_only |

### model_family

| stratum | N | TP | FP | TN | FN | precision | recall | FPR | gate |
| --- | ---: | ---: | ---: | ---: | ---: | --- | --- | --- | --- |
| `attention` | 1 | 0 | 0 | 1 | 0 | -- | -- | 0.000 [0.000, 0.793] (expl.) | exploratory_only |
| `branching_mlp` | 1 | 1 | 0 | 0 | 0 | 1.000 [0.206, 1.000] (expl.) | 1.000 [0.206, 1.000] (expl.) | -- | exploratory_only |
| `cnn` | 3 | 2 | 0 | 1 | 0 | 1.000 [0.342, 1.000] (expl.) | 1.000 [0.342, 1.000] (expl.) | 0.000 [0.000, 0.793] (expl.) | exploratory_only |
| `conv_pool_classifier` | 1 | 0 | 0 | 1 | 0 | -- | -- | 0.000 [0.000, 0.793] (expl.) | exploratory_only |
| `device_module` | 1 | 1 | 0 | 0 | 0 | 1.000 [0.206, 1.000] (expl.) | 1.000 [0.206, 1.000] (expl.) | -- | exploratory_only |
| `gradient_mlp` | 1 | 1 | 0 | 0 | 0 | 1.000 [0.206, 1.000] (expl.) | 1.000 [0.206, 1.000] (expl.) | -- | exploratory_only |
| `matrix_module` | 1 | 1 | 0 | 0 | 0 | 1.000 [0.206, 1.000] (expl.) | 1.000 [0.206, 1.000] (expl.) | -- | exploratory_only |
| `mlp` | 2 | 1 | 0 | 1 | 0 | 1.000 [0.206, 1.000] (expl.) | 1.000 [0.206, 1.000] (expl.) | 0.000 [0.000, 0.793] (expl.) | exploratory_only |
| `norm_mlp` | 1 | 0 | 0 | 1 | 0 | -- | -- | 0.000 [0.000, 0.793] (expl.) | exploratory_only |
| `normalization_cnn` | 1 | 0 | 0 | 1 | 0 | -- | -- | 0.000 [0.000, 0.793] (expl.) | exploratory_only |
| `phase_mlp` | 1 | 0 | 0 | 1 | 0 | -- | -- | 0.000 [0.000, 0.793] (expl.) | exploratory_only |
| `reshape_module` | 1 | 1 | 0 | 0 | 0 | 1.000 [0.206, 1.000] (expl.) | 1.000 [0.206, 1.000] (expl.) | -- | exploratory_only |
| `residual_cnn` | 1 | 0 | 0 | 1 | 0 | -- | -- | 0.000 [0.000, 0.793] (expl.) | exploratory_only |

### source

| stratum | N | TP | FP | TN | FN | precision | recall | FPR | gate |
| --- | ---: | ---: | ---: | ---: | ---: | --- | --- | --- | --- |
| `canonical_clean` | 8 | 0 | 0 | 8 | 0 | -- | -- | 0.000 [0.000, 0.324] | exploratory_only |
| `canonical_pattern` | 2 | 2 | 0 | 0 | 0 | 1.000 [0.342, 1.000] (expl.) | 1.000 [0.342, 1.000] (expl.) | -- | exploratory_only |
| `pytorch_issue` | 6 | 6 | 0 | 0 | 0 | 1.000 [0.610, 1.000] | 1.000 [0.610, 1.000] | -- | precision_recall_claimable |

## Provenance corpus sample-size coverage

The GitHub-mined corpus below is positive-only; it supports sample-size coverage claims, not precision/recall claims.

### operator_family

| stratum | records | share with Wilson CI | sample-size gate |
| --- | ---: | --- | --- |
| `broadcast` | 398 | 0.147 [0.134, 0.161] | pass |
| `concat_stack` | 398 | 0.147 [0.134, 0.161] | pass |
| `convolution` | 399 | 0.148 [0.135, 0.161] | pass |
| `device_placement` | 291 | 0.108 [0.097, 0.120] | pass |
| `dtype_device_contract` | 19 | 0.007 [0.004, 0.011] (expl.) | insufficient_n |
| `indexing_dimension` | 400 | 0.148 [0.135, 0.162] | pass |
| `matmul_linear` | 400 | 0.148 [0.135, 0.162] | pass |
| `reshape_view` | 399 | 0.148 [0.135, 0.161] | pass |

### framework

| stratum | records | share with Wilson CI | sample-size gate |
| --- | ---: | --- | --- |
| `pytorch` | 2704 | 1.000 [0.999, 1.000] | pass |

### bug_class

| stratum | records | share with Wilson CI | sample-size gate |
| --- | ---: | --- | --- |
| `broadcast_mismatch` | 398 | 0.147 [0.134, 0.161] | pass |
| `cat_stack_mismatch` | 398 | 0.147 [0.134, 0.161] | pass |
| `conv_channel_mismatch` | 399 | 0.148 [0.135, 0.161] | pass |
| `device_mismatch` | 291 | 0.108 [0.097, 0.120] | pass |
| `dim_out_of_range` | 400 | 0.148 [0.135, 0.162] | pass |
| `dtype_device_input_mismatch` | 19 | 0.007 [0.004, 0.011] (expl.) | insufficient_n |
| `matmul_linear_mismatch` | 400 | 0.148 [0.135, 0.162] | pass |
| `view_reshape_total_size` | 399 | 0.148 [0.135, 0.161] | pass |

### model_family

| stratum | records | share with Wilson CI | sample-size gate |
| --- | ---: | --- | --- |
| `diffusion` | 451 | 0.167 [0.153, 0.181] | pass |
| `graph_learning` | 40 | 0.015 [0.011, 0.020] | pass |
| `language_model` | 164 | 0.061 [0.052, 0.070] | pass |
| `unclassified_public_repo` | 1835 | 0.679 [0.661, 0.696] | pass |
| `vision_detection` | 214 | 0.079 [0.070, 0.090] | pass |

### source

| stratum | records | share with Wilson CI | sample-size gate |
| --- | ---: | --- | --- |
| `github_issue` | 2499 | 0.924 [0.914, 0.934] | pass |
| `github_pull_request` | 205 | 0.076 [0.066, 0.086] | pass |

## Honesty notes

- Executable precision/recall strata are scored from the frozen real-code predictions; the script never re-runs detectors.
- The framework axis is intentionally degenerate in the current evidence: all executable models are PyTorch nn.Module cases and all mined records come from PyTorch runtime-error signatures.
- Operator-family, bug-class, and model-family slices are sparse and partly collinear on the 16-case executable corpus; sparse slices are marked exploratory instead of being promoted to paper claims.
- The 2,704-record provenance corpus is positive-only and is used here only for sample-size coverage, not for precision/recall scoring.
