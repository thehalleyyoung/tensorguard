# Operator frequency census (real model corpora)

Operators counted by `torch.fx` trace over 13 torchvision models, weighted by occurrence and cross-referenced against the operators TensorGuard reasons about. Generated against torch `2.9.1`, torchvision `0.24.1`.

Frequency-weighted coverage: **2437** of **2569** operator occurrences are covered (ratio 0.949).

Step 22 added denotational transfer functions for the highest-frequency previously-uncovered shape operators: `permute`, `expand`, `repeat`.

## Top operators by frequency

| Operator | Frequency | Covered |
|----------|-----------|---------|
| `Conv2d` | 577 | yes |
| `BatchNorm2d` | 456 | yes |
| `ReLU` | 359 | yes |
| `add` | 136 | yes |
| `getitem` | 120 | yes |
| `Linear` | 101 | yes |
| `cat` | 87 | yes |
| `LayerNorm` | 72 | yes |
| `Dropout` | 67 | yes |
| `mul` | 51 | yes |
| `stochastic_depth` | 51 | NO |
| `SiLU` | 49 | yes |
| `permute` | 49 | yes |
| `GELU` | 42 | yes |
| `AdaptiveAvgPool2d` | 40 | yes |
| `ReLU6` | 35 | yes |
| `Sigmoid` | 32 | yes |
| `view` | 32 | yes |
| `floordiv` | 18 | NO |
| `contiguous` | 16 | yes |
| `size` | 16 | NO |
| `transpose` | 16 | yes |
| `_assert` | 15 | yes |
| `eq` | 15 | yes |
| `getattr` | 15 | yes |
| `chunk` | 13 | yes |
| `dim` | 13 | NO |
| `MaxPool2d` | 12 | yes |
| `MultiheadAttention` | 12 | yes |
| `_get_relative_position_bias` | 12 | NO |

## Remaining uncovered operators (ranked)

* `stochastic_depth` (51)
* `floordiv` (18)
* `size` (16)
* `dim` (13)
* `_get_relative_position_bias` (12)
* `shifted_window_attention` (12)
* `layer_norm` (5)
* `_patch_merging_pad` (3)
* `adaptive_avg_pool2d` (2)
