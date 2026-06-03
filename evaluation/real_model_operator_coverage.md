# Step 208 real-model operator coverage

FX census over 18 no-download real models from torchvision, timm, and HuggingFace Transformers. Library versions: torch `2.9.1`, torchvision `0.24.1`, timm `1.0.26`, transformers `4.57.3`.

| Matrix | Covered / total occurrences | Frequency-weighted coverage |
|--------|-----------------------------|-----------------------------|
| before Step 208 | 3564 / 3800 | 0.9379 |
| after Step 208 | 3683 / 3800 | 0.9692 |

Newly covered hot operators:

* `stochastic_depth` (51 occurrences): torchvision.ops.stochastic_depth is an FX leaf whose output shape equals its input shape.
* `layer_norm` (53 occurrences): F.layer_norm now builds a synthetic LayerNorm layer and checks normalized trailing dims.
* `adaptive_avg_pool2d` (2 occurrences): F.adaptive_avg_pool2d now builds a synthetic AdaptiveAvgPool2d layer with exact output_size.
* `scaled_dot_product_attention` (13 occurrences): F.scaled_dot_product_attention maps to OpKind.SDPA with live torch parity tests.

Shape-metadata operators intentionally excluded from the new coverage numerator: `dim`, `floordiv`, `size`.

## Top after-Step-208 operators

| Operator | Frequency | Covered |
|----------|-----------|---------|
| `Conv2d` | 673 | yes |
| `BatchNorm2d` | 476 | yes |
| `ReLU` | 390 | yes |
| `Identity` | 242 | yes |
| `getitem` | 212 | yes |
| `add` | 200 | yes |
| `Linear` | 195 | yes |
| `Dropout` | 145 | yes |
| `permute` | 107 | yes |
| `getattr` | 100 | yes |
| `cat` | 88 | yes |
| `mul` | 78 | yes |
| `LayerNorm` | 75 | yes |
| `GELU` | 54 | yes |
| `layer_norm` | 53 | yes |
| `eq` | 52 | yes |
| `_assert` | 51 | yes |
| `stochastic_depth` | 51 | yes |
| `SiLU` | 49 | yes |
| `reshape` | 44 | yes |
| `AdaptiveAvgPool2d` | 43 | yes |
| `ReLU6` | 35 | yes |
| `view` | 35 | yes |
| `batch_norm` | 34 | NO |
| `transpose` | 33 | yes |
| `Sigmoid` | 32 | yes |
| `size` | 21 | NO |
| `Hardswish` | 19 | yes |
| `gelu` | 19 | yes |
| `floordiv` | 18 | NO |
| `contiguous` | 16 | yes |
| `dim` | 14 | NO |
| `MaxPool2d` | 13 | yes |
| `chunk` | 13 | yes |
| `scaled_dot_product_attention` | 13 | yes |
