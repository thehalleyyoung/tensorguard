# Auto-generated shape stubs vs live PyTorch forwards

TensorGuard derives shape stubs directly from real `torch.nn` constructor signatures (`src/stub_autogen.py`). Autogeneration is **sound by abstention**: a stub is emitted only when the layer's shape contract is exactly known, and every emitted stub is validated against the layer's live forward output shape.

- Target classes: **26**
- Stubs generated: **23** (validated against live torch: **23**)
- Soundly abstained (UNSUPPORTED): **3** (`Conv2d`, `MaxPool2d`, `MultiheadAttention`)
- All generated stub shapes match live forward: **True**

| class | category | stub | predicted | live | match |
| --- | --- | --- | --- | --- | --- |
| `Linear` | last_dim_linear | True | [2, 3, 5] | [2, 3, 5] | True |
| `ReLU` | shape_preserving | True | [2, 3, 8] | [2, 3, 8] | True |
| `ReLU6` | shape_preserving | True | [2, 3, 8] | [2, 3, 8] | True |
| `GELU` | shape_preserving | True | [2, 3, 8] | [2, 3, 8] | True |
| `SiLU` | shape_preserving | True | [2, 3, 8] | [2, 3, 8] | True |
| `Mish` | shape_preserving | True | [2, 3, 8] | [2, 3, 8] | True |
| `Sigmoid` | shape_preserving | True | [2, 3, 8] | [2, 3, 8] | True |
| `Tanh` | shape_preserving | True | [2, 3, 8] | [2, 3, 8] | True |
| `ELU` | shape_preserving | True | [2, 3, 8] | [2, 3, 8] | True |
| `LeakyReLU` | shape_preserving | True | [2, 3, 8] | [2, 3, 8] | True |
| `Hardswish` | shape_preserving | True | [2, 3, 8] | [2, 3, 8] | True |
| `Softmax` | shape_preserving | True | [2, 3, 8] | [2, 3, 8] | True |
| `LogSoftmax` | shape_preserving | True | [2, 3, 8] | [2, 3, 8] | True |
| `Dropout` | shape_preserving | True | [2, 3, 8] | [2, 3, 8] | True |
| `Dropout2d` | shape_preserving | True | [2, 3, 4, 4] | [2, 3, 4, 4] | True |
| `AlphaDropout` | shape_preserving | True | [2, 3, 8] | [2, 3, 8] | True |
| `LayerNorm` | shape_preserving | True | [2, 3, 8] | [2, 3, 8] | True |
| `RMSNorm` | shape_preserving | True | [2, 3, 8] | [2, 3, 8] | True |
| `BatchNorm1d` | shape_preserving | True | [4, 3, 8] | [4, 3, 8] | True |
| `BatchNorm2d` | shape_preserving | True | [4, 3, 8, 8] | [4, 3, 8, 8] | True |
| `GroupNorm` | shape_preserving | True | [2, 4, 8] | [2, 4, 8] | True |
| `InstanceNorm2d` | shape_preserving | True | [2, 3, 8, 8] | [2, 3, 8, 8] | True |
| `Identity` | shape_preserving | True | [2, 3, 8] | [2, 3, 8] | True |
| `Conv2d` | unsupported | False | — | — | — |
| `MultiheadAttention` | unsupported | False | — | — | — |
| `MaxPool2d` | unsupported | False | — | — | — |

## Third-party layer (never seen by the verifier)

A user-defined `MyProjection(in_features, out_features)` is classified `last_dim_linear`, gets an auto-stub, and its predicted shape `[2, 7, 9]` matches the live forward `[2, 7, 9]` (**match = True**) — coverage generalizes beyond `torch.nn` from the constructor signature alone.
