# TensorGuard on the Llama model family (HuggingFace)

## Command

```bash
python3 reproducibility/hf_extra_model_family.py
```

## Module set

Six representative modules from the Llama 2/3 decoder architecture
as implemented in HuggingFace Transformers (≥4.34.0).

## Results

| Module | Verdict | First bug (if RP) |
|---|---|---|
| LlamaMLP | Verified | — |
| LlamaRMSNorm | Verified | — |
| LlamaRotaryEmbedding | Verified | — |
| LlamaAttention | RP | Potential division by zero: 'num_heads' not guarded |
| LlamaDecoderLayer | Verified | — |
| LlamaMLP_buggy | RP | [SHAPE-INCOMPATIBLE] Linear expects last dim=8192, got 11008 |

## Summary

| Verdict | Count |
|---|---:|
| Verified | 4 |
| RP | 2 |
| Abstain | 0 |
| Error | 0 |

## Interpretation

TensorGuard handles the Llama architecture's clean modules (MLP,
RMSNorm, RotaryEmbedding, DecoderLayer) with Verified verdicts,
confirming that the analyser generalises beyond the torchvision/timm
corpora to a prominent generative-model family.

The intentionally buggy LlamaMLP_buggy variant (intermediate_size
mismatch: gate/up produce 11008 but down_proj expects 8192 input) is
caught with a SHAPE-INCOMPATIBLE refutation, demonstrating that the
static shape checker catches real shape-arithmetic mistakes in
Llama-style gated MLP blocks without requiring a model instantiation.

LlamaAttention is flagged RP with a conservative division-by-zero
guard: the analyser cannot rule out `num_heads == 0` without a runtime
value for the constructor argument (this fires because `head_dim =
hidden_size // num_heads` is not guarded at analysis time). In
practice `num_heads` is always a positive config integer; this is
a known LW/conservative RP pattern arising from config-bound division
arithmetic in `__init__`, consistent with the paper's framing of the
0/113 config-attribute silent-miss class.

## Paper claim (T6)

This artefact demonstrates cross-family generalisation to the Llama
model family with 4 Verified and 2 RP (1 genuine shape-mismatch bug,
1 conservative config-division guard) out of 6 representative modules.
