# TensorGuard on the Qwen2 model family (Round 1 — Comet cycle)

## Command

```
python3 reproducibility/hf_extra_family_round_comet1.py
```

## Module set

Five Qwen2 modules, self-contained (no HF imports).

## Results

| Module | Verdict | First bug |
|---|---|---|
| Qwen2RMSNorm | Verified |  |
| Qwen2MLP | Verified |  |
| Qwen2GQAAttention | Verified |  |
| Qwen2DecoderLayer | Verified |  |
| Qwen2MLP_buggy | RP | [SHAPE-INCOMPATIBLE] Linear expects last dim=12288, got 18944 |

## Summary

| Verdict | Count |
|---|---:|
| Verified | 4 |
| RP | 1 |

## Interpretation

TensorGuard generalises to the Qwen2 family (5 modules). The buggy `Qwen2MLP_buggy` variant (intermediate_size mismatch 18944 vs 12288) is the intentional negative control.  Qwen2 is a strictly new family relative to the 488-block torchvision/timm/transformers corpus and the prior Llama expansion.

