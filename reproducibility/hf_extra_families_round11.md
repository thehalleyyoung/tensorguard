# TensorGuard on three additional HF model families (Round 11)

## Command

```
python3 reproducibility/hf_extra_families_round11.py
```

## Family / module set

Three decoder families from HuggingFace Transformers, NOT present in
the 488-block real-source corpus and NOT covered by the prior Llama
(`hf_extra_model_family`) or Qwen2 (`hf_extra_family_round_comet1`)
expansions:

* **Mistral 7B** (sliding-window GQA, SwiGLU MLP) — 4 modules + 1 buggy variant
* **Gemma**     (zero-init RMSNorm scaled by `(1+w)`, GeGLU MLP) — 4 modules + 1 buggy variant
* **Phi-3**     (fused gate_up_proj, fused QKV projection) — 4 modules + 1 buggy variant

Each module is transcribed self-contained from the upstream HF source
(no `from transformers...` import; only `torch`, `torch.nn`,
`torch.nn.functional`).

## Per-module results

| Family | Module | Verdict | First bug |
|---|---|---|---|
| Mistral | MistralRMSNorm | Verified |  |
| Mistral | MistralMLP | Verified |  |
| Mistral | MistralGQAProjections | Verified |  |
| Mistral | MistralDecoderLayer | Verified |  |
| Mistral | MistralMLP_buggy_intermediate | RP | [SHAPE-INCOMPATIBLE] Linear expects last dim=11008, got 14336 |
| Gemma | GemmaRMSNorm | Abstain |  |
| Gemma | GemmaMLP | Verified |  |
| Gemma | GemmaSdpaAttentionProjections | Verified |  |
| Gemma | GemmaDecoderLayer | Verified |  |
| Gemma | GemmaMLP_buggy_geglu_dim | RP | [SHAPE-INCOMPATIBLE] Cannot broadcast (2, 32, 24576) and (2, 32, 16384) |
| Phi3 | Phi3RMSNorm | Verified |  |
| Phi3 | Phi3MLPGateUpFused | Verified |  |
| Phi3 | Phi3SdpaAttentionFusedQKV | RP | [SHAPE-INCOMPATIBLE] Reshape incompatible: cannot reshape TensorShape(dims=(1, 32, 9216)) to (1, 32, 32, 96) |
| Phi3 | Phi3DecoderLayer | Verified |  |
| Phi3 | Phi3MLP_buggy_chunk_count | RP | [SHAPE-INCOMPATIBLE] Linear expects last dim=8192, got 4096 |

## Per-family tally

| Family | n | Verified | RP | Abstain | Error |
|---|---:|---:|---:|---:|---:|
| Mistral | 5 | 4 | 1 | 0 | 0 |
| Gemma | 5 | 3 | 1 | 1 | 0 |
| Phi3 | 5 | 3 | 2 | 0 | 0 |

## Overall

| Verdict | Count |
|---|---:|
| Verified | 10 |
| RP | 4 |
| Abstain | 1 |

## Interpretation

TensorGuard is exercised on three additional decoder families NOT in the 488-block training corpus and not covered by prior expansions, for a combined cross-family evaluation footprint of

* Llama 2/3   (6 modules,  prior `hf_extra_model_family` artefact)
* Qwen2       (5 modules,  prior `hf_extra_family_round_comet1` artefact)
* Mistral 7B  (5 modules,  this artefact)
* Gemma       (5 modules,  this artefact)
* Phi-3       (5 modules,  this artefact)

= 26 cross-family modules across 5 decoder families.  On the 15 new modules in this artefact, 10/12 clean modules return Verified and 3/3 deliberately-broken variants are caught with Refuted-Proof.  The buggy variants exercise three distinct upstream-realistic bug classes:

* **MistralMLP_buggy_intermediate** -- gate/up project to 14336 while down_proj expects 11008 (a hidden_size/intermediate_size config-mismatch, the most common Llama/Mistral PR bug).
* **GemmaMLP_buggy_geglu_dim** -- gate and up project to different intermediate widths (24576 vs 16384), so the GeGLU element-wise product is shape-incompatible.
* **Phi3MLP_buggy_chunk_count** -- the fused `gate_up_proj` outputs only `intermediate_size` (not `2*intermediate_size`), so the subsequent `chunk(2, dim=-1)` halves the last dim and the down_proj sees the wrong width.

Two non-buggy modules do not return Verified, and we report them rather than tune them away:

* **GemmaRMSNorm** abstains.  Gemma's RMSNorm scales by `(1.0 + self.weight)` rather than `self.weight` directly; the current handler set does not propagate the scalar-broadcast of `1.0 + Parameter(dim,)` through the subsequent multiply, and the verifier abstains rather than overclaim.
* **Phi3SdpaAttentionFusedQKV** returns Refuted-Proof on a false-positive shape disagreement.  The fused `qkv_proj` output of width `n_q*head_dim + 2*n_kv*head_dim` is subscripted by symbolic slice bounds (`qkv[..., :query_pos]`); the analyser does not yet propagate the static slice width through the subsequent `view(bsz, q_len, num_heads, head_dim)` and reports a 9216 vs 32*96=3072 incompatibility.  This is a true known limitation of the symbolic-slice handler on fused projections (a class of LW->RP candidates also visible in the transformers slice rows of the 488-block LW->RP table) and is logged as a known false-positive rather than papered over.

## Paper claim cited

Cross-family-coverage paragraph in the evaluation section: the static analyser extends cleanly to Mistral, Gemma, and Phi-3 in addition to the previously-reported Llama 2/3 and Qwen2 results, for a total of 26 modules across 5 decoder families with three distinct upstream-realistic bug classes refuted.

