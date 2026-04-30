# Per-block Lean-rule pinning for the 5 audited-footprint unconditional-RP catches

Each of the 5 unconditional-RP catches whose detected handler set lies entirely in the audited sub-catalogue is shown below, with each detected handler pinned to the specific Lean rule (or pen-and-paper rule) that discharges its per-step soundness obligation. The Subject Reduction theorem then composes these per-step lemmas to discharge the whole-forward-body verdict.

## `timm__VisionTransformerDistilled__032a9d92`

- library: `timm`, category: `vision_vit`, LOC: 97
- detected handlers: `cat, expand, linear, view`
- no_non_audited_handler_in_proof: **True**

| handler | scope | Lean theorem | Lean file:line | pen-paper rule |
|---|---|---|---|---|
| `cat` | lean_verified | `applyOp_sound_cat` | `lean/TensorGuard/SoundnessV5.lean:121` | — |
| `expand` | lean_verified | `applyOp_sound_expand` | `lean/TensorGuard/SoundnessV5.lean:103` | — |
| `linear` | lean_verified | `applyOp_sound_linear_v5` | `lean/TensorGuard/SoundnessV5.lean:198` | — |
| `view` | lean_verified | `applyOp_sound_view_v5` | `lean/TensorGuard/SoundnessV5.lean:79` | — |

## `transformers__BloomPreTrainedModel__4a772e64`

- library: `transformers`, category: `transformer`, LOC: 28
- detected handlers: `embed, layer_norm, linear`
- no_non_audited_handler_in_proof: **True**

| handler | scope | Lean theorem | Lean file:line | pen-paper rule |
|---|---|---|---|---|
| `embed` | lean_verified | `applyOp_sound_embed` | `lean/TensorGuard/SoundnessV5.lean:175` | — |
| `layer_norm` | lean_verified | `applyOp_sound_layer_norm` | `lean/TensorGuard/SoundnessV5.lean:179` | — |
| `linear` | lean_verified | `applyOp_sound_linear_v5` | `lean/TensorGuard/SoundnessV5.lean:198` | — |

## `transformers__ElectraForPreTraining__74bdacae`

- library: `transformers`, category: `transformer`, LOC: 99
- detected handlers: `squeeze, view`
- no_non_audited_handler_in_proof: **True**

| handler | scope | Lean theorem | Lean file:line | pen-paper rule |
|---|---|---|---|---|
| `squeeze` | lean_verified | `applyOp_sound_squeeze` | `lean/TensorGuard/SoundnessV5.lean:238` | — |
| `view` | lean_verified | `applyOp_sound_view_v5` | `lean/TensorGuard/SoundnessV5.lean:79` | — |

## `transformers__FalconPreTrainedModel__55936532`

- library: `transformers`, category: `transformer`, LOC: 40
- detected handlers: `embed, layer_norm, linear`
- no_non_audited_handler_in_proof: **True**

| handler | scope | Lean theorem | Lean file:line | pen-paper rule |
|---|---|---|---|---|
| `embed` | lean_verified | `applyOp_sound_embed` | `lean/TensorGuard/SoundnessV5.lean:175` | — |
| `layer_norm` | lean_verified | `applyOp_sound_layer_norm` | `lean/TensorGuard/SoundnessV5.lean:179` | — |
| `linear` | lean_verified | `applyOp_sound_linear_v5` | `lean/TensorGuard/SoundnessV5.lean:198` | — |

## `transformers__WhisperModel__2cb724a8`

- library: `transformers`, category: `transformer`, LOC: 190
- detected handlers: `expand`
- no_non_audited_handler_in_proof: **True**

| handler | scope | Lean theorem | Lean file:line | pen-paper rule |
|---|---|---|---|---|
| `expand` | lean_verified | `applyOp_sound_expand` | `lean/TensorGuard/SoundnessV5.lean:103` | — |

Reproduce with `python3 reproducibility/audited_footprint_per_block_lean_pinning.py`.
