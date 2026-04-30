# Soundness footprint of the 5 post-freeze catches (Round 2 Q4)

## Headline

| Footprint | Count |
|---|---:|
| Lean-audited or pen-and-paper | 0 |
| tested-only | 0 |
| mixed (touches both) | 5 |
| uncovered-only | 0 |

## Per-catch

| id | footprint | source handlers (scope) |
|---|---|---|
| rb_pf_001_diffusers_longcat_ffmult | mixed (touches both) | linear(Lean-audited), mul(uncovered) |
| rb_pf_003_peft_lora_moe_swap | mixed (touches both) | expand(Lean-audited), add(uncovered), einsum(pen-and-paper), unsqueeze(tested-only) |
| rb_pf_004_routerparallel_topk | mixed (touches both) | linear(Lean-audited), softmax(tested-only) |
| rb_uf_008_wan_vae_decoder | mixed (touches both) | view(Lean-audited), reshape(Lean-audited), mul(uncovered) |
| rb_uf_012_hunyuan_vae_nan_branch | mixed (touches both) | view(Lean-audited), permute(Lean-audited), conv2d(Lean-audited), mul(uncovered) |
