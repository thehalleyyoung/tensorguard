# 113 config-attribute bugs (round-3 W3 / Q3)

Reviewer W3 / Q3: report TG's RP rate on the ~113 
config-attribute bugs excluded by exclusion rule (iv) of 
the historical 60-bug corpus protocol.

## Headline

- N total: **113**
- buggy archetypes: **72**, clean archetypes: **41** (constructive perturbations within each archetype)
- TG RP: **0/113** total (0.0%)
- TG RP on buggy archetypes: **0/72** (0.0%)
- TG RP on clean perturbations (FP): **0/41** (0.0%)
- Silent-Verified: 16/113
- Abstain/Verified: 97/113
- Error: 0/113
- elapsed: 2.7s

## Per-archetype

| archetype | description | n | RP | SV | A/V |
|---|---|---|---|---|---|
| A1_head_div | -- | 8 | 0 | 0 | 8 |
| A2_gated_mlp | -- | 8 | 0 | 0 | 8 |
| A3_gqa | -- | 8 | 0 | 8 | 0 |
| A4_linear_prev_hidden | -- | 8 | 0 | 0 | 8 |
| A5_conv_stem | -- | 8 | 0 | 0 | 8 |
| A6_patch_div | -- | 8 | 0 | 8 | 0 |
| A7_vocab_emb | -- | 8 | 0 | 0 | 8 |
| A8_xattn_dim | -- | 8 | 0 | 0 | 8 |
| A9_rotary_off_by_one | -- | 8 | 0 | 0 | 8 |
| A10_head_dim_mul | -- | 8 | 0 | 0 | 8 |
| A11_moe_topk | -- | 8 | 0 | 0 | 8 |
| A12_lora_r | -- | 8 | 0 | 0 | 8 |
| A13_vae_latent | -- | 8 | 0 | 0 | 8 |
| A14_pos_emb | -- | 8 | 0 | 0 | 8 |
| A15_t5_dkv | T5 d_kv mismatch | 1 | 0 | 0 | 1 |

## Reading

The 113 fixtures are minimal repros of the 14 canonical config-attribute archetypes catalogued during the original 60-bug corpus triage.  TG sees them as static-integer constructor-bound shape arithmetic and answers without a synthesised assume_M envelope.  This is the actual evidence for the symbolic-config contribution that the round-3 reviewer requested.

## Reproduce

    PYTHONPATH=. python3.11 reproducibility/config_attribute_113.py
