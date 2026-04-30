# Right-reason audit of the 5 non-rb_uf_010 RP fires (round-4 Q2)

Reviewer: rb_uf_010 was an off-axis fire (device-mismatch where
the upstream PR fixes a dtype bug).  The other 5 RP fires must be
independently audited for right-axis catch.

**Headline.** 5/5 right-reason RP across the 5 non-rb_uf_010 fires; rb_uf_010 is independently confirmed off-axis (eval).

| id | bug class | n_bugs | matching | right-reason | PR |
|---|---|---|---|---|---|
| rb_pf_001 | config_dependent_linear_chain | 2 | 2 | YES | https://github.com/huggingface/diffusers/pull/13494 |
| rb_pf_003 | lora_in_out_swap_3d | 4 | 3 | YES | https://github.com/huggingface/peft/pull/3165 |
| rb_pf_004 | router_topk_vs_num_experts | 2 | 2 | YES | https://github.com/huggingface/transformers/pull/45473 |
| rb_uf_008 | view_total_size_mismatch | 1 | 1 | YES | https://github.com/huggingface/diffusers/pull/13520 |
| rb_uf_012 | data_dependent_control_flow | 2 | 2 | YES | https://github.com/huggingface/diffusers/pull/13561 |

## Per-item bug messages and matched keywords

### rb_pf_001 (config_dependent_linear_chain)

- Upstream axis: Linear-chain in/out feature-dim cast: the upstream diffusers#13494 fix replaces `int(dim*ff_mult)` with the constructor-folded equivalent that matches the next Linear's in_features.  The bug axis is the feature-dim of a Linear layer.

- L39, conf=0.99: `[SHAPE-INCOMPATIBLE] Linear expects last dim=9216, got 10240`
- L39, conf=0.8: `[SHAPE-INCOMPATIBLE] Z3 violation (shape_incompatible) at step 1:
  phase_s1 = TRAIN_1
  phase_s0 = TRAIN_1
  dev_x_s0 = CPU_1
  grad_x_s0 = False
  dev_h_s1 = CPU_1
  dev_x_s1 = CPU_1
  grad_x_s1 = F`

### rb_pf_003 (lora_in_out_swap_3d)

- Upstream axis: LoRA in/out swap: peft#3165 swaps in_features/out_features in a 3-D LoRA path.  The bug axis is in_features vs. out_features of the LoRA Linear.

- L48, conf=0.99: `[SHAPE-INCOMPATIBLE] Cannot broadcast (8, 512, 2048) and (8, 2048, 512)`
- L48, conf=0.8: `[SHAPE-INCOMPATIBLE] Z3 violation (shape_incompatible) at step 2:
  dev_base_out_s2 = CPU_3
  dev_base_out_s1 = CPU_3
  grad_base_out_s1 = False
  dev_x_s0 = CPU_3
  grad_x_s0 = False
  phase_s2 = TRA`
- L0, conf=0.95: `[CEGAR-REAL-BUG] Cannot broadcast (8, 512, 2048) and (8, 2048, 512)`
- L0, conf=0.95: `[CEGAR-REAL-BUG] Z3 violation (shape_incompatible) at step 2:
  _einsum = 1
  grad_lora_delta_s2 = False
  phase_s2 = TRAIN_4
  grad_base_out_s1 = False
  grad_base_out_s2 = False
  phase_s1 = TRAIN_4`

### rb_pf_004 (router_topk_vs_num_experts)

- Upstream axis: Router top-k vs. num_experts: transformers#45473 fixes a router that selects top_k logits from a num_experts-wide tensor when top_k > num_experts.  The bug axis is the top_k argument of the topk call.

- L43, conf=0.99: `[SHAPE-INCOMPATIBLE] Linear expects last dim=2, got 8`
- L43, conf=0.8: `[SHAPE-INCOMPATIBLE] Z3 violation (shape_incompatible) at step 2:
  grad_scores_s2 = True
  grad_x_s0 = False
  dev_logits_s2 = CPU_5
  phase_s2 = TRAIN_5
  dev_scores_s2 = CPU_5
  dev_x_s2 = CPU_5
  `

### rb_uf_008 (view_total_size_mismatch)

- Upstream axis: Wan VAE decoder view total-size mismatch: diffusers#13520 fixes a view whose target dims do not multiply to the input element count.  The bug axis is view total-size.

- L46, conf=0.99: `[SHAPE-INCOMPATIBLE] Conv3d expects 5D input, got 1D`

### rb_uf_012 (data_dependent_control_flow)

- Upstream axis: Hunyuan VAE NaN-branch: diffusers#13561 patches a data-dependent branch where a NaN guard switches between two reshape branches; the bug axis at the call site is either a view total-size mismatch or a transpose axis mismatch on one of the two paths.

- L43, conf=0.99: `[SHAPE-INCOMPATIBLE] Reshape incompatible: cannot reshape TensorShape(dims=(1, 4, 32, 32)) to (-2, -3, -1, 64, -1, 64)`
- L46, conf=0.99: `[SHAPE-INCOMPATIBLE] Reshape incompatible: cannot reshape TensorShape(dims=(1, 4, 32, 32)) to (-1, -3, 64, 64)`


Run with `python3.11 reproducibility/postfreeze_right_reason_audit.py`.
