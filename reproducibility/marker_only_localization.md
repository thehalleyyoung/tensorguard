# Marker-only localisation audit (round-2 W6)

Reviewer W6: replace the 33/33 figure (heuristic-derived GT) with
a marker-only audit at N>=30.  This audit uses author-placed
`# BUG ...` comments in the rb_* repros as the ground-truth marker;
the bug line is the next executable line after the marker.

- marker-bearing repros: **30**
- refuted by TG with computable distance: **17**

## Summary (within-K hit rate)

| version | within_1 | within_5 | within_10 | miss |
|---|---|---|---|---|
| v4 (raw bug.location.line) | 10 | 14 | 17 | 0 |
| v5 (localize) | 11 | 14 | 17 | 0 |

v5 within ±5: **14/17** (82%)
v5 within ±1: **11/17** (64%)

## Command

```
python3.11 reproducibility/marker_only_localization.py
```

## Per-item (refuted)

| id | corpus | gt_line | tg_line_v5 | dist_v5 |
|---|---|---|---|---|
| rb_001_xlstm_matq_view | real_bugs_upstream | 31 | None | None |
| rb_002_xlstm_matk_view | real_bugs_upstream | 21 | None | None |
| rb_003_gptneox_odd_heads | real_bugs_upstream | 28 | 32 | 4 |
| rb_004_convbert_head_ratio | real_bugs_upstream | 21 | 22 | 1 |
| rb_005_longformer_global_attn | real_bugs_upstream | 20 | 20 | 0 |
| rb_006_longt5_tp_attention | real_bugs_upstream | 22 | 22 | 0 |
| rb_007_gptneox_gqa_reshape | real_bugs_upstream | 20 | 21 | 1 |
| rb_008_diffusers_unet1d_fourier | real_bugs_upstream | 26 | 27 | 1 |
| rb_009_peft_prefix_tuning | real_bugs_upstream | 23 | 23 | 0 |
| rb_010_peft_dora_conv_groups | real_bugs_upstream | 24 | 24 | 0 |
| rb_pf_001_diffusers_longcat_ffmult | real_bugs_postfreeze | 31 | 39 | 8 |
| rb_pf_002_t5gemma2_xattn_cache | real_bugs_postfreeze | 35 | None | None |
| rb_pf_003_peft_lora_moe_swap | real_bugs_postfreeze | 38 | 48 | 10 |
| rb_pf_004_routerparallel_topk | real_bugs_postfreeze | 40 | 43 | 3 |
| rb_pf_005_diffusers_npu_mask | real_bugs_postfreeze | 40 | None | None |
| rb_pf_006_qwenimage_batch_ordering | real_bugs_postfreeze | 37 | None | None |
| rb_uf_007_idefics3_patch_merger | real_bugs_unfiltered | 28 | None | None |
| rb_uf_008_wan_vae_decoder | real_bugs_unfiltered | 39 | 46 | 7 |
| rb_uf_009_glm45_moe_chunk | real_bugs_unfiltered | 45 | None | None |
| rb_uf_010_phi5_dtype | real_bugs_unfiltered | 31 | 36 | 5 |
| rb_uf_012_hunyuan_vae_nan_branch | real_bugs_unfiltered | 42 | 43 | 1 |
| rb_uf_013_peft_vera_scaler | real_bugs_unfiltered | 29 | None | None |
| rb_uf_014_smollm3_grad_ckpt | real_bugs_unfiltered | 32 | None | None |
| rb_uf_015_cosmos2_vision_transpose | real_bugs_unfiltered | 37 | None | None |
| bug_001_sdpa_attn_mask_gqa | bug_repros | 16 | None | None |
| bug_002_isclose_broadcast | bug_repros | 14 | None | None |
| bug_003_view_total_size | bug_repros | 14 | 14 | 0 |
| bug_004_view_empty | bug_repros | 14 | 14 | 0 |
| bug_005_broadcast_mismatch | bug_repros | 19 | 19 | 0 |
| bug_006_cross_entropy_weight | bug_repros | 19 | None | None |
