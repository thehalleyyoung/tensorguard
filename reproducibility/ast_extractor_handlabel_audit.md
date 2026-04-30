# AST extractor hand-labelled OOD audit

## Inputs / configuration

Hand-labelled ground truth derived by manual semantic
inspection of each fixture's ``__init__`` body, NOT by
running the deployed extractor.  Compared against the
deployed ``_InitExtractor`` on the same fixtures.

## Summary

* Modules audited: ``n = 20``
* ``init_param_names`` exact agreement: ``1/20``
* ``symbolic_config_attrs`` exact agreement: ``20/20``
* ``symbolic_config_attrs`` subset (deployed ⊆ hand-label): ``20/20``

Subset agreement is the soundness-direction comparison:
any disagreement on the (deployed ⊄ hand-label) side would
indicate the deployed extractor over-extracted relative to
the literal source, which is the unsafe direction.

## Per-fixture detail

### experiments_v5/v8/real_bugs_upstream/rb_001_xlstm_matq_view.py

* ``family``: ``xlstm``
* ``class``: ``BuggyModule``
* ``ground_truth_init_params``: ``['chunk_size', 'dqk', 'num_chunks', 'num_heads']``
* ``deployed_init_params``: ``[]``
* ``init_match``: ``False``
* ``ground_truth_symbolic_config_attrs``: ``[]``
* ``deployed_symbolic_config_attrs``: ``[]``
* ``symbolic_config_attrs_eq``: ``True``
* ``symbolic_config_attrs_subset``: ``True``

### experiments_v5/v8/real_bugs_upstream/rb_002_xlstm_matk_view.py

* ``family``: ``xlstm``
* ``class``: ``BuggyModule``
* ``ground_truth_init_params``: ``['chunk_size', 'dqk', 'num_chunks', 'num_heads']``
* ``deployed_init_params``: ``[]``
* ``init_match``: ``False``
* ``ground_truth_symbolic_config_attrs``: ``[]``
* ``deployed_symbolic_config_attrs``: ``[]``
* ``symbolic_config_attrs_eq``: ``True``
* ``symbolic_config_attrs_subset``: ``True``

### experiments_v5/v8/real_bugs_upstream/rb_003_gptneox_odd_heads.py

* ``family``: ``gptneox``
* ``class``: ``BuggyModule``
* ``ground_truth_init_params``: ``['hidden_size', 'num_attention_heads']``
* ``deployed_init_params``: ``[]``
* ``init_match``: ``False``
* ``ground_truth_symbolic_config_attrs``: ``[]``
* ``deployed_symbolic_config_attrs``: ``[]``
* ``symbolic_config_attrs_eq``: ``True``
* ``symbolic_config_attrs_subset``: ``True``

### experiments_v5/v8/real_bugs_upstream/rb_004_convbert_head_ratio.py

* ``family``: ``convbert``
* ``class``: ``BuggyModule``
* ``ground_truth_init_params``: ``['head_ratio', 'hidden_size', 'num_attention_heads']``
* ``deployed_init_params``: ``[]``
* ``init_match``: ``False``
* ``ground_truth_symbolic_config_attrs``: ``[]``
* ``deployed_symbolic_config_attrs``: ``[]``
* ``symbolic_config_attrs_eq``: ``True``
* ``symbolic_config_attrs_subset``: ``True``

### experiments_v5/v8/real_bugs_upstream/rb_005_longformer_global_attn.py

* ``family``: ``longformer``
* ``class``: ``BuggyModule``
* ``ground_truth_init_params``: ``['max_num_global_attn_indices', 'num_heads']``
* ``deployed_init_params``: ``[]``
* ``init_match``: ``False``
* ``ground_truth_symbolic_config_attrs``: ``[]``
* ``deployed_symbolic_config_attrs``: ``[]``
* ``symbolic_config_attrs_eq``: ``True``
* ``symbolic_config_attrs_subset``: ``True``

### experiments_v5/v8/real_bugs_upstream/rb_006_longt5_tp_attention.py

* ``family``: ``t5``
* ``class``: ``BuggyModule``
* ``ground_truth_init_params``: ``['d_kv', 'd_model', 'num_heads', 'tp_world_size']``
* ``deployed_init_params``: ``[]``
* ``init_match``: ``False``
* ``ground_truth_symbolic_config_attrs``: ``[]``
* ``deployed_symbolic_config_attrs``: ``[]``
* ``symbolic_config_attrs_eq``: ``True``
* ``symbolic_config_attrs_subset``: ``True``

### experiments_v5/v8/real_bugs_upstream/rb_007_gptneox_gqa_reshape.py

* ``family``: ``gptneox``
* ``class``: ``BuggyModule``
* ``ground_truth_init_params``: ``['hn', 'kvp', 'np']``
* ``deployed_init_params``: ``[]``
* ``init_match``: ``False``
* ``ground_truth_symbolic_config_attrs``: ``[]``
* ``deployed_symbolic_config_attrs``: ``[]``
* ``symbolic_config_attrs_eq``: ``True``
* ``symbolic_config_attrs_subset``: ``True``

### experiments_v5/v8/real_bugs_upstream/rb_008_diffusers_unet1d_fourier.py

* ``family``: ``diffusers``
* ``class``: ``GaussianFourierProjection``
* ``ground_truth_init_params``: ``['embedding_size']``
* ``deployed_init_params``: ``[]``
* ``init_match``: ``False``
* ``ground_truth_symbolic_config_attrs``: ``[]``
* ``deployed_symbolic_config_attrs``: ``[]``
* ``symbolic_config_attrs_eq``: ``True``
* ``symbolic_config_attrs_subset``: ``True``

### experiments_v5/v8/real_bugs_upstream/rb_009_peft_prefix_tuning.py

* ``family``: ``peft``
* ``class``: ``BuggyModule``
* ``ground_truth_init_params``: ``['d_model', 'num_attention_heads', 'num_layers', 'num_virtual_tokens']``
* ``deployed_init_params``: ``[]``
* ``init_match``: ``False``
* ``ground_truth_symbolic_config_attrs``: ``[]``
* ``deployed_symbolic_config_attrs``: ``[]``
* ``symbolic_config_attrs_eq``: ``True``
* ``symbolic_config_attrs_subset``: ``True``

### experiments_v5/v8/real_bugs_upstream/rb_010_peft_dora_conv_groups.py

* ``family``: ``peft``
* ``class``: ``BuggyModule``
* ``ground_truth_init_params``: ``['groups', 'in_channels', 'kernel', 'out_channels']``
* ``deployed_init_params``: ``[]``
* ``init_match``: ``False``
* ``ground_truth_symbolic_config_attrs``: ``[]``
* ``deployed_symbolic_config_attrs``: ``[]``
* ``symbolic_config_attrs_eq``: ``True``
* ``symbolic_config_attrs_subset``: ``True``

### experiments_v5/v8/real_bugs_postfreeze/rb_pf_001_diffusers_longcat_ffmult.py

* ``family``: ``diffusers``
* ``class``: ``BuggyModule``
* ``ground_truth_init_params``: ``['dim', 'ff_mult_actual', 'ff_mult_hardcoded']``
* ``deployed_init_params``: ``[]``
* ``init_match``: ``False``
* ``ground_truth_symbolic_config_attrs``: ``[]``
* ``deployed_symbolic_config_attrs``: ``[]``
* ``symbolic_config_attrs_eq``: ``True``
* ``symbolic_config_attrs_subset``: ``True``

### experiments_v5/v8/real_bugs_postfreeze/rb_pf_002_t5gemma2_xattn_cache.py

* ``family``: ``t5gemma``
* ``class``: ``BuggyModule``
* ``ground_truth_init_params``: ``['encoder_len', 'head_dim', 'num_heads', 'sliding_window']``
* ``deployed_init_params``: ``[]``
* ``init_match``: ``False``
* ``ground_truth_symbolic_config_attrs``: ``[]``
* ``deployed_symbolic_config_attrs``: ``[]``
* ``symbolic_config_attrs_eq``: ``True``
* ``symbolic_config_attrs_subset``: ``True``

### experiments_v5/v8/real_bugs_postfreeze/rb_pf_003_peft_lora_moe_swap.py

* ``family``: ``peft``
* ``class``: ``BuggyModule``
* ``ground_truth_init_params``: ``['in_features', 'num_experts', 'out_features', 'r']``
* ``deployed_init_params``: ``[]``
* ``init_match``: ``False``
* ``ground_truth_symbolic_config_attrs``: ``[]``
* ``deployed_symbolic_config_attrs``: ``[]``
* ``symbolic_config_attrs_eq``: ``True``
* ``symbolic_config_attrs_subset``: ``True``

### experiments_v5/v8/real_bugs_postfreeze/rb_pf_004_routerparallel_topk.py

* ``family``: ``routerparallel``
* ``class``: ``BuggyModule``
* ``ground_truth_init_params``: ``['hidden_size', 'num_experts', 'top_k']``
* ``deployed_init_params``: ``[]``
* ``init_match``: ``False``
* ``ground_truth_symbolic_config_attrs``: ``[]``
* ``deployed_symbolic_config_attrs``: ``[]``
* ``symbolic_config_attrs_eq``: ``True``
* ``symbolic_config_attrs_subset``: ``True``

### experiments_v5/v8/real_bugs_postfreeze/rb_pf_005_diffusers_npu_mask.py

* ``family``: ``diffusers``
* ``class``: ``BuggyModule``
* ``ground_truth_init_params``: ``['head_dim', 'num_heads', 'seq_len']``
* ``deployed_init_params``: ``[]``
* ``init_match``: ``False``
* ``ground_truth_symbolic_config_attrs``: ``[]``
* ``deployed_symbolic_config_attrs``: ``[]``
* ``symbolic_config_attrs_eq``: ``True``
* ``symbolic_config_attrs_subset``: ``True``

### experiments_v5/v8/real_bugs_postfreeze/rb_pf_006_qwenimage_batch_ordering.py

* ``family``: ``qwen``
* ``class``: ``BuggyModule``
* ``ground_truth_init_params``: ``['hidden', 'train_batch']``
* ``deployed_init_params``: ``[]``
* ``init_match``: ``False``
* ``ground_truth_symbolic_config_attrs``: ``[]``
* ``deployed_symbolic_config_attrs``: ``[]``
* ``symbolic_config_attrs_eq``: ``True``
* ``symbolic_config_attrs_subset``: ``True``

### experiments_v5/v8/real_bugs_unfiltered/rb_uf_007_idefics3_patch_merger.py

* ``family``: ``extra``
* ``class``: ``BuggyModule``
* ``ground_truth_init_params``: ``['hidden_size', 'scale_factor']``
* ``deployed_init_params``: ``[]``
* ``init_match``: ``False``
* ``ground_truth_symbolic_config_attrs``: ``[]``
* ``deployed_symbolic_config_attrs``: ``[]``
* ``symbolic_config_attrs_eq``: ``True``
* ``symbolic_config_attrs_subset``: ``True``

### experiments_v5/v8/real_bugs_unfiltered/rb_uf_008_wan_vae_decoder.py

* ``family``: ``extra``
* ``class``: ``BuggyModule``
* ``ground_truth_init_params``: ``['channels', 'h_in', 't_in', 'w_in']``
* ``deployed_init_params``: ``[]``
* ``init_match``: ``False``
* ``ground_truth_symbolic_config_attrs``: ``[]``
* ``deployed_symbolic_config_attrs``: ``[]``
* ``symbolic_config_attrs_eq``: ``True``
* ``symbolic_config_attrs_subset``: ``True``

### experiments_v5/v8/real_bugs_unfiltered/rb_uf_009_glm45_moe_chunk.py

* ``family``: ``extra``
* ``class``: ``BuggyModule``
* ``ground_truth_init_params``: ``[]``
* ``deployed_init_params``: ``[]``
* ``init_match``: ``True``
* ``ground_truth_symbolic_config_attrs``: ``[]``
* ``deployed_symbolic_config_attrs``: ``[]``
* ``symbolic_config_attrs_eq``: ``True``
* ``symbolic_config_attrs_subset``: ``True``

### experiments_v5/v8/real_bugs_unfiltered/rb_uf_010_phi5_dtype.py

* ``family``: ``extra``
* ``class``: ``BuggyModule``
* ``ground_truth_init_params``: ``['head_dim', 'hidden_size', 'num_heads']``
* ``deployed_init_params``: ``[]``
* ``init_match``: ``False``
* ``ground_truth_symbolic_config_attrs``: ``[]``
* ``deployed_symbolic_config_attrs``: ``[]``
* ``symbolic_config_attrs_eq``: ``True``
* ``symbolic_config_attrs_subset``: ``True``

