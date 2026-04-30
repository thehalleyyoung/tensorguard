# Synthesised assume_M for the post-freeze silent verifieds (round-8 Q1)

Re-run with `PYTHONPATH=. python3 experiments_v5/v8/dump_assume_m_postfreeze.py`.

On all three remaining post-freeze silent verifieds (rb_pf_002, rb_pf_005, rb_pf_006), assume_M is *non-empty*: the upstream constructor scalars are bound to concrete integers in the constructor_default_param_map, and propagate to the per-forward scalar_attrs.  Theorem 2 is therefore satisfied non-vacuously --- TG-Verified means 'no shape arithmetic violation under the concrete envelope'.  The buggy edge in each case is a per-call shape comparison TG's existing rule table currently abstains on (broadcast-add strict-equality witness; chunk-then-elementwise-mul); concrete mismatches are in the per-row 'concrete_mismatch' field below.  Closing this class is a per-rule strengthening, not an envelope-synthesis or assume-vacuity gap.

## `rb_pf_002_t5gemma2_xattn_cache.py`

- INPUT_SHAPES: `{'hidden': [1, 4097, 512]}`
- Buggy edge   : `matmul(q,k.T) -> (B,N,4097,5018) plus bad_mask (B,1,4097,4097)`
- Edge class   : broadcast-add strict-equality witness
- Concrete mismatch (under assume_M): `scores last-dim=5018 != bad_mask last-dim=4097`

### Synthesised assume_M

```
{
  "constructor_default_param_map": {
    "num_heads": 8,
    "head_dim": 64,
    "sliding_window": 4096,
    "encoder_len": 5018
  },
  "init_time_scalar_attrs": {
    "num_heads": 8,
    "head_dim": 64,
    "q_len_truncated": 4097,
    "k_len_full": 5018
  },
  "symbolic_config_attrs": {},
  "divisibility_axioms": []
}
```

**Reading.** The constructor scalars are *bound to concrete integers*; assume_M is not vacuous.  The buggy edge has a concrete (statically-known) mismatch on integer dims, but the current operator-rule table for that edge asks for a divisibility witness rather than a strict-equality witness, so the analyser correctly does not raise --- which is consistent with the soundness story (no false positives, narrowest fragment) and with the round-8 limitation paragraph.

## `rb_pf_005_diffusers_npu_mask.py`

- INPUT_SHAPES: `{'hidden': [1, 128, 512], 'mask': [1, 1, 1, 128]}`
- Buggy edge   : `scores (B,N,128,128) + bad_mask = mask.expand(B,1,128,129)`
- Edge class   : broadcast-add strict-equality witness
- Concrete mismatch (under assume_M): `scores last-dim=128 != bad_mask last-dim=129`

### Synthesised assume_M

```
{
  "constructor_default_param_map": {
    "num_heads": 8,
    "head_dim": 64,
    "seq_len": 128
  },
  "init_time_scalar_attrs": {
    "num_heads": 8,
    "head_dim": 64,
    "seq_len": 128
  },
  "symbolic_config_attrs": {},
  "divisibility_axioms": []
}
```

**Reading.** The constructor scalars are *bound to concrete integers*; assume_M is not vacuous.  The buggy edge has a concrete (statically-known) mismatch on integer dims, but the current operator-rule table for that edge asks for a divisibility witness rather than a strict-equality witness, so the analyser correctly does not raise --- which is consistent with the soundness story (no false positives, narrowest fragment) and with the round-8 limitation paragraph.

## `rb_pf_006_qwenimage_batch_ordering.py`

- INPUT_SHAPES: `{'latents': [8, 64]}`
- Buggy edge   : `(model_pred-target) * weighting where weighting=(2*B,1) but model_pred=(B,_)`
- Edge class   : chunk-then-elementwise-mul
- Concrete mismatch (under assume_M): `model_pred dim0=4 vs weighting dim0=8 (= 2 * train_batch)`

### Synthesised assume_M

```
{
  "constructor_default_param_map": {
    "hidden": 64,
    "train_batch": 4
  },
  "init_time_scalar_attrs": {
    "train_batch": 4
  },
  "symbolic_config_attrs": {},
  "divisibility_axioms": []
}
```

**Reading.** The constructor scalars are *bound to concrete integers*; assume_M is not vacuous.  The buggy edge has a concrete (statically-known) mismatch on integer dims, but the current operator-rule table for that edge asks for a divisibility witness rather than a strict-equality witness, so the analyser correctly does not raise --- which is consistent with the soundness story (no false positives, narrowest fragment) and with the round-8 limitation paragraph.
