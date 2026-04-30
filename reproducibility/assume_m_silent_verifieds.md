# Synthesised assume_M for residual silent verifieds (round-7 Q1)

Re-run with `python3.11 experiments_v5/v8/dump_assume_m_silent_verifieds.py`.

assume_M is *not* empty: TG synthesises concrete bindings for num_heads, num_chunks, chunk_size, dqk from the upstream __init__ defaults, which the analyser uses when reasoning about the view target.  Under those bindings, the buggy view target and the correct (post-#43209) view target both have the same total element count as the input on the supplied INPUT_SHAPES, so the shape-arithmetic check correctly returns 'no shape mismatch'.  The bug is a *semantic* axis-decomposition error (wrong-shape but right-size-for-this-input), which Theorem 2 (no shape mismatch under assume_M) does not claim to forbid.  Theorem 2 is therefore satisfied on these inputs, not vacuous: TG-Verified means 'no shape arithmetic violation under the concrete envelope', and the residual silent-miss class is delineated as 'semantic-only view bugs' in the limitations section.

## `rb_001_xlstm_matq_view.py`

- INPUT_SHAPES: `{'matQ': [2, 4, 128, 192]}`

### Synthesised assume_M (constructor-default param map)

```
{
  "constructor_default_param_map": {
    "num_heads": 4,
    "num_chunks": 2,
    "chunk_size": 64,
    "dqk": 192
  },
  "init_time_scalar_attrs": {
    "num_heads": 4,
    "num_chunks": 2,
    "chunk_size": 64,
    "dqk": 192
  },
  "symbolic_config_attrs": {},
  "divisibility_axioms": []
}
```

### Shape-arithmetic distinguishability

```
{
  "input_shape_total": 196608,
  "buggy_view_target": [
    2,
    4,
    2,
    64,
    192
  ],
  "buggy_view_total": 196608,
  "buggy_matches_input_total": true,
  "interpretation": "The buggy view target's total element count equals the input total under the synthesised assume_M, so the shape-arithmetic check correctly returns 'no shape mismatch' (Theorem 2 is satisfied non-vacuously).  The bug is a *semantic* axis-decomposition error: the trailing dim should have been dqk // num_chunks (per upstream PR #43209), but the resulting tensor has the same total size as the input, just with the wrong factorisation of the seq dim, which no purely shape-arithmetic rule can refute."
}
```

The buggy view target has **equal total element count to the input** under the synthesised assume_M, so the shape-arithmetic check correctly reports 'no shape mismatch'.  Theorem 2 is satisfied (and not vacuously: assume_M is concrete with 4 constructor-default bindings).  The bug is *semantic* — wrong factorisation of the seq dim, not a wrong total — which is outside what shape arithmetic can refute.

## `rb_002_xlstm_matk_view.py`

- INPUT_SHAPES: `{'matK': [2, 4, 128, 192]}`

### Synthesised assume_M (constructor-default param map)

```
{
  "constructor_default_param_map": {
    "num_heads": 4,
    "num_chunks": 2,
    "chunk_size": 64,
    "dqk": 192
  },
  "init_time_scalar_attrs": {
    "num_heads": 4,
    "num_chunks": 2,
    "chunk_size": 64,
    "dqk": 192
  },
  "symbolic_config_attrs": {},
  "divisibility_axioms": []
}
```

### Shape-arithmetic distinguishability

```
{
  "input_shape_total": 196608,
  "buggy_view_target": [
    2,
    4,
    2,
    64,
    192
  ],
  "buggy_view_total": 196608,
  "buggy_matches_input_total": true,
  "interpretation": "The buggy view target's total element count equals the input total under the synthesised assume_M, so the shape-arithmetic check correctly returns 'no shape mismatch' (Theorem 2 is satisfied non-vacuously).  The bug is a *semantic* axis-decomposition error: the trailing dim should have been dqk // num_chunks (per upstream PR #43209), but the resulting tensor has the same total size as the input, just with the wrong factorisation of the seq dim, which no purely shape-arithmetic rule can refute."
}
```

The buggy view target has **equal total element count to the input** under the synthesised assume_M, so the shape-arithmetic check correctly reports 'no shape mismatch'.  Theorem 2 is satisfied (and not vacuously: assume_M is concrete with 4 constructor-default bindings).  The bug is *semantic* — wrong factorisation of the seq dim, not a wrong total — which is outside what shape arithmetic can refute.
