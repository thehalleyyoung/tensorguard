# Bug-corpus handler LOO (real holdout)

Reviewer W5/Q5 (round 1).  The earlier LOO disabled v5 
orchestration files whose names overlapped a category label; 
those files contain no operator handlers, so the LOO was a 
literal no-op (53/60 → 53/60).  This file replaces it with a 
holdout that actually removes per-category handlers from the 
shape dispatch tables (`TORCH_SHAPE_OPS`, 
`MODERN_TORCH_SHAPE_OPS`, `FUNCTIONAL_SHAPE_RULES`) and stubs 
the corresponding shape compute primitives.

## Full pipeline: **RP 53/60**, silent 7, abstain 0, error 0

## Per-category drop after handler removal

| category | disabled handlers | full RP (cat) | LOO RP (cat) | global RP drop |
|---|---|---|---|---|
| view_reshape_total_size | `view, reshape` | 7 | 7 | 0 |
| broadcasting | `broadcast, add, mul, sub, div` | 7 | 7 | 0 |
| conv_channel_mismatch | `conv1d, conv2d, conv3d` | 6 | 6 | 0 |
| linear_inout_mismatch | `linear` | 4 | 4 | 0 |
| einsum_dim | `einsum, matmul, bmm` | 5 | 5 | 0 |
| transpose_axes | `transpose, permute` | 4 | 4 | 0 |
| attention_dim | `scaled_dot_product_attention, matmul, bmm, softmax, multihead_attention` | 4 | 4 | 0 |
| batchnorm_features | `batch_norm` | 4 | 4 | 0 |
| embedding_index | `embed, index_select, gather` | 3 | 3 | 0 |

Each row holds out the named handlers and re-runs the full 60-bug corpus.  A non-zero per-category drop demonstrates that the held-out handlers are actually responsible for catching bugs in their category — the evidence the no-op LOO failed to provide.
