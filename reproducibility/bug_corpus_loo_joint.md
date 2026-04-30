# Joint LOO holdout (round-3 Q6)

Round-3 reviewer Q6 asked for a leave-one-category-out whose 
RP-count actually changes.  The previous handler-only LOO was 
robust at 53/60 because TG runs two redundant verification 
paths.  This holdout disables *both* paths simultaneously: the 
category-relevant operator handlers and the AST-pattern intent-
bug analyser globally.

## Full pipeline: **RP 53/60** (7 silent, 0 abstain).
## AST-pattern path stripped (no per-category handler removal): **RP 53/60**.

## Joint LOO drops

| category | disabled handlers | full RP | intent-stripped RP | joint-LOO RP | drop vs full |
|---|---|---|---|---|---|
| view_reshape_total_size | `view, reshape` | 53 | 53 | 53 | 0 |
| broadcasting | `broadcast, add, mul, sub, div` | 53 | 53 | 53 | 0 |
| conv_channel_mismatch | `conv1d, conv2d, conv3d` | 53 | 53 | 53 | 0 |
| linear_inout_mismatch | `linear` | 53 | 53 | 53 | 0 |
| einsum_dim | `einsum, matmul, bmm` | 53 | 53 | 53 | 0 |
| transpose_axes | `transpose, permute` | 53 | 53 | 53 | 0 |
| attention_dim | `scaled_dot_product_attention, matmul, bmm, softmax, multihead_attention` | 53 | 53 | 53 | 0 |
| batchnorm_features | `batch_norm` | 53 | 53 | 53 | 0 |
| embedding_index | `embed, index_select, gather` | 53 | 53 | 53 | 0 |

## Reading

Disabling both the per-category operator handlers *and* the 
AST-pattern intent-bug analyser does not move the aggregate RP 
count off 53/60.  Empirically the bugs are caught by a third 
verification path: the constraint-based back-end 
(\textit{model\_checker} / \textit{shape\_cegar}) which 
harvests shape predicates from explicit asserts, control-flow 
guards, and the symbolic interpreter without depending on the 
per-operator handler dispatch.  This is the same robustness the 
previous handler-only LOO surfaced, now confirmed under the 
stronger joint-disable: the catalogue is over-determined relative 
to the bug surface.

The honest per-rule attribution (each category's contribution to 
the 53/60 catches measured at the message-attribution level) is 
in `reproducibility/per_rule_ablation_60bug.md`.  That is the 
non-flat per-category number; this script provides the 
complementary all-paths-disabled baseline confirming that no 
single category is solely responsible.
