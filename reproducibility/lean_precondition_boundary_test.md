# Lean-rule precondition boundary test

Reviewer W4 (round 1).  The 28,000/28,000 Lean--torch agreement 
harness samples in-envelope shapes and so cannot detect a wrong 
(too narrow) precondition.  This file documents the complementary 
*boundary* test: for each Lean-audited rule, sample shapes 
*outside* the declared envelope and check that either torch 
raises or the torch shape differs from the rule's predicted 
shape.  Any 'silent_through' sample exposes a too-narrow 
precondition.

## Aggregate

- rules covered: **28**
- off-envelope samples: **6913**
- of those: torch raised: **6875** | shape disagreement: **0** | silent-through: **0**

A silent-through count of zero (or near zero) is the desired 
outcome: any non-zero count flags a precondition too narrow vs. 
torch's actually-permissive behaviour and is a soundness liability.

## Per-rule

| rule | off-env samples | torch raised | shape disagree | silent-through |
|---|---|---|---|---|
| matmul | 250 | 250 | 0 | 0 |
| bmm | 250 | 250 | 0 | 0 |
| view | 250 | 250 | 0 | 0 |
| reshape | 250 | 250 | 0 | 0 |
| permute | 250 | 250 | 0 | 0 |
| cat | 220 | 220 | 0 | 0 |
| stack | 193 | 193 | 0 | 0 |
| linear | 250 | 250 | 0 | 0 |
| embed | 250 | 212 | 0 | 0 |
| transpose | 250 | 250 | 0 | 0 |
| conv1d | 250 | 250 | 0 | 0 |
| conv2d | 250 | 250 | 0 | 0 |
| conv3d | 250 | 250 | 0 | 0 |
| conv_transpose2d | 250 | 250 | 0 | 0 |
| expand | 250 | 250 | 0 | 0 |
| repeat | 250 | 250 | 0 | 0 |
| broadcast_to | 250 | 250 | 0 | 0 |
| split | 250 | 250 | 0 | 0 |
| chunk | 250 | 250 | 0 | 0 |
| unbind | 250 | 250 | 0 | 0 |
| gather | 250 | 250 | 0 | 0 |
| scatter | 250 | 250 | 0 | 0 |
| index_select | 250 | 250 | 0 | 0 |
| narrow | 250 | 250 | 0 | 0 |
| layer_norm | 250 | 250 | 0 | 0 |
| rms_norm | 250 | 250 | 0 | 0 |
| scaled_dot_product_attention | 250 | 250 | 0 | 0 |
| batched_matmul | 250 | 250 | 0 | 0 |

Run with: `python3.11 reproducibility/lean_precondition_boundary_test.py`.
