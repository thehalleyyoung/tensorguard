# IC3/PDR Finite-Theory Anchoring Ablation Results

## Overview

This ablation study addresses Reviewer Sinha's concern that IC3/PDR ≤3-frame
convergence could be explained by benchmark simplicity rather than finite-theory
anchoring. We compare:

- **Full 5-theory**: T_shape × T_device × T_phase × T_stride × T_perm
- **Reduced 2-theory**: T_shape × T_stride only (removing finite theories T_device, T_phase, T_perm)

**Benchmarks evaluated**: 32

## Key Findings

### Verdict Agreement: 32/32

All verdicts agree between configurations. Removing the finite theories
(T_device, T_phase, T_perm) does not change safety/unsafety conclusions
on these benchmarks.

### Frame Counts

| Metric | Full 5-theory | Reduced 2-theory | Delta |
|--------|:---:|:---:|:---:|
| Avg frames | 2.5 | 2.5 | +0.00 |
| Max frames | 3 | 3 | — |

### Timing

| Metric | Full 5-theory | Reduced 2-theory | Delta |
|--------|:---:|:---:|:---:|
| Avg time (ms) | 34.4 | 22.6 | +11.8 |
| Avg Z3 queries | 2.5 | 2.5 | — |

### Statistical Test

Wilcoxon test: Too few nonzero differences (0) for Wilcoxon test

**Wilcoxon signed-rank test (times)**:
W = 35.0000, p = 0.000002 (significant at α=0.05)

### Theory Combination Overhead

- Full: avg 1.0 arrangements checked (max 1)
- Reduced: avg 0.0 arrangements checked

### Per-Category Breakdown

| Category | N | Full Avg Frames | Reduced Avg Frames | Delta | Agree |
|----------|:-:|:---:|:---:|:---:|:---:|
| parametric | 4 | 2.0 | 2.0 | +0.00 | ✓ |
| deep_chain | 5 | 3.0 | 3.0 | +0.00 | ✓ |
| mismatch_depth | 7 | 1.9 | 1.9 | +0.00 | ✓ |
| branching | 5 | 2.2 | 2.2 | +0.00 | ✓ |
| scalability | 8 | 3.0 | 3.0 | +0.00 | ✓ |
| mixed | 3 | 3.0 | 3.0 | +0.00 | ✓ |

## Interpretation

The ablation shows that removing finite theories (T_device, T_phase, T_perm)
has **minimal impact** on IC3/PDR convergence for these benchmarks.
This suggests the ≤3-frame convergence is primarily driven by the shape
constraint structure (T_shape) rather than finite-theory anchoring.

However, the finite theories remain valuable for:
1. **Soundness**: They catch real bugs (device mismatches, phase errors)
   that shape analysis alone misses.
2. **Theory combination completeness**: The Tinelli-Zarba arrangement
   enumeration is essential for correctness when finite and infinite
   theories interact.
3. **Real-world models**: Production code with device transfers and
   train/eval mode switches requires these theories.

## Per-Benchmark Results

| Benchmark | Full Frames | Red. Frames | Δ | Full ms | Red. ms | Agree |
|-----------|:---:|:---:|:---:|:---:|:---:|:---:|
| resnet_basic_block | 3 | 3 | +0 | 103.0 | 71.7 | ✓ |
| resnet_downsample_mismatch | 1 | 1 | +0 | 72.7 | 9.5 | ✓ |
| transformer_encoder_layer | 3 | 3 | +0 | 33.0 | 14.1 | ✓ |
| transformer_dmodel_mismatch | 1 | 1 | +0 | 14.9 | 9.5 | ✓ |
| deep_chain_10layers | 3 | 3 | +0 | 17.4 | 11.8 | ✓ |
| deep_chain_20layers | 3 | 3 | +0 | 21.2 | 15.6 | ✓ |
| deep_chain_50layers | 3 | 3 | +0 | 32.0 | 53.6 | ✓ |
| deep_relu_chain_10layers | 3 | 3 | +0 | 69.6 | 35.8 | ✓ |
| deep_relu_chain_20layers | 3 | 3 | +0 | 51.2 | 36.9 | ✓ |
| mismatch_at_1_of_10 | 1 | 1 | +0 | 15.8 | 9.4 | ✓ |
| mismatch_at_5_of_10 | 1 | 1 | +0 | 13.3 | 8.7 | ✓ |
| mismatch_at_9_of_10 | 3 | 3 | +0 | 17.5 | 12.0 | ✓ |
| mismatch_at_10_of_20 | 1 | 1 | +0 | 16.0 | 10.5 | ✓ |
| mismatch_at_19_of_20 | 3 | 3 | +0 | 26.3 | 29.7 | ✓ |
| mismatch_at_25_of_50 | 1 | 1 | +0 | 53.4 | 22.9 | ✓ |
| mismatch_at_49_of_50 | 3 | 3 | +0 | 47.9 | 37.3 | ✓ |
| skip_connection_2block | 3 | 3 | +0 | 34.2 | 37.1 | ✓ |
| skip_connection_mismatch | 3 | 3 | +0 | 15.6 | 11.4 | ✓ |
| dual_path_merge | 1 | 1 | +0 | 12.7 | 7.8 | ✓ |
| dual_path_shape_mismatch | 1 | 1 | +0 | 12.3 | 6.5 | ✓ |
| deep_residual_10block | 3 | 3 | +0 | 28.8 | 22.9 | ✓ |
| scalability_1layers | 3 | 3 | +0 | 71.9 | 23.6 | ✓ |
| scalability_2layers | 3 | 3 | +0 | 40.7 | 12.2 | ✓ |
| scalability_5layers | 3 | 3 | +0 | 15.1 | 10.1 | ✓ |
| scalability_10layers | 3 | 3 | +0 | 18.2 | 12.1 | ✓ |
| scalability_15layers | 3 | 3 | +0 | 20.3 | 23.0 | ✓ |
| scalability_20layers | 3 | 3 | +0 | 32.6 | 23.5 | ✓ |
| scalability_30layers | 3 | 3 | +0 | 36.0 | 25.2 | ✓ |
| scalability_50layers | 3 | 3 | +0 | 38.0 | 35.3 | ✓ |
| conv_bn_relu_pool_chain | 3 | 3 | +0 | 26.4 | 14.8 | ✓ |
| layernorm_chain | 3 | 3 | +0 | 70.7 | 55.5 | ✓ |
| embedding_to_classifier | 3 | 3 | +0 | 23.1 | 13.8 | ✓ |
