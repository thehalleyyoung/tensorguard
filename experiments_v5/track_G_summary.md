# Track G — V5 Lean ↔ PyTorch operator parity

## TL;DR

- **28/28 operators** have a Lean shape-transfer rule + soundness
  lemma in `lean/TensorGuard/V5OperatorRules.lean`.
- **28/28 rules build** under `lake env lean TensorGuard/V5OperatorRules.lean`
  (zero errors; eight warnings, all `declaration uses 'sorry'` on
  interface lemmas — same pattern as the existing `Extended.lean` /
  `Parity.lean`).
- **28 000 / 28 000** randomly-sampled in-fragment cases agree
  with PyTorch (1.0000 overall agreement rate).
- Rule registry is *exported by Lean* via
  `lake env lean --run TensorGuard/V5OperatorRules.lean`, so Python
  cannot silently invent an operator the Lean side does not declare.

## Files

| Path                                                | What                                       |
|-----------------------------------------------------|--------------------------------------------|
| `lean/TensorGuard/V5OperatorRules.lean`             | 28 rules + soundness lemmas + JSON exporter |
| `experiments_v5/run_lean_parity_v5.py`              | Parity harness (loads Lean registry, runs torch) |
| `experiments_v5/lean_parity_v5_results.json`        | Per-op tallies (1000 cases each)           |
| `docs/paper/sections_v5/G_lean_parity.tex`          | One-page paper section + table             |

## Operators (all rate=1.0000, 1000/1000 cases)

`matmul, bmm, batched_matmul, conv1d, conv2d, conv3d,
conv_transpose2d, view, reshape, permute, transpose, expand,
repeat, broadcast_to, cat, stack, split, chunk, unbind, gather,
scatter, index_select, narrow, embed, layer_norm, rms_norm,
scaled_dot_product_attention, linear`

## Reproduce

```bash
cd lean && lake build TensorGuard
cd lean && lake env lean TensorGuard/V5OperatorRules.lean   # type-check V5
cd lean && lake env lean --run TensorGuard/V5OperatorRules.lean  # print JSON registry
python3.11 experiments_v5/run_lean_parity_v5.py             # run parity (~30 s)
```

## Calibrated honesty

- All 28 rules are *defined* sorry-free.
- 8 *soundness lemmas* use `sorry` (same convention as
  `TensorGuard/Extended.lean` and `TensorGuard/Parity.lean`):
  `matmul_sound_eqbatch`, `conv1d_sound`, `conv2d_some_iff`,
  `broadcast_to_eq`, `gather_eq`, `scatter_eq`, `layer_norm_id`,
  `rms_norm_id`. These are interface-level statements relating the
  rule's output to a reference spec; they are operationally backed by
  the 28 000-case empirical audit, not by formal proof.
- We do not modify any pre-existing Lean file (`Soundness.lean`,
  `Extended.lean`, `Parity.lean`, `TensorGuard.lean`,
  `ParityRunner.lean`, `TheoryCombination.lean` are untouched).
  V5 lives entirely in the new module.
- The harness counts mutually-rejected cases (mirror=None ∧ torch
  raises) as agreement, mutually-accepted matching shapes as
  agreement, and any other combination as disagreement.

## Distribution

- Per-dim ∈ [1, 16], rank ∈ [1, 5] for generic ops.
- Conv ops use slightly larger spatial dims (4..16) and small kernel
  / stride / dilation / padding to keep cases in-fragment.
- 1000 tests per operator; seed = `0x5C5C5C5C` (deterministic).
