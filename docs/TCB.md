# Trusted Computing Base (TCB)

What TensorGuard verifies at each assurance level.

## Lean 4 Mechanized (highest assurance)

| Property | Status |
|----------|--------|
| Theory combination soundness (Nelson-Oppen procedure correctness) | ✅ Proved |
| Product domain lattice properties (monotonicity, distributivity) | ✅ Proved |
| Shape × Device × Phase × Stride × Permutation domain well-formedness | ✅ Proved |
| IC3/PDR termination for finite-state theories | ✅ Proved |

- **Total**: ~1,587 lines, 71 theorems
- **File**: `lean/TheoryCombination.lean`
- **Status**: Zero `sorry` (all proofs complete)

## Python Implementation (tested, not mechanized)

| Component | File |
|-----------|------|
| Shape transfer functions for 300+ PyTorch operators | `src/stdlib/modern_ops.py` |
| AST-based shape predicate harvesting | `src/tensor_shapes.py` |
| Z3 constraint encoding and checking | `src/smt/z3_backend.py` |
| FX graph extraction from `nn.Module` | `src/fx_extractor.py` |
| CEGAR loop | `src/shape_cegar.py`, `src/cegar_cpa.py` |
| IC3/PDR implementation | `src/ic3_pdr.py` |

- **Testing**: 6000+ unit tests

## Assumptions (not verified)

- Z3 solver correctness
- Python runtime correctness
- PyTorch FX tracer faithfulness
- AST parsing correctness (Python `ast` module)
- OS/hardware correctness

## Gap Analysis

- Lean mechanization proves theory combination is sound **in theory**.
- Python implementation **uses** that theory but is not extracted from Lean.
- **The gap**: Lean proves the abstract algorithm correct; Python implements it separately.
- **Risk**: implementation bugs could violate the properties Lean proves.
- **Mitigation**: extensive testing, property-based testing with Hypothesis.
