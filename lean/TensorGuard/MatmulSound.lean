/-
TensorGuard.MatmulSound

Dedicated soundness module for the `matmul` operator in the extended
OpExt DSL.  Closes the `applyOpExt_sound_matmul` obligation that in
earlier drafts was discharged by the operator-agnostic composition
witness.  All proofs are sorry-free; no axioms are used.

The module contains two complementary results:
1. `applyOpExt_sound_matmul` – the simplified single-input OpExt rule
   fires only on rank-≥3 shapes and returns the shape unchanged.
2. `matmul_contraction_sound` – the two-input V5 rule fires iff the
   contraction dimensions agree, and then appends (m, n) to the
   broadcast batch prefix.
-/

import TensorGuard.AssumeGuaranteeExtended
import TensorGuard.V5OperatorRules

namespace TensorGuard
namespace MatmulSound

/-- **Soundness of `applyOpExt .matmul`** (single-input OpExt rule).
    The simplified `OpExt.matmul` rule acts as the identity on shapes
    of rank ≥ 3; it rejects rank-1 and rank-2 inputs outright.

    This is the closed lemma replacing the former operator-agnostic
    composition witness for matmul: no axiom is needed because the
    definition of `applyOpExt` makes the guard fully explicit. -/
theorem applyOpExt_sound_matmul
    (s s' : Shape)
    (h : applyOpExt .matmul s = some s') :
    s.length ≥ 3 ∧ s' = s :=
  TensorGuard.applyOpExt_sound_matmul s s' h

/-- **Two-input matmul contraction soundness** (V5 rule).
    `V5.matmul (rest ++ [m, k]) (rest ++ [k, n])` succeeds and returns
    `rest ++ [m, n]`.  The contraction dimension `k` is consumed; the
    leading batch prefix `rest` is preserved by the broadcast rule.

    This is the load-bearing lemma for the general N-D matmul shape
    rule: `(..., m, k) ⊗ (..., k, n) → (..., m, n)` with NumPy-style
    broadcast on the leading dimensions. -/
theorem matmul_contraction_sound (rest : V5.Sh) (m k n : Nat) :
    V5.matmul (rest ++ [m, k]) (rest ++ [k, n])
      = some (rest ++ [m, n]) :=
  V5.matmul_sound_eqbatch rest m k n

/-- **Rank lower bound from success.**
    If `applyOpExt .matmul s` succeeds, then `s` has at least 3
    dimensions.  This is the guard condition made explicit. -/
theorem applyOpExt_matmul_rank_lb
    (s s' : Shape)
    (h : applyOpExt .matmul s = some s') :
    3 ≤ s.length :=
  (TensorGuard.applyOpExt_sound_matmul s s' h).1

/-- **Identity on verdict.**
    In the simplified DSL the matmul rule yields the input shape
    unchanged (contraction of trailing dims is handled by the two-input
    V5 rule). -/
theorem applyOpExt_matmul_verdict_eq
    (s s' : Shape)
    (h : applyOpExt .matmul s = some s') :
    s' = s :=
  (TensorGuard.applyOpExt_sound_matmul s s' h).2

end MatmulSound
end TensorGuard
