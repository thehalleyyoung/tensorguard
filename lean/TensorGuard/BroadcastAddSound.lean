/-
TensorGuard.BroadcastAddSound

Dedicated soundness module for the `broadcast_add` operator in the
extended OpExt DSL.  Closes the `applyOpExt_sound_broadcast_add`
obligation that in earlier drafts was discharged by the
operator-agnostic composition witness.  All proofs are sorry-free;
no axioms are used.

The module contains:
1. `applyOpExt_sound_broadcast_add` – the OpExt rule is the identity
   on all shapes; the verdict equals the input shape.
2. `bcast_add_comm` – commutativity of the underlying per-dimension
   broadcast rule, derived from `bcastDim_comm` / `bcast_comm`.
3. `broadcast_add_total` – `applyOpExt .broadcast_add` succeeds on
   every input shape (totality).
-/

import TensorGuard.AssumeGuaranteeExtended
import TensorGuard.Extended

namespace TensorGuard
namespace BroadcastAddSound

/-- **Soundness of `applyOpExt .broadcast_add`**.
    The simplified `OpExt.broadcast_add` rule is the identity: it
    succeeds on every shape and returns that shape unchanged.

    This is the closed lemma replacing the former operator-agnostic
    composition witness for broadcast_add: no axiom is needed because
    the definition `applyOpExt .broadcast_add s = some s` is
    unconditional. -/
theorem applyOpExt_sound_broadcast_add
    (s s' : Shape)
    (h : applyOpExt .broadcast_add s = some s') :
    s' = s :=
  TensorGuard.applyOpExt_sound_broadcast_add s s' h

/-- **Totality of `applyOpExt .broadcast_add`**.
    The rule fires on every input shape.  This confirms that
    `broadcast_add` is not a partial function in the DSL; it can
    never block a shape-propagation chain. -/
theorem broadcast_add_total (s : Shape) :
    ∃ s', applyOpExt .broadcast_add s = some s' :=
  ⟨s, by simp [applyOpExt]⟩

/-- **Commutativity of the binary broadcast rule.**
    The two-operand `bcast` function (from `TensorGuard.Extended`) is
    commutative.  This is the key symmetry that makes `broadcast_add`
    order-independent with respect to its two operands.

    The proof follows directly from `bcast_comm`, which is proved by
    structural induction using `bcastDim_comm` at each dimension. -/
theorem bcast_add_comm (s₁ s₂ : Shape) :
    bcast s₁ s₂ = bcast s₂ s₁ :=
  bcast_comm s₁ s₂

/-- **Verdict-identity corollary.**
    If `broadcast_add` succeeds with verdict `s'`, then `s' = s`.
    Useful in contexts where the input shape is not immediately known
    and one wants to rewrite the verdict. -/
theorem broadcast_add_verdict_eq
    (s s' : Shape)
    (h : applyOpExt .broadcast_add s = some s') :
    s = s' :=
  (TensorGuard.applyOpExt_sound_broadcast_add s s' h).symm

end BroadcastAddSound
end TensorGuard
