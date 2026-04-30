/-
TensorGuard assume/guarantee composition rule (Theorem 3, weak form),
machine-checked in Lean 4.

This file extends the tiny shape DSL of `Soundness.lean` with the
assume/guarantee composition rule used at the `nn.Module` class
boundary (paper §3, Theorem 3).  We mechanise the *weak* form:
sequential composition of two operator chains is sound when the
upstream chain's *guarantee* on its output shape implies the
downstream chain's *assume* on its input shape.

This is the load-bearing piece of TG's class-boundary reasoning that
the round-1 NeurIPS reviewer flagged as currently pen-and-paper
(W3, "the assume/guarantee composition rule … is explicitly *not*
mechanised").  It is intentionally proved over the same minimal
3-operator DSL as `Soundness.lean` so the audit chain is uniform;
extending the proof to the full 79-handler Python rule-set is
out-of-scope for round 1 and is tracked in the project obligations.

Pure Lean 4 core (no mathlib).
-/

import TensorGuard.Soundness

namespace TensorGuard

/-- An *operator chain* is a list of operators applied left to right.
    `applyChain` returns `some s'` iff every operator in the chain
    successfully transitions, and `none` otherwise. -/
def applyChain : List Op → Shape → Option Shape
  | [],      s => some s
  | op :: r, s =>
      match applyOp op s with
      | none    => none
      | some s' => applyChain r s'

/-- A `Contract` is a pair (assume, guarantee) of predicates over
    shapes; `assume` constrains the chain's input, `guarantee`
    constrains the chain's output.  The trivial contract used as a
    sanity check is `(fun _ => True, fun _ => True)`. -/
structure Contract where
  assume    : Shape → Prop
  guarantee : Shape → Prop

/-- A chain `c : List Op` *satisfies* a contract if, for every input
    shape that meets the assume, the chain succeeds and the verdict
    meets the guarantee. -/
def satisfies (c : List Op) (k : Contract) : Prop :=
  ∀ s, k.assume s →
    ∃ s', applyChain c s = some s' ∧ k.guarantee s'

/-- Concatenation of operator chains threaded through `applyChain`. -/
theorem applyChain_append :
    ∀ (c1 c2 : List Op) (s : Shape),
      applyChain (c1 ++ c2) s =
        (match applyChain c1 s with
         | none    => none
         | some s' => applyChain c2 s')
  | [],         c2, s => by
      simp [applyChain]
  | op :: r,    c2, s => by
      simp [applyChain]
      cases h : applyOp op s with
      | none =>
          simp [h]
      | some s' =>
          simp [h]
          exact applyChain_append r c2 s'

/-- **Assume/guarantee composition (Theorem 3, weak form).**
    If `c1` satisfies `(A, G)` and `c2` satisfies `(A', G')` and
    `G` implies `A'` pointwise on shapes, then the concatenation
    `c1 ++ c2` satisfies `(A, G')`.  This is the load-bearing
    soundness lemma at the `nn.Module` class boundary. -/
theorem ag_composition
    (c1 c2 : List Op)
    (k1 k2 : Contract)
    (h1 : satisfies c1 k1)
    (h2 : satisfies c2 k2)
    (link : ∀ s, k1.guarantee s → k2.assume s) :
    satisfies (c1 ++ c2) ⟨k1.assume, k2.guarantee⟩ := by
  intro s ha
  -- Run c1 to obtain an intermediate shape s' meeting k1.guarantee
  obtain ⟨s', hc1, hg1⟩ := h1 s ha
  -- Use the link to obtain that s' meets k2.assume, then run c2
  have ha' : k2.assume s' := link s' hg1
  obtain ⟨s'', hc2, hg2⟩ := h2 s' ha'
  refine ⟨s'', ?_, hg2⟩
  -- Glue the two runs through applyChain_append
  rw [applyChain_append, hc1]
  simp [hc2]

/-- **Reflexive specialisation.** The trivial contract (assume/guarantee
    both `True`) is satisfied by every chain that does not abort. -/
theorem satisfies_trivial
    (c : List Op)
    (h : ∀ s, ∃ s', applyChain c s = some s') :
    satisfies c ⟨fun _ => True, fun _ => True⟩ := by
  intro s _
  obtain ⟨s', hs'⟩ := h s
  exact ⟨s', hs', trivial⟩

/-- **Empty chain is the identity contract.** -/
theorem satisfies_nil :
    satisfies [] ⟨fun s => True, fun s => True⟩ := by
  intro s _
  refine ⟨s, ?_, trivial⟩
  simp [applyChain]

end TensorGuard
