/-
TensorGuard.Symexec.Transfer.ItemNonScalar

Refutation soundness for the non-scalar ``.item()`` check
(`_check_item`, SymBugKind.ITEM_ON_NONSCALAR): ``tensor.item()`` requires the
tensor to have exactly one element; the engine reports a bug only when the
element count is a *known* natural number different from `1` (any unknown /
symbolic element count abstains).

Carries its own tiny element-count abstraction (`NumelAbs`) and its
concretization.
-/
import TensorGuard.Symexec.Lattice

namespace TensorGuard
namespace Symexec
namespace ItemNonScalar

/-- A natural-number element-count abstraction: a known `Nat`, or ⊤. -/
inductive NumelAbs
  | known : Nat → NumelAbs
  | unk   : NumelAbs
  deriving DecidableEq, Repr

/-- Concretization membership for `NumelAbs`. -/
def models (c : Nat) : NumelAbs → Prop
  | .known n => c = n
  | .unk     => True

/-- The runtime precondition: ``.item()`` needs exactly one element. -/
def ItemOk (n : Nat) : Prop := n = 1

/-- The engine's check: a known element count that is provably `≠ 1`. -/
def fires (v : NumelAbs) : Bool :=
  match v with
  | .known n => decide (n ≠ 1)
  | .unk     => false

/-- **Conservativity.** Unknown element counts abstain. -/
theorem conservative : fires .unk = false := rfl

/-- **Refutation soundness.** When the check fires, every concretization of the
element count is `≠ 1`, so the single-element precondition fails. -/
theorem refute {v : NumelAbs} (h : fires v = true) :
    ∀ n, models n v → ¬ ItemOk n := by
  cases v with
  | unk => simp [fires] at h
  | known n0 =>
    intro n hn
    simp only [models] at hn
    subst hn
    simp only [fires, decide_eq_true_eq, ne_eq] at h
    simpa only [ItemOk] using h

/-- **Certified counterexample.** -/
theorem witness {v : NumelAbs} (h : fires v = true) :
    ∃ n, models n v ∧ ¬ ItemOk n := by
  cases v with
  | unk => simp [fires] at h
  | known n0 =>
    refine ⟨n0, by simp [models], ?_⟩
    simp only [fires, decide_eq_true_eq, ne_eq] at h
    simpa only [ItemOk] using h

end ItemNonScalar
end Symexec
end TensorGuard
