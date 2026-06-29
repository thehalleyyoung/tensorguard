/-
TensorGuard.Symexec.Transfer.BoolNonScalar

Refutation soundness for the boolean-context check
(`_check_bool_context`, SymBugKind.BOOL_ON_NONSCALAR): using a tensor in a
boolean context (``if t:`` / ``while t:`` / ``not t``) requires exactly one
element; the engine reports a bug only when the element count is a *known*
natural number different from `1` (any unknown / symbolic count abstains).

Shares the element-count abstraction shape of `ItemNonScalar`.
-/
import TensorGuard.Symexec.Lattice

namespace TensorGuard
namespace Symexec
namespace BoolNonScalar

/-- A natural-number element-count abstraction: a known `Nat`, or ⊤. -/
inductive NumelAbs
  | known : Nat → NumelAbs
  | unk   : NumelAbs
  deriving DecidableEq, Repr

/-- Concretization membership for `NumelAbs`. -/
def models (c : Nat) : NumelAbs → Prop
  | .known n => c = n
  | .unk     => True

/-- The runtime precondition: a boolean context needs exactly one element. -/
def BoolOk (n : Nat) : Prop := n = 1

/-- The engine's check: a known element count that is provably `≠ 1`. -/
def fires (v : NumelAbs) : Bool :=
  match v with
  | .known n => decide (n ≠ 1)
  | .unk     => false

/-- **Conservativity.** Unknown element counts abstain. -/
theorem conservative : fires .unk = false := rfl

/-- **Refutation soundness.** When the check fires, every concretization of the
element count is `≠ 1`, so the single-element precondition of a boolean context
fails (a `RuntimeError` at runtime). -/
theorem refute {v : NumelAbs} (h : fires v = true) :
    ∀ n, models n v → ¬ BoolOk n := by
  cases v with
  | unk => simp [fires] at h
  | known n0 =>
    intro n hn
    simp only [models] at hn
    subst hn
    simp only [fires, decide_eq_true_eq, ne_eq] at h
    simpa only [BoolOk] using h

/-- **Certified counterexample.** -/
theorem witness {v : NumelAbs} (h : fires v = true) :
    ∃ n, models n v ∧ ¬ BoolOk n := by
  cases v with
  | unk => simp [fires] at h
  | known n0 =>
    refine ⟨n0, by simp [models], ?_⟩
    simp only [fires, decide_eq_true_eq, ne_eq] at h
    simpa only [BoolOk] using h

end BoolNonScalar
end Symexec
end TensorGuard
