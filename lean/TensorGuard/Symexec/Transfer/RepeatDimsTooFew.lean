/-
TensorGuard.Symexec.Transfer.RepeatDimsTooFew

Refutation soundness for the ``tensor.repeat(*sizes)`` rank check
(`_check_repeat`, SymBugKind.REPEAT_DIMS_TOO_FEW): ``repeat`` requires the number
of provided repeat dims to be **at least** the tensor's rank; torch raises a
``RuntimeError`` otherwise.  The engine reports a bug only when the tensor's rank
is a *known* natural number and the *statically-known* number of repeat dims
(`ndims`) is strictly less than it (any unknown rank or unknown dim count
abstains).

Carries its own tiny rank abstraction (`RankAbs`) and its concretization.
-/
import TensorGuard.Symexec.Lattice

namespace TensorGuard
namespace Symexec
namespace RepeatDimsTooFew

/-- A natural-number rank abstraction: a known `Nat`, or ⊤. -/
inductive RankAbs
  | known : Nat → RankAbs
  | unk   : RankAbs
  deriving DecidableEq, Repr

/-- Concretization membership for `RankAbs`. -/
def models (c : Nat) : RankAbs → Prop
  | .known n => c = n
  | .unk     => True

/-- The runtime precondition: the number of repeat dims must be ≥ the rank. -/
def RepeatOk (ndims r : Nat) : Prop := r ≤ ndims

/-- The engine's check: a known rank strictly greater than the known dim count. -/
def fires (v : RankAbs) (ndims : Nat) : Bool :=
  match v with
  | .known r => decide (ndims < r)
  | .unk     => false

/-- **Conservativity.** An unknown rank abstains. -/
theorem conservative (ndims : Nat) : fires .unk ndims = false := rfl

/-- **Refutation soundness.** When the check fires, every concretization of the
rank exceeds the supplied dim count, so the repeat precondition fails. -/
theorem refute {v : RankAbs} {ndims : Nat} (h : fires v ndims = true) :
    ∀ r, models r v → ¬ RepeatOk ndims r := by
  cases v with
  | unk => simp [fires] at h
  | known r0 =>
    intro r hr
    simp only [models] at hr
    subst hr
    simp only [fires, decide_eq_true_eq] at h
    simp only [RepeatOk, Nat.not_le]
    exact h

/-- **Certified counterexample.** -/
theorem witness {v : RankAbs} {ndims : Nat} (h : fires v ndims = true) :
    ∃ r, models r v ∧ ¬ RepeatOk ndims r := by
  cases v with
  | unk => simp [fires] at h
  | known r0 =>
    refine ⟨r0, by simp [models], ?_⟩
    simp only [fires, decide_eq_true_eq] at h
    simp only [RepeatOk, Nat.not_le]
    exact h

end RepeatDimsTooFew
end Symexec
end TensorGuard
