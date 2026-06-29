/-
TensorGuard.Symexec.Transfer.NegativeDim

Refutation soundness for the non-positive-dimension construction check
(`_check_negative_dim`, SymBugKind.NEGATIVE_DIMENSION): a tensor constructor
size must be `≥ 0`; the engine reports a bug only when the size is a *known*
integer that is provably `< 0` (a `0` or symbolic / `-1`-inference size
abstains).

Because sizes here range over the integers, this file carries its own tiny
integer-dim abstraction (`IZ`) and its concretization, rather than reusing the
`Nat`-valued `Dim`.
-/
import TensorGuard.Symexec.Lattice

namespace TensorGuard
namespace Symexec
namespace NegativeDim

/-- An integer-valued dimension abstraction: a known `Int`, or ⊤. -/
inductive IZ
  | known : Int → IZ
  | unk   : IZ
  deriving DecidableEq, Repr

/-- Concretization membership for `IZ`. -/
def izModels (c : Int) : IZ → Prop
  | .known n => c = n
  | .unk     => True

/-- The runtime precondition: a constructor dimension is non-negative. -/
def DimOk (n : Int) : Prop := 0 ≤ n

/-- The engine's check: a known size that is provably `< 0`. -/
def fires (v : IZ) : Bool :=
  match v with
  | .known n => decide (n < 0)
  | .unk     => false

/-- **Conservativity.**  Unknown sizes abstain. -/
theorem conservative : fires .unk = false := rfl

/-- **Refutation soundness.**  When the check fires, every concretization of the
size is `< 0`, so the non-negativity precondition fails. -/
theorem refute {v : IZ} (h : fires v = true) :
    ∀ n, izModels n v → ¬ DimOk n := by
  cases v with
  | unk => simp [fires] at h
  | known n0 =>
    intro n hn
    simp only [izModels] at hn
    subst hn
    simp only [fires, decide_eq_true_eq] at h
    simp only [DimOk]
    omega

/-- **Certified counterexample.** -/
theorem witness {v : IZ} (h : fires v = true) :
    ∃ n, izModels n v ∧ ¬ DimOk n := by
  cases v with
  | unk => simp [fires] at h
  | known n0 =>
    refine ⟨n0, by simp [izModels], ?_⟩
    simp only [fires, decide_eq_true_eq] at h
    simp only [DimOk]; omega

end NegativeDim
end Symexec
end TensorGuard
