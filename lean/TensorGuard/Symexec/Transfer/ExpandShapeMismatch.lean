/-
TensorGuard.Symexec.Transfer.ExpandShapeMismatch

Refutation soundness for the ``tensor.expand(*sizes)`` non-singleton check
(`_check_expand`, SymBugKind.EXPAND_SHAPE_MISMATCH): when an *existing*
dimension is aligned with a target size, ``expand`` requires the target to keep
the dimension (target `-1`), or the existing dimension to be a singleton (`1`),
or the target to equal the existing size; otherwise torch raises a
``RuntimeError`` ("The expanded size of the tensor must match the existing
size...").  The engine reports a bug only when the existing dimension is a
*known* `Nat` and the target is a *known* `Int` for which all three escape
clauses fail (any unknown operand abstains).

Carries its own tiny existing-dimension abstraction (`DimAbs`); the (known)
target size rides along as an `Int` parameter, mirroring the engine, which only
fires when both operands are concrete.
-/
import TensorGuard.Symexec.Lattice

namespace TensorGuard
namespace Symexec
namespace ExpandShapeMismatch

/-- A natural-number existing-dimension abstraction: a known `Nat`, or ⊤. -/
inductive DimAbs
  | known : Nat → DimAbs
  | unk   : DimAbs
  deriving DecidableEq, Repr

/-- Concretization membership for `DimAbs`. -/
def models (c : Nat) : DimAbs → Prop
  | .known n => c = n
  | .unk     => True

/-- The runtime precondition for an aligned existing dim `d` and target `t`:
keep (`t = -1`), singleton-expand (`d = 1`), or exact match (`t = d`). -/
def ExpandOk (d : Nat) (t : Int) : Prop :=
  t = (d : Int) ∨ d = 1 ∨ t = -1

/-- The engine's check: a known existing dim that is non-singleton while the
known target is neither `-1` nor equal to it. -/
def fires (v : DimAbs) (t : Int) : Bool :=
  match v with
  | .known d => decide (d ≠ 1 ∧ t ≠ -1 ∧ t ≠ (d : Int))
  | .unk     => false

/-- **Conservativity.** An unknown existing dimension abstains. -/
theorem conservative (t : Int) : fires .unk t = false := rfl

/-- **Refutation soundness.** When the check fires, every concretization of the
existing dimension violates the expand precondition for the supplied target. -/
theorem refute {v : DimAbs} {t : Int} (h : fires v t = true) :
    ∀ d, models d v → ¬ ExpandOk d t := by
  cases v with
  | unk => simp [fires] at h
  | known d0 =>
    intro d hd
    simp only [models] at hd
    subst hd
    simp only [fires, decide_eq_true_eq] at h
    obtain ⟨hne1, hnem1, hned⟩ := h
    simp only [ExpandOk, not_or]
    exact ⟨hned, hne1, hnem1⟩

/-- **Certified counterexample.** -/
theorem witness {v : DimAbs} {t : Int} (h : fires v t = true) :
    ∃ d, models d v ∧ ¬ ExpandOk d t := by
  cases v with
  | unk => simp [fires] at h
  | known d0 =>
    refine ⟨d0, by simp [models], ?_⟩
    simp only [fires, decide_eq_true_eq] at h
    obtain ⟨hne1, hnem1, hned⟩ := h
    simp only [ExpandOk, not_or]
    exact ⟨hned, hne1, hnem1⟩

end ExpandShapeMismatch
end Symexec
end TensorGuard
