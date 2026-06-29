/-
TensorGuard.Symexec.Transfer.EinopsRankMismatch

Refutation soundness for the einops structural rank check
(`_check_einops`, SymBugKind.EINOPS_PATTERN_MISMATCH): a literal, ellipsis-free
``einops.rearrange/reduce/repeat(t, 'lhs -> rhs')`` pattern fixes the number of
top-level LHS groups; the input tensor's rank must equal it, or einops raises an
``EinopsError``.  Within the modeled (literal, ellipsis-free) fragment the group
count is a *known* `Nat` read off the pattern, so the engine reports the bug
whenever the input rank is also known and differs (an unknown rank abstains).

The witnessed precondition is `arity_match` (the produced group arity must equal
the tensor rank).  Carries its own rank abstraction (`RankAbs`); the literal
group count rides along as a known `Nat`, mirroring the engine, which only fires
when both are concrete.
-/
import TensorGuard.Symexec.Lattice

namespace TensorGuard
namespace Symexec
namespace EinopsRankMismatch

/-- A natural-number rank abstraction: a known `Nat`, or ⊤. -/
inductive RankAbs
  | known : Nat → RankAbs
  | unk   : RankAbs
  deriving DecidableEq, Repr

/-- Concretization membership for `RankAbs`. -/
def models (c : Nat) : RankAbs → Prop
  | .known n => c = n
  | .unk     => True

/-- The runtime precondition: the input rank must equal the LHS group count. -/
def RankOk (r groups : Nat) : Prop := r = groups

/-- The engine's check: a known rank different from the known group count. -/
def fires (v : RankAbs) (groups : Nat) : Bool :=
  match v with
  | .known r => decide (r ≠ groups)
  | .unk     => false

/-- **Conservativity.** An unknown rank abstains. -/
theorem conservative (groups : Nat) : fires .unk groups = false := rfl

/-- **Refutation soundness.** When the check fires, every concretization of the
rank differs from the LHS group count, so the einops precondition fails. -/
theorem refute {v : RankAbs} {groups : Nat} (h : fires v groups = true) :
    ∀ r, models r v → ¬ RankOk r groups := by
  cases v with
  | unk => simp [fires] at h
  | known r0 =>
    intro r hr
    simp only [models] at hr
    subst hr
    simp only [fires, decide_eq_true_eq, ne_eq] at h
    simpa only [RankOk] using h

/-- **Certified counterexample.** -/
theorem witness {v : RankAbs} {groups : Nat} (h : fires v groups = true) :
    ∃ r, models r v ∧ ¬ RankOk r groups := by
  cases v with
  | unk => simp [fires] at h
  | known r0 =>
    refine ⟨r0, by simp [models], ?_⟩
    simp only [fires, decide_eq_true_eq, ne_eq] at h
    simpa only [RankOk] using h

end EinopsRankMismatch
end Symexec
end TensorGuard
