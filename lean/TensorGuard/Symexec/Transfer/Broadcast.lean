/-
TensorGuard.Symexec.Transfer.Broadcast

Refutation soundness for the elementwise-broadcast check (`_check_broadcast`,
SymBugKind.BROADCAST_MISMATCH).  Two aligned trailing dims are broadcast-
compatible iff they are equal or one of them is 1.  The engine reports a bug
only when both are known, both `> 1`, and unequal.
-/
import TensorGuard.Symexec.Core

namespace TensorGuard
namespace Symexec
namespace Broadcast

open Dim Core

/-- Runtime broadcast compatibility of two aligned dims. -/
def BroadcastOk (a b : Nat) : Prop := a = b ∨ a = 1 ∨ b = 1

/-- The check: both dims known, both `> 1`, and unequal. -/
def fires (da db : Dim) : Bool :=
  match da, db with
  | .known a, .known b => (a > 1) && (b > 1) && !(a = b)
  | _,        _        => false

theorem conservative_left (db : Dim) : fires .unk db = false := by
  cases db <;> rfl

theorem conservative_right (da : Dim) : fires da .unk = false := by
  cases da <;> rfl

/-- **Refutation soundness.**  When the check fires, every concretization of the
two dims is broadcast-incompatible. -/
theorem broadcast_refute {da db : Dim} (h : fires da db = true) :
    ∀ a b, dimModels a da → dimModels b db → ¬ BroadcastOk a b := by
  cases da with
  | unk => simp [fires] at h
  | known a0 =>
    cases db with
    | unk => simp [fires] at h
    | known b0 =>
      intro a b h1 h2
      simp only [dimModels] at h1 h2
      subst h1; subst h2
      simp only [fires, Bool.and_eq_true, decide_eq_true_eq] at h
      obtain ⟨⟨ha, hb⟩, hne⟩ := h
      simp only [BroadcastOk, not_or]
      refine ⟨?_, ?_, ?_⟩
      · simpa using hne
      · omega
      · omega

/-- **Certified counterexample.** -/
theorem broadcast_witness {da db : Dim} (h : fires da db = true) :
    ∃ a b, dimModels a da ∧ dimModels b db ∧ ¬ BroadcastOk a b := by
  cases da with
  | unk => simp [fires] at h
  | known a0 =>
    cases db with
    | unk => simp [fires] at h
    | known b0 =>
      exact ⟨a0, b0, by simp [dimModels], by simp [dimModels],
        broadcast_refute h a0 b0 (by simp [dimModels]) (by simp [dimModels])⟩

end Broadcast
end Symexec
end TensorGuard
