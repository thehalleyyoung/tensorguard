/-
TensorGuard.Symexec.Transfer.AxisOOB

Refutation soundness for the axis-out-of-range check (`_check_axis`,
SymBugKind.AXIS_OUT_OF_RANGE): a normalized non-negative axis `a` is valid for a
tensor of rank `r` iff `a < r`.  The engine reports a bug only when both are
known and `a ≥ r`, reducing to `Core.geBoundFires`.
-/
import TensorGuard.Symexec.Core

namespace TensorGuard
namespace Symexec
namespace AxisOOB

open Dim Core

/-- The runtime precondition: the axis is within `[0, rank)`. -/
def AxisOk (axis rank : Nat) : Prop := axis < rank

/-- The engine's check: known `axis ≥ rank`. -/
def fires (axis rank : Dim) : Bool := geBoundFires axis rank

theorem conservative_left (rank : Dim) : fires .unk rank = false :=
  geBound_conservative_left rank

theorem conservative_right (axis : Dim) : fires axis .unk = false :=
  geBound_conservative_right axis

/-- **Refutation soundness.**  When the check fires, every concretization has
`axis ≥ rank`, i.e. the in-range precondition fails. -/
theorem refute {axis rank : Dim} (h : fires axis rank = true) :
    ∀ a r, dimModels a axis → dimModels r rank → ¬ AxisOk a r := by
  intro a r ha hr
  have hge : a ≥ r := geBound_sound h a r ha hr
  simp only [AxisOk]
  omega

/-- **Certified counterexample.** -/
theorem witness {axis rank : Dim} (h : fires axis rank = true) :
    ∃ a r, dimModels a axis ∧ dimModels r rank ∧ ¬ AxisOk a r := by
  obtain ⟨a, r, ha, hr, hge⟩ := geBound_witness h
  exact ⟨a, r, ha, hr, by simp only [AxisOk]; omega⟩

end AxisOOB
end Symexec
end TensorGuard
