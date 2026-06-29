/-
TensorGuard.Symexec.Transfer.IndexOOB

Refutation soundness for the index-out-of-bounds checks (`_report_index_oob`,
`_check_tensor_index_bounds`, SymBugKind.TENSOR_INDEX_OOB / RANK_INDEX_ERROR): a
non-negative integer index `i` into a sequence/dimension of length `n` is valid
iff `i < n`.  The engine reports a bug only when both are known and `i ≥ n`,
reducing to `Core.geBoundFires`.
-/
import TensorGuard.Symexec.Core

namespace TensorGuard
namespace Symexec
namespace IndexOOB

open Dim Core

/-- The runtime precondition: the index is within `[0, length)`. -/
def IndexOk (idx length : Nat) : Prop := idx < length

/-- The engine's check: known `idx ≥ length`. -/
def fires (idx length : Dim) : Bool := geBoundFires idx length

theorem conservative_left (length : Dim) : fires .unk length = false :=
  geBound_conservative_left length

theorem conservative_right (idx : Dim) : fires idx .unk = false :=
  geBound_conservative_right idx

/-- **Refutation soundness.** -/
theorem refute {idx length : Dim} (h : fires idx length = true) :
    ∀ i n, dimModels i idx → dimModels n length → ¬ IndexOk i n := by
  intro i n hi hn
  have hge : i ≥ n := geBound_sound h i n hi hn
  simp only [IndexOk]
  omega

/-- **Certified counterexample.** -/
theorem witness {idx length : Dim} (h : fires idx length = true) :
    ∃ i n, dimModels i idx ∧ dimModels n length ∧ ¬ IndexOk i n := by
  obtain ⟨i, n, hi, hn, hge⟩ := geBound_witness h
  exact ⟨i, n, hi, hn, by simp only [IndexOk]; omega⟩

end IndexOOB
end Symexec
end TensorGuard
