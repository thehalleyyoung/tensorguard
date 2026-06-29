/-
TensorGuard.Symexec.Transfer.DivZero

Refutation soundness for the division-by-zero check (`_check_div_by_zero`,
SymBugKind.DIVISION_BY_ZERO): an integer division/modulo requires a non-zero
divisor; the engine reports a bug only when the divisor is the *known* constant
`0`, reducing to `Core.zeroFires`.
-/
import TensorGuard.Symexec.Core

namespace TensorGuard
namespace Symexec
namespace DivZero

open Dim Core

/-- The runtime precondition: the divisor is non-zero. -/
def DivisorOk (d : Nat) : Prop := d ≠ 0

/-- The engine's check: the divisor is the known constant `0`. -/
def fires (v : Dim) : Bool := zeroFires v

theorem conservative : fires .unk = false := zero_conservative

/-- **Refutation soundness.**  When the check fires, every concretization of the
divisor is `0`, so the non-zero precondition fails. -/
theorem refute {v : Dim} (h : fires v = true) :
    ∀ d, dimModels d v → ¬ DivisorOk d := by
  intro d hd
  have hz : d = 0 := zero_sound h d hd
  simp only [DivisorOk]
  omega

/-- **Certified counterexample.** -/
theorem witness {v : Dim} (h : fires v = true) :
    ∃ d, dimModels d v ∧ ¬ DivisorOk d := by
  obtain ⟨d, hd, hz⟩ := zero_witness h
  exact ⟨d, hd, by simp only [DivisorOk]; omega⟩

end DivZero
end Symexec
end TensorGuard
