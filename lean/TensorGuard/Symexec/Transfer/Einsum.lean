/-
TensorGuard.Symexec.Transfer.Einsum

Refutation soundness for the einsum repeated-index check (`_check_einsum`,
SymBugKind.EINSUM_DIM_MISMATCH): a label that appears in more than one operand
must bind equal dimensions.  An equality requirement, so it reduces to
`Core.disagreeFires`.
-/
import TensorGuard.Symexec.Core

namespace TensorGuard
namespace Symexec
namespace Einsum

open Dim Core

/-- The runtime precondition: the two dims bound to a shared label agree. -/
def LabelOk (x y : Nat) : Prop := x = y

/-- The engine's check. -/
def fires (a b : Dim) : Bool := disagreeFires a b

theorem conservative_left (b : Dim) : fires .unk b = false :=
  disagree_conservative_left b

theorem conservative_right (a : Dim) : fires a .unk = false :=
  disagree_conservative_right a

/-- **Refutation soundness.** -/
theorem refute {a b : Dim} (h : fires a b = true) :
    ∀ x y, dimModels x a → dimModels y b → ¬ LabelOk x y := by
  intro x y hx hy
  exact disagree_sound h x y hx hy

/-- **Certified counterexample.** -/
theorem witness {a b : Dim} (h : fires a b = true) :
    ∃ x y, dimModels x a ∧ dimModels y b ∧ ¬ LabelOk x y :=
  disagree_witness h

end Einsum
end Symexec
end TensorGuard
