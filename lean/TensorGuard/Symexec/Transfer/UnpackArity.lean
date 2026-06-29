/-
TensorGuard.Symexec.Transfer.UnpackArity

Refutation soundness for the tuple/return unpack-arity check (`_report_unpack`,
SymBugKind.UNPACK_ARITY_MISMATCH): the produced tuple arity must equal the
number of targets.  An equality requirement, so it reduces to
`Core.disagreeFires` (over arity counts carried as `Dim`).
-/
import TensorGuard.Symexec.Core

namespace TensorGuard
namespace Symexec
namespace UnpackArity

open Dim Core

/-- The runtime precondition: the produced arity equals the target count. -/
def ArityOk (produced targets : Nat) : Prop := produced = targets

/-- The engine's check. -/
def fires (a b : Dim) : Bool := disagreeFires a b

theorem conservative_left (b : Dim) : fires .unk b = false :=
  disagree_conservative_left b

theorem conservative_right (a : Dim) : fires a .unk = false :=
  disagree_conservative_right a

/-- **Refutation soundness.** -/
theorem refute {a b : Dim} (h : fires a b = true) :
    ∀ x y, dimModels x a → dimModels y b → ¬ ArityOk x y := by
  intro x y hx hy
  exact disagree_sound h x y hx hy

/-- **Certified counterexample.** -/
theorem witness {a b : Dim} (h : fires a b = true) :
    ∃ x y, dimModels x a ∧ dimModels y b ∧ ¬ ArityOk x y :=
  disagree_witness h

end UnpackArity
end Symexec
end TensorGuard
