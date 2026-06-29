/-
TensorGuard.Symexec.Transfer.CatStack

Refutation soundness for the cat/stack shape check (`_check_cat_stack`,
SymBugKind.CAT_SHAPE_MISMATCH): inputs must agree on every must-match dim (all
axes for `stack`, the non-concat axes for `cat`).  The requirement is an
*equality* between two dims, so it reduces to `Core.disagreeFires`.
-/
import TensorGuard.Symexec.Core

namespace TensorGuard
namespace Symexec
namespace CatStack

open Dim Core

/-- The runtime precondition: the two must-match dims agree. -/
def MustMatchOk (x y : Nat) : Prop := x = y

/-- The engine's check. -/
def fires (a b : Dim) : Bool := disagreeFires a b

theorem conservative_left (b : Dim) : fires .unk b = false :=
  disagree_conservative_left b

theorem conservative_right (a : Dim) : fires a .unk = false :=
  disagree_conservative_right a

/-- **Refutation soundness.** -/
theorem refute {a b : Dim} (h : fires a b = true) :
    ∀ x y, dimModels x a → dimModels y b → ¬ MustMatchOk x y := by
  intro x y hx hy
  exact disagree_sound h x y hx hy

/-- **Certified counterexample.** -/
theorem witness {a b : Dim} (h : fires a b = true) :
    ∃ x y, dimModels x a ∧ dimModels y b ∧ ¬ MustMatchOk x y :=
  disagree_witness h

end CatStack
end Symexec
end TensorGuard
