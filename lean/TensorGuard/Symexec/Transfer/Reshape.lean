/-
TensorGuard.Symexec.Transfer.Reshape

Refutation soundness for the reshape numel check (`_check_reshape`,
SymBugKind.RESHAPE_SIZE_MISMATCH): a reshape with no `-1` placeholder requires
the input element count to equal the product of the requested dims.  An
equality requirement over the two (known) element counts, so it reduces to
`Core.disagreeFires`.

We additionally connect the abstraction to the concrete element count by
defining `prod` over concrete shapes, mirroring `Shape.prod` used elsewhere in
the development.
-/
import TensorGuard.Symexec.Core

namespace TensorGuard
namespace Symexec
namespace Reshape

open Dim Core

/-- Element count of a concrete shape. -/
def prod : CShape → Nat
  | []      => 1
  | n :: ns => n * prod ns

/-- The runtime precondition: input numel equals the target numel. -/
def ReshapeOk (inNumel targetNumel : Nat) : Prop := inNumel = targetNumel

/-- The engine's check, over the two known element counts. -/
def fires (a b : Dim) : Bool := disagreeFires a b

theorem conservative_left (b : Dim) : fires .unk b = false :=
  disagree_conservative_left b

theorem conservative_right (a : Dim) : fires a .unk = false :=
  disagree_conservative_right a

/-- **Refutation soundness.**  When the check fires, no concretization of the
two element counts can be equal, so the reshape is unrunnable. -/
theorem refute {a b : Dim} (h : fires a b = true) :
    ∀ x y, dimModels x a → dimModels y b → ¬ ReshapeOk x y := by
  intro x y hx hy
  exact disagree_sound h x y hx hy

/-- **Certified counterexample.** -/
theorem witness {a b : Dim} (h : fires a b = true) :
    ∃ x y, dimModels x a ∧ dimModels y b ∧ ¬ ReshapeOk x y :=
  disagree_witness h

/-- A reshape between two concrete shapes is runnable only if their element
counts match — the property the abstract check refutes. -/
theorem reshape_spec (s t : CShape) :
    ReshapeOk (prod s) (prod t) ↔ prod s = prod t := Iff.rfl

end Reshape
end Symexec
end TensorGuard
