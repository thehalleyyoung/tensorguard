/-
TensorGuard.Symexec.Transfer.RequiresGradNonFloat

Refutation soundness for the integer/bool ``requires_grad`` check
(`_check_requires_grad_dtype`, SymBugKind.REQUIRES_GRAD_NON_FLOAT): a tensor
constructor that sets ``requires_grad=true`` on an integer/bool dtype raises a
``RuntimeError`` ("Only Tensors of floating point and complex dtype can require
gradients").  The engine reports a bug only when ``requires_grad`` is positively
known ``true`` *and* the dtype is a *known* non-differentiable (integer/bool)
type; an unknown flag or dtype abstains.

Models the two relevant inputs as three-valued boolean flags (`Tri`): the
``requires_grad`` flag, and a "dtype is non-differentiable" flag.
-/
import TensorGuard.Symexec.Lattice

namespace TensorGuard
namespace Symexec
namespace RequiresGradNonFloat

/-- A three-valued boolean-flag abstraction: known ``true``/``false``, or ⊤. -/
inductive Tri
  | yes : Tri
  | no  : Tri
  | unk : Tri
  deriving DecidableEq, Repr

/-- Concretization membership for `Tri`. -/
def models (c : Bool) : Tri → Prop
  | .yes => c = true
  | .no  => c = false
  | .unk => True

/-- The runtime precondition: it is *not* the case that an integer/bool tensor
(`nondiff = true`) also requires grad (`rg = true`). -/
def DtypeRgOk (rg nondiff : Bool) : Prop := ¬ (rg = true ∧ nondiff = true)

/-- The engine's check: both flags are positively known ``true``. -/
def fires (rg nondiff : Tri) : Bool :=
  match rg, nondiff with
  | .yes, .yes => true
  | _,    _    => false

/-- **Conservativity.** An unknown ``requires_grad`` flag abstains. -/
theorem conservative_rg_unk (nondiff : Tri) : fires .unk nondiff = false := by
  cases nondiff <;> rfl

/-- **Conservativity.** An unknown dtype flag abstains. -/
theorem conservative_dtype_unk (rg : Tri) : fires rg .unk = false := by
  cases rg <;> rfl

/-- **Refutation soundness.** When the check fires, every concretization sets
both ``requires_grad`` and the non-differentiable-dtype flag to ``true``, so the
constructor precondition fails. -/
theorem refute {a b : Tri} (h : fires a b = true) :
    ∀ rg nd, models rg a → models nd b → ¬ DtypeRgOk rg nd := by
  cases a <;> cases b <;> simp [fires] at h
  intro rg nd hrg hnd
  simp only [models] at hrg hnd
  subst hrg; subst hnd
  simp [DtypeRgOk]

/-- **Certified counterexample.** -/
theorem witness {a b : Tri} (h : fires a b = true) :
    ∃ rg nd, models rg a ∧ models nd b ∧ ¬ DtypeRgOk rg nd := by
  cases a <;> cases b <;> simp [fires] at h
  exact ⟨true, true, by simp [models], by simp [models], by simp [DtypeRgOk]⟩

end RequiresGradNonFloat
end Symexec
end TensorGuard
