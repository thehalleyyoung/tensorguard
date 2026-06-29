/-
TensorGuard.Symexec.Transfer.NumpyOnGrad

Refutation soundness for the ``.numpy()``-on-grad check
(`_check_numpy`, SymBugKind.NUMPY_ON_GRAD): ``tensor.numpy()`` raises a
``RuntimeError`` ("Can't call numpy() on Tensor that requires grad") when the
tensor requires grad.  The engine reports a bug only when ``requires_grad`` is
*positively known* to be ``true`` (an unknown or ``false`` flag abstains;
``.detach()`` clears the flag and is never flagged).

Carries a tiny three-valued boolean-flag abstraction (`Tri`).
-/
import TensorGuard.Symexec.Lattice

namespace TensorGuard
namespace Symexec
namespace NumpyOnGrad

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

/-- The runtime precondition: ``.numpy()`` needs a tensor that does *not*
require grad. -/
def NumpyOk (rg : Bool) : Prop := rg = false

/-- The engine's check: ``requires_grad`` is positively known ``true``. -/
def fires (v : Tri) : Bool :=
  match v with
  | .yes => true
  | _    => false

/-- **Conservativity.** Unknown flags abstain. -/
theorem conservative_unk : fires .unk = false := rfl

/-- **Conservativity.** A known-``false`` flag abstains. -/
theorem conservative_no : fires .no = false := rfl

/-- **Refutation soundness.** When the check fires, every concretization of the
flag is ``true``, so the ``.numpy()`` precondition fails. -/
theorem refute {v : Tri} (h : fires v = true) :
    ∀ rg, models rg v → ¬ NumpyOk rg := by
  cases v with
  | no  => simp [fires] at h
  | unk => simp [fires] at h
  | yes =>
    intro rg hrg
    simp only [models] at hrg
    subst hrg
    simp [NumpyOk]

/-- **Certified counterexample.** -/
theorem witness {v : Tri} (h : fires v = true) :
    ∃ rg, models rg v ∧ ¬ NumpyOk rg := by
  cases v with
  | no  => simp [fires] at h
  | unk => simp [fires] at h
  | yes => exact ⟨true, by simp [models], by simp [NumpyOk]⟩

end NumpyOnGrad
end Symexec
end TensorGuard
