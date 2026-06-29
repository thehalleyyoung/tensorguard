/-
TensorGuard.Symexec.Transfer.Linear

Refutation soundness for the `nn.Linear` in-feature check (`_apply_linear`,
SymBugKind.LAYER_DIM_MISMATCH): the last input dim must equal `in_features`.
An equality requirement, so it reduces to `Core.disagreeFires`.
-/
import TensorGuard.Symexec.Core

namespace TensorGuard
namespace Symexec
namespace Linear

open Dim Core

/-- The runtime precondition: the input's last dim equals the layer's
`in_features`. -/
def FeatureOk (inDim inFeatures : Nat) : Prop := inDim = inFeatures

/-- The engine's check. -/
def fires (a b : Dim) : Bool := disagreeFires a b

theorem conservative_left (b : Dim) : fires .unk b = false :=
  disagree_conservative_left b

theorem conservative_right (a : Dim) : fires a .unk = false :=
  disagree_conservative_right a

/-- **Refutation soundness.** -/
theorem refute {a b : Dim} (h : fires a b = true) :
    ∀ x y, dimModels x a → dimModels y b → ¬ FeatureOk x y := by
  intro x y hx hy
  exact disagree_sound h x y hx hy

/-- **Certified counterexample.** -/
theorem witness {a b : Dim} (h : fires a b = true) :
    ∃ x y, dimModels x a ∧ dimModels y b ∧ ¬ FeatureOk x y :=
  disagree_witness h

end Linear
end Symexec
end TensorGuard
