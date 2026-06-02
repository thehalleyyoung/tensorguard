/-
TensorGuard `nn.BatchNorm` shape-rule, machine-checked in Lean 4 (Step 159).

BatchNorm (1d/2d/3d) is **shape-preserving** but validates the channel/feature
dimension: it requires `input.dims[1] == num_features`.  This is exactly what
`src/model_checker.py::_propagate_batchnorm` computes (output = input, plus the
feature-count guard).

Modelling the input shape as `n :: feat :: rest`, we prove:

  * **shape preservation** (`bn_preserves`): the output shape equals the input;
  * **numel preserved** (`bn_numel`): a corollary (no elements created/dropped);
  * **feature guard** (`featValid_iff`, `feat_mismatch_flagged`): the channel dim
    matches `num_features` iff the layer is applicable — the soundness direction
    behind the "BatchNorm expects N features" refutation.

The companion test `tests/test_batchnorm_rule.py` replays the rule on **real**
`nn.BatchNorm1d/2d` modules: the output shape equals the input, and a wrong
channel count makes torch **raise** exactly when the Lean guard flags it.

Pure Lean 4 core (no mathlib).
-/

namespace TensorGuard
namespace BatchNormRule

def prod : List Nat → Nat
  | [] => 1
  | d :: rest => d * prod rest

/-- BatchNorm output shape: identical to the input. -/
def bnShape (n feat : Nat) (rest : List Nat) : List Nat := n :: feat :: rest

theorem bn_preserves (n feat : Nat) (rest : List Nat) :
    bnShape n feat rest = n :: feat :: rest := rfl

theorem bn_numel (n feat : Nat) (rest : List Nat) :
    prod (bnShape n feat rest) = prod (n :: feat :: rest) := rfl

/-- Feature guard: the channel dim (index 1) must equal `num_features`. -/
def featValid (feat numFeatures : Nat) : Bool := feat == numFeatures

theorem featValid_iff (feat numFeatures : Nat) :
    featValid feat numFeatures = true ↔ feat = numFeatures := by
  unfold featValid; exact beq_iff_eq

theorem feat_mismatch_flagged (feat numFeatures : Nat) (h : feat ≠ numFeatures) :
    featValid feat numFeatures = false := by
  unfold featValid; simp [h]

/-- The channel dim sits at index 1 and is exactly `feat`. -/
theorem bn_channel_index (n feat : Nat) (rest : List Nat) :
    (bnShape n feat rest).get? 1 = some feat := rfl

end BatchNormRule
end TensorGuard
