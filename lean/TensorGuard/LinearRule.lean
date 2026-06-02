/-
TensorGuard `nn.Linear` shape-rule, machine-checked in Lean 4 (Step 151).

`nn.Linear(in_features, out_features)` maps `(*, in_features) → (*, out_features)`:
it checks the **last** input dim equals `in_features` and replaces it with
`out_features`, leaving all leading (batch) dims untouched.  This is exactly what
`src/model_checker.py::_propagate_linear` computes.

Modelling the input shape as `pre ++ [in_features]`, we prove:

  * **rank preserved** (`lin_rank`): Linear does not change the rank;
  * **last dim** (`lin_last`): the trailing dim of the output is `out_features`;
  * **pre preserved** (`lin_prefix`): every leading/batch dim is unchanged;
  * **numel scaling** (`lin_numel`): `numel' = numel · out_features / in_features`
    stated multiplicatively as `prod pre · out_features`;
  * **input-feature guard** (`linValid_iff`, `mismatch_flagged`): the last dim
    matches `in_features` iff the layer is applicable — the soundness direction
    behind the "Linear expects last dim = in_features" refutation.

The companion test `tests/test_linear_rule.py` replays the rule on **real**
`nn.Linear` modules: the output shape equals `input.shape[:-1] + (out_features,)`,
and a wrong last dim makes torch **raise** exactly when the Lean guard flags it.

Pure Lean 4 core (no mathlib).
-/

namespace TensorGuard
namespace LinearRule

def prod : List Nat → Nat
  | [] => 1
  | d :: rest => d * prod rest

theorem prod_append (xs ys : List Nat) : prod (xs ++ ys) = prod xs * prod ys := by
  induction xs with
  | nil => simp [prod]
  | cons a t ih => simp [prod, ih, Nat.mul_assoc]

/-- Output of Linear given the (batch) pre and `out_features`. -/
def linShape (pre : List Nat) (outF : Nat) : List Nat := pre ++ [outF]

/-- Applicability guard: the input's last dim must equal `in_features`. -/
def linValid (lastDim inF : Nat) : Bool := lastDim == inF

theorem lin_rank (pre : List Nat) (inF outF : Nat) :
    (linShape pre outF).length = (pre ++ [inF]).length := by
  simp [linShape]

theorem lin_last (pre : List Nat) (outF : Nat) :
    (linShape pre outF).drop pre.length = [outF] := by
  simp [linShape, List.drop_left]

theorem lin_prefix (pre : List Nat) (outF : Nat) :
    (linShape pre outF).take pre.length = pre := by
  simp [linShape, List.take_left]

theorem lin_numel (pre : List Nat) (outF : Nat) :
    prod (linShape pre outF) = prod pre * outF := by
  simp [linShape, prod_append, prod]

/-- The guard fires (is valid) **iff** the last dim equals `in_features`. -/
theorem linValid_iff (lastDim inF : Nat) : linValid lastDim inF = true ↔ lastDim = inF := by
  unfold linValid; exact beq_iff_eq

/-- A last dim different from `in_features` is flagged (guard false) — the
    soundness direction behind the Linear feature-mismatch refutation. -/
theorem mismatch_flagged (lastDim inF : Nat) (h : lastDim ≠ inF) :
    linValid lastDim inF = false := by
  unfold linValid; simp [h]

end LinearRule
end TensorGuard
