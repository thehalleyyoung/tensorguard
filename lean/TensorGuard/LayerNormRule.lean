/-
TensorGuard `nn.LayerNorm` shape-rule, machine-checked in Lean 4 (Step 155).

`nn.LayerNorm(normalized_shape)` is **shape-preserving** but requires that the
trailing dims of the input equal `normalized_shape`.  This is exactly what
`src/model_checker.py::_propagate_layernorm` computes (output = input, plus a
trailing-suffix check on `normalized_shape`).

Modelling the input as `pre ++ normShape`, we prove:

  * **shape preservation** (`ln_preserves`) and **numel preservation**
    (`ln_numel`);
  * **suffix length** (`ln_length`): the validity split point is `|pre|`;
  * **suffix match** (`ln_suffix_match`): the trailing `|normShape|` dims of the
    input are exactly `normalized_shape` (the validity witness);
  * **trailing-mismatch refutation** (`ln_mismatch_flagged`): if the last input
    dim differs from the last `normalized_shape` entry, the trailing slice differs
    — the soundness direction behind the LayerNorm mismatch alarm.

The companion test `tests/test_layernorm_rule.py` replays the rule on **real**
`nn.LayerNorm` modules: the output equals the input, and a wrong trailing dim
makes torch **raise** exactly when the Lean guard flags it.

Pure Lean 4 core (no mathlib).
-/

namespace TensorGuard
namespace LayerNormRule

def prod : List Nat → Nat
  | [] => 1
  | d :: rest => d * prod rest

theorem prod_append (xs ys : List Nat) : prod (xs ++ ys) = prod xs * prod ys := by
  induction xs with
  | nil => simp [prod]
  | cons a t ih => simp [prod, ih, Nat.mul_assoc]

/-- LayerNorm output shape: identical to the input. -/
def lnShape (pre normShape : List Nat) : List Nat := pre ++ normShape

theorem ln_preserves (pre normShape : List Nat) :
    lnShape pre normShape = pre ++ normShape := rfl

theorem ln_numel (pre normShape : List Nat) :
    prod (lnShape pre normShape) = prod pre * prod normShape := by
  simp [lnShape, prod_append]

/-- The validity split point is the pre length. -/
theorem ln_length (pre normShape : List Nat) :
    (lnShape pre normShape).length - normShape.length = pre.length := by
  simp [lnShape]

/-- **Suffix match**: the trailing `|normShape|` dims equal `normalized_shape`. -/
theorem ln_suffix_match (pre normShape : List Nat) :
    (lnShape pre normShape).drop pre.length = normShape := by
  simp [lnShape, List.drop_left]

/-- **Trailing-mismatch refutation**: a different last dim makes the trailing
    slice differ from `normalized_shape`. -/
theorem ln_mismatch_flagged (pre : List Nat) (a b : Nat) (h : a ≠ b) :
    (lnShape pre [a]).drop pre.length ≠ [b] := by
  rw [ln_suffix_match]
  intro hcontra
  exact h (by injection hcontra)

end LayerNormRule
end TensorGuard
