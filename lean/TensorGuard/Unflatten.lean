/-
TensorGuard `nn.Unflatten` shape-rule, machine-checked in Lean 4 (Step 158).

`nn.Unflatten(dim, unflattened_size)` replaces dim `dim` with the dims
`unflattened_size`, requiring `∏ unflattened_size == size(dim)`.  It is the
**inverse** of `flatten` (Step 147).  This is exactly what
`src/model_checker.py::_propagate_unflatten` computes.

Modelling the input as `pre ++ [orig] ++ suf`, output `pre ++ sizes ++ suf`:

  * **numel preservation** (`unflatten_numel`): when `∏ sizes = orig`, the output
    numel equals the input numel — no elements created/dropped;
  * **rank law** (`unflatten_rank`): the output rank is
    `|pre| + |sizes| + |suf|` (it adds `|sizes| − 1` dims);
  * **flatten/unflatten inverse** (`unflatten_then_flatten`): collapsing the
    inserted `sizes` span back to a single dim recovers the original
    `pre ++ [orig] ++ suf` (round-trip with Step 147);
  * **size guard** (`unflattenValid_iff`, `size_mismatch_flagged`): the split is
    admitted iff `∏ sizes = orig`.

The companion test `tests/test_unflatten_rule.py` replays the rule on **real**
`nn.Unflatten` modules and against the verifier's propagator, and confirms a
non-matching product makes torch **raise** exactly when the Lean guard flags it.

Pure Lean 4 core (no mathlib).
-/

namespace TensorGuard
namespace Unflatten

def prod : List Nat → Nat
  | [] => 1
  | d :: rest => d * prod rest

theorem prod_append (xs ys : List Nat) : prod (xs ++ ys) = prod xs * prod ys := by
  induction xs with
  | nil => simp [prod]
  | cons a t ih => simp [prod, ih, Nat.mul_assoc]

/-- Output shape: the dim `orig` (sitting between `pre` and `suf`) replaced by
    the list `sizes`. -/
def unflattenShape (pre sizes suf : List Nat) : List Nat := pre ++ sizes ++ suf

/-- **Numel preservation** under the validity condition `∏ sizes = orig`. -/
theorem unflatten_numel (pre sizes suf : List Nat) (orig : Nat)
    (h : prod sizes = orig) :
    prod (unflattenShape pre sizes suf) = prod (pre ++ [orig] ++ suf) := by
  unfold unflattenShape
  rw [prod_append, prod_append, prod_append, prod_append, h]
  simp [prod]

/-- **Rank law**. -/
theorem unflatten_rank (pre sizes suf : List Nat) :
    (unflattenShape pre sizes suf).length = pre.length + sizes.length + suf.length := by
  unfold unflattenShape
  simp [List.length_append]
  omega

/-- **Inverse of flatten**: collapsing the inserted `sizes` span to its product
    recovers the original `pre ++ [orig] ++ suf`. -/
theorem unflatten_then_flatten (pre sizes suf : List Nat) :
    pre ++ [prod sizes] ++ suf =
      pre ++ [prod (unflattenShape pre sizes suf |>.drop pre.length |>.take sizes.length)] ++ suf := by
  unfold unflattenShape
  simp [List.drop_left, List.take_left]

/-- Size guard: the split is admitted iff `∏ sizes = orig`. -/
def unflattenValid (sizes : List Nat) (orig : Nat) : Bool := prod sizes == orig

theorem unflattenValid_iff (sizes : List Nat) (orig : Nat) :
    unflattenValid sizes orig = true ↔ prod sizes = orig := by
  unfold unflattenValid; exact beq_iff_eq

theorem size_mismatch_flagged (sizes : List Nat) (orig : Nat) (h : prod sizes ≠ orig) :
    unflattenValid sizes orig = false := by
  unfold unflattenValid; simp [h]

end Unflatten
end TensorGuard
