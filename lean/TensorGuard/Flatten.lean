/-
TensorGuard flatten shape-rule, machine-checked in Lean 4 (Step 147).

`torch.flatten(x, start_dim, end_dim)` collapses the *inclusive* dim span
`[start_dim, end_dim]` into a single dimension whose size is the **product** of
the spanned sizes, leaving the pre (`dims < start_dim`) and suf
(`dims > end_dim`) untouched.  The verifier implements exactly this in
`src/model_checker.py::_propagate_flatten` (pre / span / suf split, the
span replaced by the product of its concrete sizes).

This file models the concrete-size rule as a list transform and proves the laws
the verifier relies on:

  * **numel preservation** (`prod_flatten`): the total element count is invariant
    — flatten neither creates nor drops elements (the soundness direction that
    lets the verifier reuse the incoming numel fact across a flatten);
  * **rank law** (`length_flatten`): the output rank is `|pre| + 1 + |suf|`
    (it drops `|span| - 1` dims when the span is non-empty);
  * **full flatten** (`flatten_full`): flattening the whole shape yields the
    single-dimension shape `[numel]` — the `nn.Flatten()`/`x.flatten()` case;
  * **trivial span** (`flatten_singleton`): collapsing a one-dim span is the
    identity on sizes.

The companion test `tests/test_flatten_rule.py` replays each split on a **real
tensor** via `torch.flatten` and against the verifier's `_propagate_flatten`,
asserting the product/rank predictions hold against the live engine.

Pure Lean 4 core (no mathlib).
-/

namespace TensorGuard
namespace Flatten

/-- Product (numel) of a shape given as a list of concrete dim sizes. -/
def prod : List Nat → Nat
  | [] => 1
  | d :: rest => d * prod rest

/-- Flatten the inclusive span (given already split as pre / span / suf)
    into a single dimension equal to the product of the spanned sizes. -/
def flattenShape (pre span suf : List Nat) : List Nat :=
  pre ++ (prod span :: suf)

/- ===================================================================== -/
/- 1. Product is a monoid homomorphism over concatenation                -/
/- ===================================================================== -/

/-- The numel of a concatenation is the product of the numels. -/
theorem prod_append (xs ys : List Nat) : prod (xs ++ ys) = prod xs * prod ys := by
  induction xs with
  | nil => simp [prod]
  | cons a t ih => simp [prod, ih, Nat.mul_assoc]

/- ===================================================================== -/
/- 2. Numel preservation                                                 -/
/- ===================================================================== -/

/-- **Numel preservation**: flatten preserves the total element count.  The
    product over `pre ++ [prod span] ++ suf` equals the product over the
    original `pre ++ span ++ suf`. -/
theorem prod_flatten (pre span suf : List Nat) :
    prod (flattenShape pre span suf) = prod (pre ++ span ++ suf) := by
  unfold flattenShape
  rw [prod_append, prod_append, prod_append]
  -- LHS: prod pre * (prod span * prod suf)   [from prod (prod span :: suf)]
  -- RHS: prod pre * (prod span * prod suf)
  simp [prod, Nat.mul_assoc]

/- ===================================================================== -/
/- 3. Rank law                                                           -/
/- ===================================================================== -/

/-- **Rank law**: the flattened shape has rank `|pre| + 1 + |suf|`. -/
theorem length_flatten (pre span suf : List Nat) :
    (flattenShape pre span suf).length = pre.length + 1 + suf.length := by
  unfold flattenShape
  simp only [List.length_append, List.length_cons]
  omega

/- ===================================================================== -/
/- 4. Boundary cases                                                     -/
/- ===================================================================== -/

/-- **Full flatten** (`x.flatten()` / `nn.Flatten(0)`): with empty pre and
    suf the result is the single-dimension shape `[numel]`. -/
theorem flatten_full (span : List Nat) :
    flattenShape [] span [] = [prod span] := by
  simp [flattenShape]

/-- **Trivial span**: collapsing a one-dimension span returns that dimension
    unchanged (flatten over a length-1 span is a size no-op). -/
theorem flatten_singleton (pre : List Nat) (d : Nat) (suf : List Nat) :
    flattenShape pre [d] suf = pre ++ (d :: suf) := by
  simp [flattenShape, prod]

/-- The flattened dim is exactly the product of the spanned sizes (it sits right
    after the pre). -/
theorem flatten_dim_value (pre span suf : List Nat) :
    (flattenShape pre span suf).drop pre.length
      = prod span :: suf := by
  unfold flattenShape
  rw [List.drop_left]

end Flatten
end TensorGuard
