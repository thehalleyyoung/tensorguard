/-
TensorGuard concatenation (`torch.cat`) shape-rule, machine-checked in Lean 4
(Step 148).

`torch.cat([a, b], dim=axis)` requires **every dimension except `axis` to match**
and produces a tensor whose `axis` dimension is the **sum** of the operands'
`axis` sizes (all other dims unchanged).  We model an operand's shape, relative
to the concatenation axis, as a triple `(pre, axisSize, post)` where `pre` are
the dims before the axis and `post` the dims after — so two operands are
cat-compatible iff they share the same `pre` and `post` (the "all other dims
equal" rule).

**Scope (honest):** this models the standard same-rank case (the regime the
verifier reasons about); PyTorch's legacy 1-D empty-tensor exception
(`torch.cat([x, torch.zeros(0)])`) is out of scope — flagging it would be a
precision loss (false positive), never unsound.

Proved laws:

  * **compatibility characterization** (`catValid_iff`): the rule admits the cat
    iff the non-axis dims coincide — the refutation soundness direction (a
    genuine mismatch is always flagged);
  * **axis additivity** (`catAxis_value`): the output axis size is exactly
    `a + b`;
  * **numel additivity** (`prod_cat`): under compatibility the result's element
    count is the **sum** of the operands' counts — the fact the verifier
    propagates across a cat;
  * **commutativity / associativity of sizes** (`cat_axis_comm`,
    `cat_assoc`): concatenation order does not change the resulting shape's axis
    arithmetic;
  * **empty identity** (`cat_zero_right`): concatenating a zero-length tensor
    along the axis is a size no-op.

The companion test `tests/test_cat_rule.py` replays each case on **real tensors**
via `torch.cat`, asserting the output shape, numel additivity and the
compatibility rule (torch raises iff a non-axis dim differs) match the Lean
predictions.

Pure Lean 4 core (no mathlib).
-/

namespace TensorGuard
namespace CatRule

/-- Product (numel) of a list of concrete dim sizes. -/
def prod : List Nat → Nat
  | [] => 1
  | d :: rest => d * prod rest

theorem prod_append (xs ys : List Nat) : prod (xs ++ ys) = prod xs * prod ys := by
  induction xs with
  | nil => simp [prod]
  | cons a t ih => simp [prod, ih, Nat.mul_assoc]

/-- A shape described relative to the concat axis: dims before the axis, the
    axis size, dims after the axis. -/
structure CatShape where
  pre : List Nat
  axisSize : Nat
  post : List Nat
deriving DecidableEq, Repr

/-- Materialise a `CatShape` back to a flat dim list. -/
def toList (s : CatShape) : List Nat := s.pre ++ (s.axisSize :: s.post)

/-- Two operands are cat-compatible iff every non-axis dim matches. -/
def catValid (a b : CatShape) : Bool :=
  decide (a.pre = b.pre) && decide (a.post = b.post)

/-- The concatenated shape: same non-axis dims, axis size summed. -/
def catShape (a b : CatShape) : CatShape :=
  { pre := a.pre, axisSize := a.axisSize + b.axisSize, post := a.post }

/- ===================================================================== -/
/- 1. Compatibility characterization                                     -/
/- ===================================================================== -/

/-- **Refutation soundness**: the cat is admitted iff the non-axis dims coincide. -/
theorem catValid_iff (a b : CatShape) :
    catValid a b = true ↔ (a.pre = b.pre ∧ a.post = b.post) := by
  unfold catValid
  simp [Bool.and_eq_true, decide_eq_true_eq]

/-- A mismatch on a non-axis (prefix) dim is always flagged. -/
theorem catValid_pre_mismatch (a b : CatShape) (h : a.pre ≠ b.pre) :
    catValid a b = false := by
  unfold catValid
  simp [decide_eq_false h]

/- ===================================================================== -/
/- 2. Axis additivity                                                    -/
/- ===================================================================== -/

/-- The output axis size is exactly the sum of the operands' axis sizes. -/
theorem catAxis_value (a b : CatShape) :
    (catShape a b).axisSize = a.axisSize + b.axisSize := rfl

/- ===================================================================== -/
/- 3. Numel additivity                                                   -/
/- ===================================================================== -/

/-- **Numel additivity**: when compatible, the concatenated numel is the sum of
    the operands' numels.  (Compatibility makes `b`'s non-axis dims equal `a`'s,
    so the shared `prod pre * prod post` factors out of the sum.) -/
theorem prod_cat (a b : CatShape) (h : catValid a b = true) :
    prod (toList (catShape a b)) = prod (toList a) + prod (toList b) := by
  rw [catValid_iff] at h
  obtain ⟨hpre, hpost⟩ := h
  unfold toList catShape
  simp only []
  rw [prod_append, prod_append, prod_append]
  simp only [prod]
  -- prod a.pre * ((a.axisSize + b.axisSize) * prod a.post)
  --   = prod a.pre * (a.axisSize * prod a.post)
  --   + prod b.pre * (b.axisSize * prod b.post)
  rw [← hpre, ← hpost]
  rw [Nat.add_mul, Nat.mul_add]

/- ===================================================================== -/
/- 4. Order laws on the axis arithmetic                                  -/
/- ===================================================================== -/

/-- Concatenation is size-commutative: swapping operands sums the axis the same
    way (and, when compatible, the non-axis dims are equal anyway). -/
theorem cat_axis_comm (a b : CatShape) :
    (catShape a b).axisSize = (catShape b a).axisSize := by
  unfold catShape; simp [Nat.add_comm]

/-- Associativity of the axis sum for a 3-way concatenation. -/
theorem cat_assoc (a b c : CatShape) :
    (catShape (catShape a b) c).axisSize = (catShape a (catShape b c)).axisSize := by
  unfold catShape; simp [Nat.add_assoc]

/-- **Empty identity**: concatenating a zero-length tensor along the axis leaves
    the axis size unchanged. -/
theorem cat_zero_right (a : CatShape) :
    (catShape a { pre := a.pre, axisSize := 0, post := a.post }).axisSize
      = a.axisSize := by
  unfold catShape; simp

end CatRule
end TensorGuard
