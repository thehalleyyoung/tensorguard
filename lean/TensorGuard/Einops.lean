/-
TensorGuard einops decomposition and axis-bijection rules, machine-checked in
Lean 4 (Step 229).

The Python checker `src/einops_verify.py::verify_einops` handles the core
einops `rearrange` obligations that have historically caused real model bugs:

  * a grouped left-hand axis such as `(h w)` may be decomposed only when all but
    one sub-axis are known and their product divides the consumed tensor axis;
  * the inferred sub-axis is exactly `axis / knownProduct`, so the group product
    reconstructs the original tensor axis and preserves the number of elements;
  * `rearrange` is axis-bijective: every named axis on the left must appear
    exactly once on the right, with no drops, additions, or duplicates.

The companion test `tests/test_einops_lean_conformance.py` generates concrete
decomposition and axis-bijection cases from these theorem shapes, runs the real
`einops` package, and checks TensorGuard's `verify_einops` verdict and output
shape agree with both the Lean-side predicate and real execution.

Pure Lean 4 core (no mathlib).
-/

namespace TensorGuard
namespace Einops

/- ===================================================================== -/
/- 1. Group decomposition / divisibility                                  -/
/- ===================================================================== -/

/-- Product (numel) of a list of concrete dimensions. -/
def prod : List Nat → Nat
  | [] => 1
  | d :: rest => d * prod rest

theorem prod_append (xs ys : List Nat) : prod (xs ++ ys) = prod xs * prod ys := by
  induction xs with
  | nil => simp [prod]
  | cons a t ih => simp [prod, ih, Nat.mul_assoc]

/-- The unknown sub-axis inferred for `(known unknown)` consuming `axis`. -/
def inferSubaxis (axis knownProduct : Nat) : Nat := axis / knownProduct

/-- A decomposition is accepted exactly when the known sub-axis product is
    positive and divides the consumed tensor axis. -/
def decompValid (axis knownProduct : Nat) : Bool :=
  decide (0 < knownProduct) && decide (knownProduct ∣ axis)

/-- The reconstructed two-factor decomposition shape.  Multi-factor groups are
    reduced to the same proof obligation by multiplying all known factors into
    `knownProduct`; the Python bridge generates the multi-factor cases. -/
def decomposedGroup (axis knownProduct : Nat) : List Nat :=
  [knownProduct, inferSubaxis axis knownProduct]

theorem decompValid_iff (axis knownProduct : Nat) :
    decompValid axis knownProduct = true ↔
      (0 < knownProduct ∧ knownProduct ∣ axis) := by
  unfold decompValid
  simp [Bool.and_eq_true, decide_eq_true_eq]

theorem decompValid_imp_dvd (axis knownProduct : Nat)
    (h : decompValid axis knownProduct = true) : knownProduct ∣ axis :=
  ((decompValid_iff axis knownProduct).mp h).2

/-- **Non-divisible decompositions are flagged**: this is the Lean counterpart
    of `error_kind == "non_divisible"` in `verify_einops`. -/
theorem nondivisible_decomposition_flagged (axis knownProduct : Nat)
    (h : ¬ knownProduct ∣ axis) : decompValid axis knownProduct = false := by
  unfold decompValid
  simp [decide_eq_false h]

/-- **Inference correctness**: the inferred sub-axis reconstructs the original
    consumed axis whenever the decomposition is valid. -/
theorem inferSubaxis_spec (axis knownProduct : Nat)
    (_hpos : 0 < knownProduct) (hdvd : knownProduct ∣ axis) :
    knownProduct * inferSubaxis axis knownProduct = axis := by
  unfold inferSubaxis
  exact Nat.mul_div_cancel' hdvd

/-- **Group product preservation**: splitting a grouped axis does not create or
    drop elements. -/
theorem decomposedGroup_product (axis knownProduct : Nat)
    (h : decompValid axis knownProduct = true) :
    prod (decomposedGroup axis knownProduct) = axis := by
  have hv := (decompValid_iff axis knownProduct).mp h
  unfold decomposedGroup
  simp [prod, inferSubaxis_spec axis knownProduct hv.1 hv.2]

/-- The inferred sub-axis occupies the trailing slot of the decomposed group. -/
theorem inferSubaxis_position (axis knownProduct : Nat) :
    (decomposedGroup axis knownProduct).getLast? =
      some (inferSubaxis axis knownProduct) := by
  unfold decomposedGroup
  rfl

/- ===================================================================== -/
/- 2. Named-axis bijection for `rearrange`                                -/
/- ===================================================================== -/

/-- Multiplicity of one named axis in a flattened side of an einops pattern.
    Axis names are modeled by natural identifiers; anonymous constants are
    absent from this flattened list, mirroring `_flatten_names` in Python. -/
def countAxis (a : Nat) : List Nat → Nat
  | [] => 0
  | x :: xs => (if x = a then 1 else 0) + countAxis a xs

/-- A rearrange side is bijective iff every named axis has the same
    multiplicity on both sides.  This catches drops, additions, and duplicates
    as count mismatches. -/
def axisBijection (lhs rhs : List Nat) : Prop :=
  ∀ a, countAxis a lhs = countAxis a rhs

theorem axisBijection_iff_counts (lhs rhs : List Nat) :
    axisBijection lhs rhs ↔ ∀ a, countAxis a lhs = countAxis a rhs := by
  rfl

theorem axisBijection_refl (xs : List Nat) : axisBijection xs xs := by
  intro a
  rfl

theorem axisBijection_sym {lhs rhs : List Nat}
    (h : axisBijection lhs rhs) : axisBijection rhs lhs := by
  intro a
  exact (h a).symm

theorem axisBijection_trans {xs ys zs : List Nat}
    (hxy : axisBijection xs ys) (hyz : axisBijection ys zs) :
    axisBijection xs zs := by
  intro a
  exact Eq.trans (hxy a) (hyz a)

/-- Adjacent-axis swaps are valid rearranges: `a b rest -> b a rest`. -/
theorem adjacent_swap_axis_bijection (a b : Nat) (rest : List Nat) :
    axisBijection (a :: b :: rest) (b :: a :: rest) := by
  intro x
  simp [countAxis, Nat.add_assoc, Nat.add_comm, Nat.add_left_comm]

/-- Dropping a distinct named axis violates the rearrange bijection condition. -/
theorem dropped_axis_not_bijection {a b : Nat} (h : a ≠ b) :
    ¬ axisBijection [a, b] [a] := by
  intro hb
  have hc := hb b
  simp [countAxis, h, h.symm] at hc

/-- Adding a distinct named axis violates the rearrange bijection condition. -/
theorem added_axis_not_bijection {a b : Nat} (h : a ≠ b) :
    ¬ axisBijection [a] [a, b] := by
  intro hb
  have hc := hb b
  simp [countAxis, h, h.symm] at hc

/-- Duplicating an existing axis on the right violates bijectivity. -/
theorem duplicated_axis_not_bijection (a : Nat) :
    ¬ axisBijection [a] [a, a] := by
  intro hb
  have hc := hb a
  simp [countAxis] at hc

end Einops
end TensorGuard
