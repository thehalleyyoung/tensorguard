/-
TensorGuard reshape/`view` `-1`-inference rule, machine-checked in Lean 4
(Step 150).

`x.reshape(known…, -1)` (or `view`) infers the single `-1` dimension as
`numel(x) / ∏ known`.  PyTorch requires the inference to be **exact**: it is a
runtime error unless `∏ known` divides `numel(x)` (and is positive).  The
verifier implements this in `src/tensor_shapes.py::compute_reshape_shape`, which
counts the `-1`, multiplies the specified dims and checks
`numel % specified == 0` before filling the inferred slot.

Proved laws:

  * **inference correctness** (`inferDim_spec`): when `∏ known` divides the
    numel, the inferred dim is exactly `numel / ∏ known`, and reconstituting the
    shape gives back the **same numel** (`prod_reshape_valid`) — reshape moves
    elements around but never creates or drops them;
  * **validity characterization** (`reshapeValid_iff`): the reshape is admitted
    iff `∏ known` is positive and divides the numel — the refutation soundness
    direction (a non-dividing product is always flagged);
  * **non-divisible is flagged** (`nondivisible_flagged`): if `∏ known` does not
    divide the numel the guard returns `false` (PyTorch raises);
  * **divisibility is the exact criterion** (`reshapeValid_imp_dvd`): a valid
    reshape entails the divisibility fact the numel-preservation proof needs.

The companion test `tests/test_reshape_infer.py` replays the rule on **real
tensors** via `x.reshape(..., -1)` and against the verifier's
`compute_reshape_shape`: the inferred dim and total numel match the Lean
prediction, and a non-dividing specification makes torch **raise** — exactly when
the Lean guard flags it.

Pure Lean 4 core (no mathlib).
-/

namespace TensorGuard
namespace ReshapeInfer

/-- Product (numel) of a list of concrete dim sizes. -/
def prod : List Nat → Nat
  | [] => 1
  | d :: rest => d * prod rest

theorem prod_append (xs ys : List Nat) : prod (xs ++ ys) = prod xs * prod ys := by
  induction xs with
  | nil => simp [prod]
  | cons a t ih => simp [prod, ih, Nat.mul_assoc]

/-- The dimension inferred for the `-1` slot. -/
def inferDim (total : Nat) (known : List Nat) : Nat := total / prod known

/-- The reshape is admitted iff `∏ known` is positive and divides the numel. -/
def reshapeValid (total : Nat) (known : List Nat) : Bool :=
  decide (0 < prod known) && decide (prod known ∣ total)

/-- The reshaped shape with the inferred dim placed in the `-1` slot (here the
    trailing position, as `view(known…, -1)`). -/
def reshapeShape (total : Nat) (known : List Nat) : List Nat :=
  known ++ [inferDim total known]

/- ===================================================================== -/
/- 1. Validity characterization                                          -/
/- ===================================================================== -/

theorem reshapeValid_iff (total : Nat) (known : List Nat) :
    reshapeValid total known = true ↔ (0 < prod known ∧ prod known ∣ total) := by
  unfold reshapeValid
  simp [Bool.and_eq_true, decide_eq_true_eq]

/-- A valid reshape entails the divisibility fact. -/
theorem reshapeValid_imp_dvd (total : Nat) (known : List Nat)
    (h : reshapeValid total known = true) : prod known ∣ total :=
  ((reshapeValid_iff total known).mp h).2

/-- **Non-divisible is flagged**: if `∏ known` does not divide the numel the
    guard returns `false` (PyTorch raises here). -/
theorem nondivisible_flagged (total : Nat) (known : List Nat)
    (h : ¬ prod known ∣ total) : reshapeValid total known = false := by
  unfold reshapeValid
  simp [decide_eq_false h]

/- ===================================================================== -/
/- 2. Inference correctness & numel preservation                         -/
/- ===================================================================== -/

/-- **Inference correctness**: when `∏ known` is positive and divides the numel,
    multiplying the known dims by the inferred dim recovers the numel exactly.
    (The positivity hypothesis rules out the ambiguous `∏ known = 0` case that
    PyTorch also rejects.) -/
theorem inferDim_spec (total : Nat) (known : List Nat)
    (_hpos : 0 < prod known) (hdvd : prod known ∣ total) :
    prod known * inferDim total known = total := by
  unfold inferDim
  exact Nat.mul_div_cancel' hdvd

/-- **Numel preservation**: the reshaped shape has exactly the original numel
    (under validity).  Reshape rearranges elements but conserves their count. -/
theorem prod_reshape_valid (total : Nat) (known : List Nat)
    (h : reshapeValid total known = true) :
    prod (reshapeShape total known) = total := by
  have hdvd : prod known ∣ total := reshapeValid_imp_dvd total known h
  have hpos : 0 < prod known := ((reshapeValid_iff total known).mp h).1
  unfold reshapeShape
  rw [prod_append]
  simp only [prod]
  -- prod known * (inferDim total known * 1) = prod known * inferDim … = total
  rw [Nat.mul_one]
  exact inferDim_spec total known hpos hdvd

/-- The inferred dim sits in the trailing slot and equals `numel / ∏ known`. -/
theorem reshape_infer_position (total : Nat) (known : List Nat) :
    (reshapeShape total known).getLast? = some (inferDim total known) := by
  unfold reshapeShape
  rw [List.getLast?_concat]

end ReshapeInfer
end TensorGuard
