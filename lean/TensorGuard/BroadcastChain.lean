/-
TensorGuard broadcast **dim-chain** transfer, machine-checked in Lean 4
(Step 144).

NumPy/torch broadcasting aligns shapes dimension-by-dimension: two sizes are
compatible iff they are equal or one of them is `1`, and the broadcast size is
their `max` (the non-`1` one).  A multi-operand broadcast (`a + b + c`,
`torch.broadcast_shapes(...)`, a residual tower) folds this per-dimension rule
across a chain of operand sizes.  The verifier tracks the running broadcast size
for each dimension by folding `bcDim`.  This file models the **single-dimension
broadcast over a chain of operand sizes** and proves the laws the transfer relies
on:

  * compositionality over concatenation (`bcRun_append`);
  * `none` (an incompatibility) is absorbing (`bcRun_none`) — once the verifier
    has refuted a dimension it never silently recovers;
  * the rule is **commutative** (`bcDim_comm`) and `1` is a **two-sided
    identity** (`bcDim_one_left`, `bcDim_one_right`) — operand order and unit
    dims do not change the result, matching torch;
  * on compatible sizes the broadcast size is exactly their `max`
    (`bcDim_compat_max`);
  * **refutation soundness**: the rule flags (`= none`) *iff* the two sizes are
    genuinely incompatible — both `≠ 1` and unequal (`bcDim_none_iff`), so it
    never raises a false broadcast error.

The companion test `tests/test_broadcast_dim_chain.py` replays each chain on
**real torch** via `torch.broadcast_shapes`, asserting the live broadcaster
agrees with `bcRun` on both the compatible size (`= max`) and the incompatibility
(raises iff `none`).

Pure Lean 4 core (no mathlib).
-/

namespace TensorGuard
namespace BroadcastChain

/-- Per-dimension broadcast of two sizes; `none` marks an incompatibility. -/
def bcDim (a b : Nat) : Option Nat :=
  if a = 1 then some b
  else if b = 1 then some a
  else if a = b then some a
  else none

/-- One broadcast step against a running (possibly already-refuted) size. -/
def bcStep : Option Nat → Nat → Option Nat
  | none, _ => none
  | some a, b => bcDim a b

/-- Fold the broadcast over a chain of operand sizes. -/
def bcRun (acc : Option Nat) : List Nat → Option Nat
  | [] => acc
  | x :: rest => bcRun (bcStep acc x) rest

/- ===================================================================== -/
/- 1. Per-dimension laws                                                 -/
/- ===================================================================== -/

/-- Commutativity of the per-dimension rule. -/
theorem bcDim_comm (a b : Nat) : bcDim a b = bcDim b a := by
  unfold bcDim
  by_cases ha : a = 1
  · subst ha
    by_cases hb : b = 1
    · subst hb; rfl
    · simp [hb]
  · by_cases hb : b = 1
    · subst hb; simp [ha]
    · by_cases hab : a = b
      · subst hab; simp
      · have hba : ¬ b = a := fun h => hab h.symm
        simp [ha, hb, hab, hba]

/-- `1` is a right identity: broadcasting against a unit dim returns the dim. -/
theorem bcDim_one_right (a : Nat) : bcDim a 1 = some a := by
  unfold bcDim
  by_cases ha : a = 1 <;> simp_all

/-- `1` is a left identity. -/
theorem bcDim_one_left (a : Nat) : bcDim 1 a = some a := by
  unfold bcDim; simp

/-- Broadcasting a size against itself is a no-op. -/
theorem bcDim_self (a : Nat) : bcDim a a = some a := by
  unfold bcDim
  by_cases ha : a = 1 <;> simp_all

/-- On compatible **positive** sizes the broadcast size is exactly their `max`. -/
theorem bcDim_compat_max (a b : Nat) (ha1 : a ≥ 1) (hb1 : b ≥ 1)
    (h : a = 1 ∨ b = 1 ∨ a = b) :
    bcDim a b = some (max a b) := by
  rcases h with h | h | h
  · subst h; rw [bcDim_one_left]; congr 1; omega
  · subst h; rw [bcDim_one_right]; congr 1; omega
  · subst h; rw [bcDim_self]; congr 1; omega

/-- **Refutation soundness**: the rule flags iff the sizes are genuinely
    incompatible (both non-unit and unequal). -/
theorem bcDim_none_iff (a b : Nat) :
    bcDim a b = none ↔ (a ≠ 1 ∧ b ≠ 1 ∧ a ≠ b) := by
  unfold bcDim
  by_cases ha : a = 1 <;> by_cases hb : b = 1 <;> by_cases hab : a = b <;>
    simp_all

/- ===================================================================== -/
/- 2. Chain laws                                                         -/
/- ===================================================================== -/

/-- Compositionality: the run decomposes over concatenation. -/
theorem bcRun_append (acc : Option Nat) (xs ys : List Nat) :
    bcRun acc (xs ++ ys) = bcRun (bcRun acc xs) ys := by
  induction xs generalizing acc with
  | nil => rfl
  | cons x rest ih => simp [bcRun, ih]

/-- `none` (a refuted dimension) is absorbing along the whole chain. -/
theorem bcRun_none (xs : List Nat) : bcRun none xs = none := by
  induction xs with
  | nil => rfl
  | cons x rest ih => simp only [bcRun, bcStep]; exact ih

/-- Folding only unit dims preserves the running size. -/
theorem bcRun_ones (acc : Option Nat) (xs : List Nat)
    (h : ∀ x ∈ xs, x = 1) : bcRun acc xs = acc := by
  induction xs generalizing acc with
  | nil => rfl
  | cons x rest ih =>
    have hx : x = 1 := h x (List.mem_cons_self _ _)
    have hrest : ∀ y ∈ rest, y = 1 := fun y hy => h y (List.mem_cons_of_mem _ hy)
    subst hx
    cases acc with
    | none => simp only [bcRun, bcStep]; exact ih none hrest
    | some a => simp only [bcRun, bcStep, bcDim_one_right]; exact ih (some a) hrest

end BroadcastChain
end TensorGuard
