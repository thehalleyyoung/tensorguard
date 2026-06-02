/-
TensorGuard reduction rank-transfer semantics, machine-checked in Lean 4
(Step 142).

Shape reasoning for reductions (`sum`, `mean`, `amax`, ...) hinges on how the
*rank* of a tensor changes: reducing over one dimension with `keepdim=True`
preserves the rank (the reduced dim becomes size 1), while `keepdim=False`
lowers the rank by one.  The verifier folds this transfer along a chain of
reductions to track the running rank.  This file models the
**rank transfer over a chain of reductions** and proves the laws it relies on:

  * `reduceKeep` (`keepdim=True`) preserves the rank;
  * `reduceDrop` (`keepdim=False`) lowers the rank by one (truncated at 0);
  * the run is compositional and **monotone non-increasing** (a reduction never
    raises the rank — the soundness direction that prevents the verifier from
    inventing dimensions);
  * the outgoing rank equals the input rank minus the number of `keepdim=False`
    reductions (a closed form the verifier can use directly).

The companion test `tests/test_rank_transfer.py` replays each chain on a **real
tensor** via `torch.sum(dim=0, keepdim=...)` and asserts `out.dim()` equals
`rankRun`, so the proved transfer holds against the live torch shape machinery.

Pure Lean 4 core (no mathlib).
-/

namespace TensorGuard
namespace RankTransfer

/-- One reduction step. -/
inductive RedOp
  | reduceKeep  -- reduce one dim, `keepdim=True`:  rank preserved
  | reduceDrop  -- reduce one dim, `keepdim=False`: rank lowered by one
  deriving DecidableEq, Repr

/-- Per-op transfer on the rank (truncated `Nat` subtraction at 0). -/
def rankStep (r : Nat) : RedOp → Nat
  | RedOp.reduceKeep => r
  | RedOp.reduceDrop => r - 1

/-- Number of rank-lowering (`keepdim=False`) reductions in a chain. -/
def countDrop : List RedOp → Nat
  | [] => 0
  | RedOp.reduceDrop :: rest => 1 + countDrop rest
  | RedOp.reduceKeep :: rest => countDrop rest

/-- Fold the transfer over a chain of reductions. -/
def rankRun (r0 : Nat) : List RedOp → Nat
  | [] => r0
  | op :: rest => rankRun (rankStep r0 op) rest

/- ===================================================================== -/
/- 1. Per-op laws                                                        -/
/- ===================================================================== -/

theorem keep_id (r : Nat) : rankStep r RedOp.reduceKeep = r := rfl
theorem drop_pred (r : Nat) : rankStep r RedOp.reduceDrop = r - 1 := rfl

/-- A single step never raises the rank. -/
theorem step_le (r : Nat) (op : RedOp) : rankStep r op ≤ r := by
  cases op
  · exact Nat.le_refl r
  · exact Nat.sub_le r 1

/- ===================================================================== -/
/- 2. Chain laws                                                         -/
/- ===================================================================== -/

/-- The run decomposes over concatenation (compositionality of the transfer). -/
theorem rankRun_append (r0 : Nat) (xs ys : List RedOp) :
    rankRun r0 (xs ++ ys) = rankRun (rankRun r0 xs) ys := by
  induction xs generalizing r0 with
  | nil => rfl
  | cons op rest ih => simp [rankRun, ih]

/-- **Closed form**: the outgoing rank is the input rank minus the number of
    `keepdim=False` reductions (truncated at 0). -/
theorem rankRun_eq_sub_countDrop (r0 : Nat) (xs : List RedOp) :
    rankRun r0 xs = r0 - countDrop xs := by
  induction xs generalizing r0 with
  | nil => simp [rankRun, countDrop]
  | cons op rest ih =>
    cases op
    · -- reduceKeep
      simp only [rankRun, rankStep, countDrop]
      exact ih r0
    · -- reduceDrop
      simp only [rankRun, rankStep, countDrop]
      rw [ih (r0 - 1)]
      omega

/-- **Monotone non-increasing**: a chain of reductions never raises the rank. -/
theorem rankRun_le (r0 : Nat) (xs : List RedOp) : rankRun r0 xs ≤ r0 := by
  rw [rankRun_eq_sub_countDrop]
  exact Nat.sub_le r0 (countDrop xs)

/-- **`keepdim=True` everywhere preserves the rank.** -/
theorem rankRun_allKeep (r0 : Nat) (xs : List RedOp)
    (h : ∀ op ∈ xs, op = RedOp.reduceKeep) :
    rankRun r0 xs = r0 := by
  rw [rankRun_eq_sub_countDrop]
  have : countDrop xs = 0 := by
    induction xs with
    | nil => rfl
    | cons op rest ih =>
      have hop : op = RedOp.reduceKeep := h op (List.mem_cons_self _ _)
      have hrest : ∀ o ∈ rest, o = RedOp.reduceKeep :=
        fun o ho => h o (List.mem_cons_of_mem _ ho)
      subst hop
      simp [countDrop, ih hrest]
  simp [this]

/-- When the chain stays within rank (no underflow), the rank drops by exactly
    the number of `keepdim=False` reductions — stated as the exact identity that
    the no-underflow test cases exercise. -/
theorem rankRun_exact (r0 : Nat) (xs : List RedOp) (h : countDrop xs ≤ r0) :
    rankRun r0 xs + countDrop xs = r0 := by
  rw [rankRun_eq_sub_countDrop]
  omega

end RankTransfer
end TensorGuard
