/-
TensorGuard dtype-promotion **chain** transfer, machine-checked in Lean 4
(Step 143).

A multi-operand elementwise op (`torch.cat([...])`, a running `sum` of `add`s,
`torch.stack`, a fused residual) folds torch's type-promotion join over a *list*
of operand dtypes.  The verifier tracks the running result dtype by folding the
single-pair promotion `dtPromote` (proved a commutative/idempotent/associative
semilattice join in `DeviceDtype.lean`, Step 137) along the operand chain.  This
file lifts those pair laws to the **chain** and proves the properties the
multi-operand transfer relies on:

  * compositionality over concatenation (`promoteRun_append`);
  * the running dtype is an **upper bound** of the accumulator and of every
    operand (`promoteRun_ge_acc`, `promoteRun_ge_elem`) — the soundness
    direction: promotion never *narrows* below an operand, so the transfer can
    never under-approximate the result dtype;
  * **order independence** under adjacent transposition (`promoteRun_swap`) —
    operand order does not change the promoted dtype, matching torch;
  * `unknown` is absorbing along the chain (`promoteRun_unknown`).

The companion test `tests/test_dtype_promote_chain.py` mirrors `promoteRun` in
Python and replays each chain on **real torch** via `torch.promote_types`
(restricted to the dtype sub-alphabet on which the modeled lattice is exactly
torch's), so the proved chain transfer holds against the live promotion
machinery.

Pure Lean 4 core (no mathlib).  Reuses `Dt`/`dtPromote` from `DeviceDtype.lean`.
-/

import TensorGuard.DeviceDtype

namespace TensorGuard
namespace DtypePromoteChain

open TensorGuard.DevDtype

/-- Fold torch's promotion join over an operand chain, left to right. -/
def promoteRun (acc : Dt) : List Dt → Dt
  | [] => acc
  | x :: rest => promoteRun (dtPromote acc x) rest

/-- Partial order induced by the join: `a ≤ b` iff joining `a` into `b` is a
    no-op.  Decidable over the finite dtype set. -/
def dtLe (a b : Dt) : Bool := decide (dtPromote a b = b)

/- ===================================================================== -/
/- 0. Finite pair laws (all by exhaustive `decide` over the 8 dtypes)     -/
/- ===================================================================== -/

theorem dtLe_refl (a : Dt) : dtLe a a = true := by cases a <;> decide
theorem dtLe_trans (a b c : Dt) (hab : dtLe a b = true) (hbc : dtLe b c = true) :
    dtLe a c = true := by revert hab hbc; cases a <;> cases b <;> cases c <;> decide
theorem dtLe_antisymm (a b : Dt) (hab : dtLe a b = true) (hba : dtLe b a = true) :
    a = b := by revert hab hba; cases a <;> cases b <;> decide

/-- The join dominates its left operand. -/
theorem le_promote_left (a b : Dt) : dtLe a (dtPromote a b) = true := by
  cases a <;> cases b <;> decide
/-- The join dominates its right operand. -/
theorem le_promote_right (a b : Dt) : dtLe b (dtPromote a b) = true := by
  cases a <;> cases b <;> decide

/-- The two-step fold is order-independent (commutativity+associativity of the
    join, specialised to the accumulator update). -/
theorem promote_step_swap (acc x y : Dt) :
    dtPromote (dtPromote acc x) y = dtPromote (dtPromote acc y) x := by
  cases acc <;> cases x <;> cases y <;> decide

/-- `unknown` is absorbing for the single step. -/
theorem promote_unknown_step (x : Dt) : dtPromote Dt.unknown x = Dt.unknown := by
  cases x <;> decide

/- ===================================================================== -/
/- 1. Chain laws                                                          -/
/- ===================================================================== -/

/-- Compositionality: the run decomposes over concatenation. -/
theorem promoteRun_append (acc : Dt) (xs ys : List Dt) :
    promoteRun acc (xs ++ ys) = promoteRun (promoteRun acc xs) ys := by
  induction xs generalizing acc with
  | nil => rfl
  | cons x rest ih => simp [promoteRun, ih]

/-- The running dtype dominates the accumulator (monotone non-decreasing). -/
theorem promoteRun_ge_acc (acc : Dt) (xs : List Dt) :
    dtLe acc (promoteRun acc xs) = true := by
  induction xs generalizing acc with
  | nil => exact dtLe_refl acc
  | cons x rest ih =>
    have h1 : dtLe acc (dtPromote acc x) = true := le_promote_left acc x
    have h2 : dtLe (dtPromote acc x) (promoteRun (dtPromote acc x) rest) = true := ih _
    exact dtLe_trans _ _ _ h1 h2

/-- The running dtype dominates **every** operand in the chain — promotion never
    narrows below an operand. -/
theorem promoteRun_ge_elem (acc : Dt) (xs : List Dt) (x : Dt) (hx : x ∈ xs) :
    dtLe x (promoteRun acc xs) = true := by
  induction xs generalizing acc with
  | nil => cases hx
  | cons y rest ih =>
    rcases List.mem_cons.mp hx with h | h
    · subst h
      have h1 : dtLe x (dtPromote acc x) = true := le_promote_right acc x
      have h2 : dtLe (dtPromote acc x) (promoteRun (dtPromote acc x) rest) = true :=
        promoteRun_ge_acc _ _
      exact dtLe_trans _ _ _ h1 h2
    · exact ih _ h

/-- **Order independence**: swapping the first two operands leaves the promoted
    dtype unchanged (an adjacent transposition; iterating gives full permutation
    invariance). -/
theorem promoteRun_swap (acc x y : Dt) (rest : List Dt) :
    promoteRun acc (x :: y :: rest) = promoteRun acc (y :: x :: rest) := by
  show promoteRun (dtPromote (dtPromote acc x) y) rest
     = promoteRun (dtPromote (dtPromote acc y) x) rest
  rw [promote_step_swap]

/-- `unknown` is absorbing along the whole chain: once the running dtype is
    `unknown`, it stays `unknown`. -/
theorem promoteRun_unknown (xs : List Dt) :
    promoteRun Dt.unknown xs = Dt.unknown := by
  induction xs with
  | nil => rfl
  | cons x rest ih =>
    simp only [promoteRun, promote_unknown_step]
    exact ih

end DtypePromoteChain
end TensorGuard
