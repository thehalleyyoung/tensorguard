/-
TensorGuard gradient-flow transfer semantics, machine-checked in Lean 4
(Step 138).

`DeviceDtype.lean` models the single-op `detach` gradient check.  This file lifts
it to the **gradient-flow transfer function over a chain of ops**, exactly as the
verifier propagates the `requires_grad` bit through a `forward`:

  * a differentiable op (`keep`, e.g. `x * 2`, `x + b`) propagates the incoming
    bit unchanged (under an enabled grad context);
  * `detach` (`.detach()`) and an op executed under `torch.no_grad()` (`noGrad`)
    force the outgoing bit to `false`;
  * `reattach` (`.requires_grad_(True)` on a leaf) forces it to `true`.

`gradRun b0 ops` folds the per-op transfer over a chain.  We prove the algebraic
laws the verifier relies on (resetting ops are absorbing, `keep` is the identity,
the run decomposes over concatenation, and — for the reattach-free fragment the
cross-check exercises — the outgoing bit is `true` iff the input required grad
and no resetting op intervened).

The companion test `tests/test_grad_flow_transfer.py` replays each chain on a
**real autograd tensor** and asserts `out.requires_grad` equals `gradRun`, so the
proved transfer holds against the live torch autograd engine.

Pure Lean 4 core (no mathlib).
-/

namespace TensorGuard
namespace GradFlow

/-- One step of the gradient-flow transfer. -/
inductive GradOp
  | keep      -- differentiable op: propagate the incoming bit
  | detach    -- `.detach()`: output never requires grad
  | noGrad    -- op under `torch.no_grad()`: output never requires grad
  | reattach  -- `.requires_grad_(True)` on a leaf: output requires grad
  deriving DecidableEq, Repr

/-- Per-op transfer on the `requires_grad` bit. -/
def gradStep (b : Bool) : GradOp → Bool
  | GradOp.keep     => b
  | GradOp.detach   => false
  | GradOp.noGrad   => false
  | GradOp.reattach => true

/-- Whether an op *resets* the bit irrespective of its input. -/
def isReset : GradOp → Bool
  | GradOp.detach => true
  | GradOp.noGrad => true
  | _ => false

/-- Fold the transfer over a chain of ops. -/
def gradRun (b0 : Bool) : List GradOp → Bool
  | [] => b0
  | op :: rest => gradRun (gradStep b0 op) rest

/- ===================================================================== -/
/- 1. Per-op laws                                                        -/
/- ===================================================================== -/

theorem keep_id (b : Bool) : gradStep b GradOp.keep = b := rfl
theorem detach_false (b : Bool) : gradStep b GradOp.detach = false := rfl
theorem noGrad_false (b : Bool) : gradStep b GradOp.noGrad = false := rfl
theorem reattach_true (b : Bool) : gradStep b GradOp.reattach = true := rfl

/-- A resetting op is **absorbing**: its output bit does not depend on the input
    (this is what makes `detach` / `no_grad` sound regardless of upstream). -/
theorem reset_absorbing (op : GradOp) (h : isReset op = true) (b b' : Bool) :
    gradStep b op = gradStep b' op := by
  cases op <;> simp_all [isReset, gradStep]

/- ===================================================================== -/
/- 2. Chain laws                                                         -/
/- ===================================================================== -/

/-- The run decomposes over concatenation (compositionality of the transfer). -/
theorem gradRun_append (b0 : Bool) (xs ys : List GradOp) :
    gradRun b0 (xs ++ ys) = gradRun (gradRun b0 xs) ys := by
  induction xs generalizing b0 with
  | nil => rfl
  | cons op rest ih => simp [gradRun, ih]

/-- Appending a resetting op makes the whole prefix irrelevant: the result is the
    op's reset value, independent of `b0`. -/
theorem run_after_reset (b0 b0' : Bool) (xs : List GradOp)
    (op : GradOp) (h : isReset op = true) :
    gradRun b0 (xs ++ [op]) = gradRun b0' (xs ++ [op]) := by
  rw [gradRun_append, gradRun_append]
  have : gradStep (gradRun b0 xs) op = gradStep (gradRun b0' xs) op :=
    reset_absorbing op h _ _
  simp [gradRun, this]

/-- **Characterization for the reattach-free fragment** (the one the torch
    cross-check exercises): the outgoing bit is `true` iff the input required
    grad and no resetting op intervened. -/
theorem run_noReattach_true_iff (b0 : Bool) (xs : List GradOp)
    (h : ∀ op ∈ xs, op ≠ GradOp.reattach) :
    gradRun b0 xs = (b0 && xs.all (fun op => !isReset op)) := by
  induction xs generalizing b0 with
  | nil => simp [gradRun]
  | cons op rest ih =>
    have hrest : ∀ o ∈ rest, o ≠ GradOp.reattach :=
      fun o ho => h o (List.mem_cons_of_mem _ ho)
    have hop : op ≠ GradOp.reattach := h op (List.mem_cons_self _ _)
    cases op
    · -- keep
      simp only [gradRun, gradStep, isReset, List.all_cons, Bool.not_false,
                 Bool.true_and]
      exact ih b0 hrest
    · -- detach
      simp only [gradRun, gradStep, isReset, List.all_cons, Bool.not_true,
                 Bool.false_and, Bool.and_false]
      simpa using ih false hrest
    · -- noGrad
      simp only [gradRun, gradStep, isReset, List.all_cons, Bool.not_true,
                 Bool.false_and, Bool.and_false]
      simpa using ih false hrest
    · -- reattach contradicts hop
      exact absurd rfl hop

end GradFlow
end TensorGuard
