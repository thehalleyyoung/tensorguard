/-
TensorGuard contiguity-bit transfer under a transpose/contiguous chain,
machine-checked in Lean 4 (Step 145).

Stride-sensitive operators (`view`, some fused kernels) require a *contiguous*
tensor.  Whether a tensor is contiguous flows through a chain of layout ops.  **Scope
(honest):** this models the concrete regime the verifier reasons about — a
*freshly contiguous, rank-2, non-degenerate* tensor under repeated `transpose`
of dims `(0,1)`, `.contiguous()`, and layout-preserving no-ops.  In that regime a
`transpose` toggles the contiguity bit (verified against torch: `t.t()` is
non-contiguous, `t.t().t()` is contiguous again), `.contiguous()` forces it
`true`, and a no-op preserves it.  (Outside this regime — degenerate/size-1 dims,
transposes of other dim pairs, tensors made non-contiguous by other means —
transpose is *not* a pure toggle; the verifier conservatively does not claim the
bit there.)  This file models the **contiguity transfer over a chain of such
layout ops** and proves the laws it relies on:

  * compositionality over concatenation (`ctgRun_append`);
  * `.contiguous()` **erases history** (`run_cons_contig`, `run_after_contig`):
    after it the bit is `true` regardless of the prior layout — the soundness
    direction that lets the verifier *clear* a stride alarm;
  * `transpose` is an **involution** on the chain (`run_transpose_involution`):
    two transposes of the same dims cancel, matching `t.t().t() = t`;
  * a layout-preserving (`keep`-only) chain leaves the bit unchanged
    (`run_allKeep`).

The companion test `tests/test_contiguity_transfer.py` replays each chain on a
**real 2-D tensor** via `.t()` / `.contiguous()` and asserts `is_contiguous()`
equals the Lean `ctgRun` prediction.

Pure Lean 4 core (no mathlib).
-/

namespace TensorGuard
namespace ContigFlow

/-- A single layout step. -/
inductive LayoutOp
  | keep        -- layout-preserving step: contiguity unchanged
  | transpose   -- swap dims (0,1): toggles contiguity
  | contiguous  -- `.contiguous()`: forces contiguity true
  deriving DecidableEq, Repr

/-- Per-op transfer on the contiguity bit. -/
def ctgStep (b : Bool) : LayoutOp → Bool
  | LayoutOp.keep       => b
  | LayoutOp.transpose  => !b
  | LayoutOp.contiguous => true

/-- Fold the transfer over a chain of layout ops. -/
def ctgRun (b0 : Bool) : List LayoutOp → Bool
  | [] => b0
  | op :: rest => ctgRun (ctgStep b0 op) rest

/- ===================================================================== -/
/- 1. Per-op laws                                                        -/
/- ===================================================================== -/

theorem keep_id (b : Bool) : ctgStep b LayoutOp.keep = b := rfl
theorem transpose_not (b : Bool) : ctgStep b LayoutOp.transpose = !b := rfl
theorem contiguous_true (b : Bool) : ctgStep b LayoutOp.contiguous = true := rfl

/- ===================================================================== -/
/- 2. Chain laws                                                         -/
/- ===================================================================== -/

/-- Compositionality: the run decomposes over concatenation. -/
theorem ctgRun_append (b0 : Bool) (xs ys : List LayoutOp) :
    ctgRun b0 (xs ++ ys) = ctgRun (ctgRun b0 xs) ys := by
  induction xs generalizing b0 with
  | nil => rfl
  | cons op rest ih => simp [ctgRun, ih]

/-- `.contiguous()` **erases history**: the result depends only on the suffix. -/
theorem run_cons_contig (b0 : Bool) (rest : List LayoutOp) :
    ctgRun b0 (LayoutOp.contiguous :: rest) = ctgRun true rest := rfl

/-- General form: anything before a `.contiguous()` is irrelevant. -/
theorem run_after_contig (b0 : Bool) (pre post : List LayoutOp) :
    ctgRun b0 (pre ++ LayoutOp.contiguous :: post) = ctgRun true post := by
  rw [ctgRun_append]
  rfl

/-- `transpose` is an **involution**: two consecutive transposes cancel
    (`t.t().t() = t`). -/
theorem run_transpose_involution (b0 : Bool) (rest : List LayoutOp) :
    ctgRun b0 (LayoutOp.transpose :: LayoutOp.transpose :: rest) = ctgRun b0 rest := by
  show ctgRun (!(!b0)) rest = ctgRun b0 rest
  rw [Bool.not_not]

/-- A layout-preserving (`keep`-only) chain leaves the contiguity bit unchanged. -/
theorem run_allKeep (b0 : Bool) (xs : List LayoutOp)
    (h : ∀ op ∈ xs, op = LayoutOp.keep) : ctgRun b0 xs = b0 := by
  induction xs generalizing b0 with
  | nil => rfl
  | cons op rest ih =>
    have hop : op = LayoutOp.keep := h op (List.mem_cons_self _ _)
    have hrest : ∀ o ∈ rest, o = LayoutOp.keep :=
      fun o ho => h o (List.mem_cons_of_mem _ ho)
    subst hop
    simp only [ctgRun, ctgStep]
    exact ih b0 hrest

end ContigFlow
end TensorGuard
