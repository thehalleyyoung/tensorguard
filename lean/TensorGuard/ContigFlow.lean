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

Step 235 extends the same file with concrete rank-4 layout algebra: row-major
strides, canonical non-degenerate NCHW channels-last strides, storage-offset
preservation for metadata-only reorders, channel slicing offsets, and the
adjacent-stride viewability condition used when collapsing the CHW tail.  The
companion test `tests/test_contiguity_transfer.py` replays the bit-level chain on
a **real 2-D tensor** and the rank-4 algebra on **real PyTorch NCHW tensors**,
including the channels-last caveat that PyTorch may report a tensor as
channels-last contiguous while `.view(N, C*H*W)` is illegal.

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

/-
===============================================================================
Step 235: rank-4 layout algebra beyond the rank-2 transpose fragment
===============================================================================
-/

namespace LayoutAlgebra

/-- A concrete tensor layout: logical shape, element strides, and storage offset. -/
structure Layout where
  shape : List Nat
  strides : List Nat
  storageOffset : Nat
  deriving DecidableEq, Repr

/-- Product of dimensions, with scalar tensors/empty suffixes having product 1. -/
def prod : List Nat → Nat
  | [] => 1
  | d :: rest => d * prod rest

/-- Row-major/C-order contiguous strides for an arbitrary concrete shape. -/
def contigStrides : List Nat → List Nat
  | [] => []
  | _ :: rest => prod rest :: contigStrides rest

/-- Exact row-major contiguity predicate for concrete layouts. -/
def isContiguous (l : Layout) : Bool :=
  l.strides == contigStrides l.shape

/-- The non-degenerate NCHW regime where the canonical channels-last strides are
unambiguous and distinct from row-major strides.  Size-1/zero axes are deliberately
outside this executable predicate because PyTorch treats those memory-format
answers as ambiguous. -/
def nondegenerateNCHW : List Nat → Bool
  | [_n, c, h, w] => decide (1 < c) && decide (1 < h) && decide (1 < w)
  | _ => false

/-- Canonical rank-4 NCHW channels-last strides: logical shape remains NCHW, but
C is the innermost storage dimension. -/
def canonicalChannelsLast4Strides : List Nat → Option (List Nat)
  | [n, c, h, w] =>
      if nondegenerateNCHW [n, c, h, w] then
        some [h * w * c, 1, w * c, c]
      else
        none
  | _ => none

/-- Canonical non-degenerate channels-last predicate.  This intentionally
abstains on degenerate shapes where PyTorch may set both row-major and
channels-last contiguity bits. -/
def isCanonicalChannelsLast4 (l : Layout) : Bool :=
  match canonicalChannelsLast4Strides l.shape with
  | some s => s == l.strides
  | none => false

/-- Adjacent dimensions can be collapsed into a view exactly when the left stride
matches `rightSize * rightStride` in the non-degenerate fragment. -/
def canCollapsePair (leftStride rightSize rightStride : Nat) : Bool :=
  leftStride == rightSize * rightStride

/-- Viewability of `x.view(N, C*H*W)` for a concrete NCHW layout in the
non-degenerate CHW-tail fragment. -/
def canViewNCHWTail (l : Layout) : Bool :=
  match l.shape, l.strides with
  | [_n, _c, h, w], [_sn, sc, sh, sw] =>
      canCollapsePair sh w sw && canCollapsePair sc h sh
  | _, _ => false

/-- Resulting layout for a successful `view(N, C*H*W)` without copying. -/
def viewNCHWTail (l : Layout) : Option Layout :=
  match l.shape, l.strides with
  | [n, c, h, w], [sn, _sc, _sh, sw] =>
      if canViewNCHWTail l then
        some { shape := [n, c * h * w], strides := [sn, sw], storageOffset := l.storageOffset }
      else
        none
  | _, _ => none

/-- Metadata-only transpose of dims 0 and 1 preserves storage offset. -/
def transpose01 (l : Layout) : Layout :=
  { shape :=
      match l.shape with
      | a :: b :: rest => b :: a :: rest
      | _ => l.shape
    strides :=
      match l.strides with
      | a :: b :: rest => b :: a :: rest
      | _ => l.strides
    storageOffset := l.storageOffset }

/-- Metadata-only NCHW→NHWC permutation.  This is *not* the same as making a
tensor channels-last contiguous; it only reorders logical axes and strides. -/
def permuteNCHWtoNHWC (l : Layout) : Layout :=
  { shape :=
      match l.shape with
      | [n, c, h, w] => [n, h, w, c]
      | _ => l.shape
    strides :=
      match l.strides with
      | [sn, sc, sh, sw] => [sn, sh, sw, sc]
      | _ => l.strides
    storageOffset := l.storageOffset }

/-- Channel-axis narrow.  The new storage offset is `old + start * stride_C`. -/
def narrowChannel (start newC : Nat) (l : Layout) : Option Layout :=
  match l.shape, l.strides with
  | [n, c, h, w], [_sn, sc, _sh, _sw] =>
      if start + newC ≤ c then
        some {
          shape := [n, newC, h, w],
          strides := l.strides,
          storageOffset := l.storageOffset + start * sc
        }
      else
        none
  | _, _ => none

/-- Any layout built with `contigStrides` is row-major contiguous. -/
theorem contigLayout_is_contiguous (shape : List Nat) (off : Nat) :
    isContiguous { shape := shape, strides := contigStrides shape, storageOffset := off } = true := by
  simp [isContiguous]

/-- Row-major CHW tails always satisfy the adjacent-stride collapse law. -/
theorem rowMajorCHWTail_viewable (n c h w off : Nat) :
    canViewNCHWTail {
      shape := [n, c, h, w],
      strides := contigStrides [n, c, h, w],
      storageOffset := off
    } = true := by
  simp [canViewNCHWTail, canCollapsePair, contigStrides, prod]

/-- Row-major `view(N, C*H*W)` computes the expected two-dimensional layout. -/
theorem rowMajorCHWTail_view_layout (n c h w off : Nat) :
    viewNCHWTail {
      shape := [n, c, h, w],
      strides := contigStrides [n, c, h, w],
      storageOffset := off
    } =
      some {
        shape := [n, c * h * w],
        strides := [prod [c, h, w], 1],
        storageOffset := off
      } := by
  simp [viewNCHWTail, canViewNCHWTail, canCollapsePair, contigStrides, prod]

/-- Metadata-only transpose preserves storage offset for every layout. -/
theorem transpose01_preserves_storageOffset (l : Layout) :
    (transpose01 l).storageOffset = l.storageOffset := rfl

/-- Metadata-only NCHW→NHWC permutation preserves storage offset for every layout. -/
theorem permuteNCHWtoNHWC_preserves_storageOffset (l : Layout) :
    (permuteNCHWtoNHWC l).storageOffset = l.storageOffset := rfl

def rowMajor4 : Layout :=
  { shape := [2, 3, 4, 5], strides := contigStrides [2, 3, 4, 5], storageOffset := 0 }

def channelsLast4 : Layout :=
  { shape := [2, 3, 4, 5], strides := [60, 1, 15, 3], storageOffset := 0 }

theorem rowMajor4_strides :
    rowMajor4.strides = [60, 20, 5, 1] := by decide

theorem rowMajor4_view_tail :
    viewNCHWTail rowMajor4 =
      some { shape := [2, 60], strides := [60, 1], storageOffset := 0 } := by decide

theorem channelsLast4_is_canonical_channels_last :
    isCanonicalChannelsLast4 channelsLast4 = true := by decide

theorem channelsLast4_not_row_major_contiguous :
    isContiguous channelsLast4 = false := by decide

theorem channelsLast4_tail_not_viewable :
    canViewNCHWTail channelsLast4 = false := by decide

theorem channelsLast4_view_tail_rejected :
    viewNCHWTail channelsLast4 = none := by decide

theorem narrowChannel_rowMajor_offset :
    narrowChannel 1 2 rowMajor4 =
      some { shape := [2, 2, 4, 5], strides := [60, 20, 5, 1], storageOffset := 20 } := by decide

theorem narrowChannel_channelsLast_offset :
    narrowChannel 1 2 channelsLast4 =
      some { shape := [2, 2, 4, 5], strides := [60, 1, 15, 3], storageOffset := 1 } := by decide

/-- Degenerate C=1 tensors are a channels-last caveat: PyTorch may report them
both row-major and channels-last contiguous, so the canonical non-degenerate
predicate abstains instead of over-claiming. -/
theorem canonicalChannelsLast4_degenerateC_abstains :
    canonicalChannelsLast4Strides [2, 1, 4, 5] = none := by decide

end LayoutAlgebra

end ContigFlow
end TensorGuard
