/-
TensorGuard chunk/split partition rules, machine-checked in Lean 4
(Step 231).

`src/tensor_shapes.py::compute_chunk_shapes` and `compute_split_shapes`
implement the PyTorch partition contract used by the source analyzer and
model checker:

  * `torch.split(x, n, dim)` partitions the selected axis into sections of
    size `n`, with a possibly smaller final section; for a zero-sized axis it
    returns one empty section.
  * `torch.split(x, [s₀, …], dim)` is admitted exactly when the section sizes
    are non-negative and reconstruct the original axis.
  * `torch.chunk(x, k, dim)` uses a ceil-sized section and may return fewer
    than `k` tensors when the dimension is positive and smaller than `k`; a
    zero-sized axis returns `k` empty tensors.
  * Concatenating every produced piece along the same axis reconstructs the
    original axis exactly.

This Lean file models the axis-local arithmetic and the shape reconstruction
obligation.  The companion test `tests/test_chunksplit_lean_conformance.py`
grounds the theorem-shaped cases against TensorGuard's real Python helpers and
live PyTorch.

Pure Lean 4 core (no mathlib).
-/

namespace TensorGuard
namespace ChunkSplit

/- ===================================================================== -/
/- 1. Small list algebra                                                  -/
/- ===================================================================== -/

/-- Sum of a concrete list of axis-section sizes. -/
def sum : List Nat → Nat
  | [] => 0
  | x :: xs => x + sum xs

/-- Product (numel factor) of concrete dimensions. -/
def prod : List Nat → Nat
  | [] => 1
  | d :: rest => d * prod rest

theorem sum_append (xs ys : List Nat) : sum (xs ++ ys) = sum xs + sum ys := by
  induction xs with
  | nil => simp [sum]
  | cons x rest ih => simp [sum, ih, Nat.add_assoc]

theorem prod_append (xs ys : List Nat) : prod (xs ++ ys) = prod xs * prod ys := by
  induction xs with
  | nil => simp [prod]
  | cons x rest ih => simp [prod, ih, Nat.mul_assoc]

/- ===================================================================== -/
/- 2. Shape model relative to a split/concat axis                         -/
/- ===================================================================== -/

/-- A tensor shape factored around the partition axis. -/
structure AxisShape where
  pre : List Nat
  axisSize : Nat
  post : List Nat
deriving DecidableEq, Repr

/-- Materialize an axis-factored shape back to a flat dimension list. -/
def toList (s : AxisShape) : List Nat := s.pre ++ (s.axisSize :: s.post)

/-- Replace only the split/concat axis size. -/
def withAxis (s : AxisShape) (axis : Nat) : AxisShape :=
  { pre := s.pre, axisSize := axis, post := s.post }

/-- Shape of the conceptual concat of all pieces along the same axis. -/
def concatShape (s : AxisShape) (sections : List Nat) : AxisShape :=
  withAxis s (sum sections)

/-- The list-valued split specification is valid exactly when it reconstructs
    the original split axis.  (Section non-negativity is automatic for `Nat`.) -/
def splitSectionsValid (s : AxisShape) (sections : List Nat) : Bool :=
  decide (sum sections = s.axisSize)

theorem splitValid_iff (s : AxisShape) (sections : List Nat) :
    splitSectionsValid s sections = true ↔ sum sections = s.axisSize := by
  unfold splitSectionsValid
  simp [decide_eq_true_eq]

theorem split_list_mismatch_flagged (s : AxisShape) (sections : List Nat)
    (h : sum sections ≠ s.axisSize) :
    splitSectionsValid s sections = false := by
  unfold splitSectionsValid
  simp [decide_eq_false h]

/-- Concatenating valid split sections reconstructs the original axis size. -/
theorem axisConcat_reconstruct (s : AxisShape) (sections : List Nat)
    (h : sum sections = s.axisSize) :
    (concatShape s sections).axisSize = s.axisSize := by
  unfold concatShape withAxis
  exact h

/-- Valid split sections reconstruct the full shape, not just the axis scalar. -/
theorem splitConcat_shape (s : AxisShape) (sections : List Nat)
    (h : splitSectionsValid s sections = true) :
    toList (concatShape s sections) = toList s := by
  have hs : sum sections = s.axisSize := (splitValid_iff s sections).mp h
  unfold toList concatShape withAxis
  simp [hs]

/-- Exact axis reconstruction preserves the concrete element count. -/
theorem splitConcat_numel (s : AxisShape) (sections : List Nat)
    (h : splitSectionsValid s sections = true) :
    prod (toList (concatShape s sections)) = prod (toList s) := by
  rw [splitConcat_shape s sections h]

/- ===================================================================== -/
/- 3. Torch split/chunk size generators                                   -/
/- ===================================================================== -/

/-- Ceiling division used by PyTorch chunk for positive divisors. -/
def ceilDiv (n d : Nat) : Nat :=
  if d = 0 then 0 else (n + d - 1) / d

/-- Fuel-bounded split-by-size generator.  The first argument is structural
    fuel; callers pass `total` as fuel, so at most one positive element of axis
    size is consumed per step. -/
def splitBySizeFuel : Nat → Nat → Nat → List Nat
  | 0, _, _ => []
  | fuel + 1, remaining, step =>
      if remaining = 0 then []
      else Nat.min step remaining :: splitBySizeFuel fuel (remaining - step) step

/-- Axis sizes for `torch.split(x, splitSize, dim)` on concrete non-negative
    axes.  The zero-axis behavior follows PyTorch/TensorGuard: one empty split. -/
def splitIntSizes (total splitSize : Nat) : Option (List Nat) :=
  if splitSize = 0 then
    if total = 0 then some [0] else none
  else if total = 0 then
    some [0]
  else
    some (splitBySizeFuel total total splitSize)

/-- Axis sizes for `torch.chunk(x, chunks, dim)` on concrete non-negative axes. -/
def chunkSizes (total chunks : Nat) : Option (List Nat) :=
  if chunks = 0 then
    none
  else if total = 0 then
    some (List.replicate chunks 0)
  else
    some (splitBySizeFuel total total (ceilDiv total chunks))

/- ===================================================================== -/
/- 4. Mechanized PyTorch edge cases used by the conformance bridge        -/
/- ===================================================================== -/

theorem split_int_uneven_example :
    splitIntSizes 13 6 = some [6, 6, 1] := by
  rfl

theorem split_int_tail_example :
    splitIntSizes 10 3 = some [3, 3, 3, 1] := by
  rfl

theorem split_int_zero_axis_example :
    splitIntSizes 0 0 = some [0] := by
  rfl

theorem split_list_with_empty_section_valid :
    splitSectionsValid { pre := [2], axisSize := 5, post := [] } [2, 0, 3] = true := by
  rfl

theorem split_list_mismatch_example :
    splitSectionsValid { pre := [2], axisSize := 5, post := [] } [2, 0, 2] = false := by
  rfl

theorem chunk_uneven_example :
    chunkSizes 10 3 = some [4, 4, 2] := by
  rfl

theorem chunk_many_sections_example :
    chunkSizes 13 6 = some [3, 3, 3, 3, 1] := by
  rfl

theorem chunk_fewer_than_requested_example :
    chunkSizes 5 8 = some [1, 1, 1, 1, 1] := by
  rfl

theorem chunk_fewer_than_requested_len :
    (match chunkSizes 5 8 with | some xs => xs.length | none => 0) = 5 := by
  rfl

theorem chunk_zero_axis_returns_requested_empties :
    chunkSizes 0 3 = some [0, 0, 0] := by
  rfl

theorem split_concat_reconstruct_example :
    toList (concatShape { pre := [2], axisSize := 5, post := [4] } [2, 0, 3])
      = [2, 5, 4] := by
  rfl

theorem chunk_concat_reconstruct_example :
    toList (concatShape { pre := [2], axisSize := 10, post := [4] } [4, 4, 2])
      = [2, 10, 4] := by
  rfl

end ChunkSplit
end TensorGuard
