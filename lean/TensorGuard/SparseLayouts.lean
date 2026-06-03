/-
TensorGuard sparse-layout shape invariants for PyTorch COO/CSR/CSC/BSR/BSC
tensors (Step 236).

Scope: this file mechanizes the concrete, shape-only contract modeled by
`src/sparse_verify.py`.  For compressed sparse layouts that contract is PyTorch's
`check_invariants=True` constructor regime plus TensorGuard's additional
`to_dense` usability check for value dense-tail agreement.  PyTorch may lazily
accept some malformed compressed tensors without invariant checking; this module
does not claim soundness for that permissive mode.

The key theorem family below is executable: accepted constructor models always
produce a sparse spec whose dense materialization shape is exactly the requested
tensor size.  Concrete theorem-shaped examples cover COO, CSR, CSC, BSR and BSC,
including batched and blocked cases, plus the main rejection modes.

Pure Lean 4 core (no mathlib).
-/

namespace TensorGuard
namespace SparseLayouts

/-- Sparse layouts modeled by TensorGuard's shape-only sparse checker. -/
inductive LayoutKind
  | coo
  | csr
  | csc
  | bsr
  | bsc
  deriving DecidableEq, Repr

/-- Resolved static facts for a sparse tensor layout. -/
structure SparseSpec where
  layout : LayoutKind
  shape : List Nat
  sparseShape : List Nat
  denseShape : List Nat
  batchShape : List Nat
  nnz : Nat
  blockRows : Option Nat
  blockCols : Option Nat
  deriving DecidableEq, Repr

/-- Total list indexing used only after Boolean rank guards have been checked. -/
def nthD : List Nat → Nat → Nat
  | [], _ => 0
  | x :: _, 0 => x
  | _ :: xs, n + 1 => nthD xs n

/-- A successful sparse constructor materializes to its logical dense shape. -/
def denseShapeOf (spec : SparseSpec) : List Nat := spec.shape

/-- Shape-soundness postcondition for an optional constructor model. -/
def acceptedShape (requested : List Nat) : Option SparseSpec → Prop
  | some spec => denseShapeOf spec = requested
  | none => True

/-- Shared constructor guard: invalid metadata abstains/rejects; valid metadata
records a spec whose logical shape is the requested tensor size. -/
def mkAccepted (ok : Bool) (spec : SparseSpec) : Option SparseSpec :=
  if ok then some spec else none

theorem dense_materialization_shape_sound (spec : SparseSpec) :
    denseShapeOf spec = spec.shape := rfl

theorem mkAccepted_dense_shape_sound (ok : Bool) (spec : SparseSpec) :
    acceptedShape spec.shape (mkAccepted ok spec) := by
  cases ok <;> simp [acceptedShape, mkAccepted, denseShapeOf]

/-- COO constructor shape contract:
indices has shape `(sparse_dim, nnz)`, values has shape `(nnz, *dense_tail)`,
and the dense tail equals `size.drop sparse_dim`. -/
def cooSpec (indicesShape valuesShape size : List Nat) : Option SparseSpec :=
  match indicesShape, valuesShape with
  | [sparseDim, nnz], valuesNnz :: valuesDenseTail =>
      mkAccepted
        ((nnz == valuesNnz)
          && decide (sparseDim ≤ size.length)
          && (valuesDenseTail == List.drop sparseDim size))
        {
          layout := LayoutKind.coo,
          shape := size,
          sparseShape := List.take sparseDim size,
          denseShape := List.drop sparseDim size,
          batchShape := [],
          nnz := nnz,
          blockRows := none,
          blockCols := none
        }
  | _, _ => none

def positiveDivides (extent block : Nat) : Bool :=
  decide (0 < block) && (extent % block == 0)

def isBlocked : LayoutKind → Bool
  | LayoutKind.bsr => true
  | LayoutKind.bsc => true
  | _ => false

def compressedAxisOk
    (layout : LayoutKind)
    (rows cols compressedLen blockRows blockCols : Nat) : Bool :=
  match layout with
  | LayoutKind.csr => compressedLen == rows + 1
  | LayoutKind.csc => compressedLen == cols + 1
  | LayoutKind.bsr =>
      positiveDivides rows blockRows
        && positiveDivides cols blockCols
        && (compressedLen == rows / blockRows + 1)
  | LayoutKind.bsc =>
      positiveDivides rows blockRows
        && positiveDivides cols blockCols
        && (compressedLen == cols / blockCols + 1)
  | LayoutKind.coo => false

/-- Shared CSR/CSC/BSR/BSC shape contract.  `compressedShape`, `plainShape`, and
`valuesShape` are the shapes of the compressed index, plain index, and value
tensors respectively; `size` is the requested dense tensor shape. -/
def compressedSpec
    (layout : LayoutKind)
    (compressedShape plainShape valuesShape size : List Nat) : Option SparseSpec :=
  if compressedShape.length == 0 then
    none
  else
    let batchRank := compressedShape.length - 1
    let batchShape := List.take batchRank size
    let minValuesRank := batchRank + (if isBlocked layout then 3 else 1)
    let valuesDenseTail := List.drop minValuesRank valuesShape
    let rows := nthD size batchRank
    let cols := nthD size (batchRank + 1)
    let compressedLen := nthD compressedShape batchRank
    let valuesNnz := nthD valuesShape batchRank
    let blockRows := nthD valuesShape (batchRank + 1)
    let blockCols := nthD valuesShape (batchRank + 2)
    let ok :=
      (plainShape.length == batchRank + 1)
        && decide (minValuesRank ≤ valuesShape.length)
        && decide (batchRank + 2 ≤ size.length)
        && (size.length == batchRank + 2 + valuesDenseTail.length)
        && (List.take batchRank compressedShape == batchShape)
        && (List.take batchRank plainShape == batchShape)
        && (List.take batchRank valuesShape == batchShape)
        && (nthD plainShape batchRank == valuesNnz)
        && (valuesDenseTail == List.drop (batchRank + 2) size)
        && compressedAxisOk layout rows cols compressedLen blockRows blockCols
    mkAccepted ok {
      layout := layout,
      shape := size,
      sparseShape := [rows, cols],
      denseShape := valuesDenseTail,
      batchShape := batchShape,
      nnz := valuesNnz,
      blockRows := if isBlocked layout then some blockRows else none,
      blockCols := if isBlocked layout then some blockCols else none
    }

def csrSpec (crowShape colShape valuesShape size : List Nat) : Option SparseSpec :=
  compressedSpec LayoutKind.csr crowShape colShape valuesShape size

def cscSpec (ccolShape rowShape valuesShape size : List Nat) : Option SparseSpec :=
  compressedSpec LayoutKind.csc ccolShape rowShape valuesShape size

def bsrSpec (crowShape colShape valuesShape size : List Nat) : Option SparseSpec :=
  compressedSpec LayoutKind.bsr crowShape colShape valuesShape size

def bscSpec (ccolShape rowShape valuesShape size : List Nat) : Option SparseSpec :=
  compressedSpec LayoutKind.bsc ccolShape rowShape valuesShape size

/- ========================================================================== -/
/- Constructor acceptance examples                                             -/
/- ========================================================================== -/

theorem coo234_accepts :
    cooSpec [2, 3] [3, 4] [2, 3, 4] =
      some {
        layout := LayoutKind.coo,
        shape := [2, 3, 4],
        sparseShape := [2, 3],
        denseShape := [4],
        batchShape := [],
        nnz := 3,
        blockRows := none,
        blockCols := none
      } := by decide

theorem csr23_accepts :
    csrSpec [3] [3] [3] [2, 3] =
      some {
        layout := LayoutKind.csr,
        shape := [2, 3],
        sparseShape := [2, 3],
        denseShape := [],
        batchShape := [],
        nnz := 3,
        blockRows := none,
        blockCols := none
      } := by decide

theorem csc23_accepts :
    cscSpec [4] [3] [3] [2, 3] =
      some {
        layout := LayoutKind.csc,
        shape := [2, 3],
        sparseShape := [2, 3],
        denseShape := [],
        batchShape := [],
        nnz := 3,
        blockRows := none,
        blockCols := none
      } := by decide

theorem bsr43_accepts :
    bsrSpec [3] [2] [2, 2, 3] [4, 3] =
      some {
        layout := LayoutKind.bsr,
        shape := [4, 3],
        sparseShape := [4, 3],
        denseShape := [],
        batchShape := [],
        nnz := 2,
        blockRows := some 2,
        blockCols := some 3
      } := by decide

theorem bsc23_accepts :
    bscSpec [2] [1] [1, 2, 3] [2, 3] =
      some {
        layout := LayoutKind.bsc,
        shape := [2, 3],
        sparseShape := [2, 3],
        denseShape := [],
        batchShape := [],
        nnz := 1,
        blockRows := some 2,
        blockCols := some 3
      } := by decide

theorem batched_csr_accepts :
    csrSpec [2, 3] [2, 3] [2, 3, 4] [2, 2, 3, 4] =
      some {
        layout := LayoutKind.csr,
        shape := [2, 2, 3, 4],
        sparseShape := [2, 3],
        denseShape := [4],
        batchShape := [2],
        nnz := 3,
        blockRows := none,
        blockCols := none
      } := by decide

theorem batched_bsr_accepts :
    bsrSpec [2, 3] [2, 2] [2, 2, 2, 3, 5] [2, 4, 3, 5] =
      some {
        layout := LayoutKind.bsr,
        shape := [2, 4, 3, 5],
        sparseShape := [4, 3],
        denseShape := [5],
        batchShape := [2],
        nnz := 2,
        blockRows := some 2,
        blockCols := some 3
      } := by decide

/- ========================================================================== -/
/- Dense-materialization shape theorems                                        -/
/- ========================================================================== -/

theorem coo234_toDense_shape :
    Option.map denseShapeOf (cooSpec [2, 3] [3, 4] [2, 3, 4]) =
      some [2, 3, 4] := by decide

theorem csr23_toDense_shape :
    Option.map denseShapeOf (csrSpec [3] [3] [3] [2, 3]) =
      some [2, 3] := by decide

theorem csc23_toDense_shape :
    Option.map denseShapeOf (cscSpec [4] [3] [3] [2, 3]) =
      some [2, 3] := by decide

theorem bsr43_toDense_shape :
    Option.map denseShapeOf (bsrSpec [3] [2] [2, 2, 3] [4, 3]) =
      some [4, 3] := by decide

theorem bsc23_toDense_shape :
    Option.map denseShapeOf (bscSpec [2] [1] [1, 2, 3] [2, 3]) =
      some [2, 3] := by decide

theorem batched_csr_toDense_shape :
    Option.map denseShapeOf (csrSpec [2, 3] [2, 3] [2, 3, 4] [2, 2, 3, 4]) =
      some [2, 2, 3, 4] := by decide

theorem batched_bsr_toDense_shape :
    Option.map denseShapeOf (bsrSpec [2, 3] [2, 2] [2, 2, 2, 3, 5] [2, 4, 3, 5]) =
      some [2, 4, 3, 5] := by decide

/- ========================================================================== -/
/- Rejection examples                                                          -/
/- ========================================================================== -/

theorem csr_bad_compressed_length_rejected :
    csrSpec [2] [3] [3] [2, 3] = none := by decide

theorem csc_bad_compressed_length_rejected :
    cscSpec [3] [3] [3] [2, 3] = none := by decide

theorem bsr_bad_row_divisibility_rejected :
    bsrSpec [3] [2] [2, 2, 3] [5, 3] = none := by decide

theorem bsr_bad_column_divisibility_rejected :
    bsrSpec [3] [2] [2, 2, 3] [4, 4] = none := by decide

theorem compressed_dense_tail_mismatch_rejected :
    csrSpec [3] [3] [3, 5] [2, 3, 4] = none := by decide

end SparseLayouts
end TensorGuard
