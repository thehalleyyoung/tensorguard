/-
TensorGuard Parity Extension for Track G (Lean ↔ Python parity).

Extends the core soundness proofs with additional operators to achieve
≥20 operator rules for comprehensive parity testing with Python implementation.

NOTE: This file is fully closed under `lake build` (no `sorry`). The
shape-transfer definitions are also exposed by `lean_rules_mirror.py`
for runtime parity testing.
-/

import TensorGuard.Extended

namespace TensorGuard

/-! ## Conv2d output formula (full 2D version) -/

def conv2dOutH (h_in pad dilation k stride : Nat) : Option Nat :=
  conv1dOut h_in pad dilation k stride

def conv2dOutW (w_in pad dilation k stride : Nat) : Option Nat :=
  conv1dOut w_in pad dilation k stride

/-! ## Conv3d output formula -/

def conv3dOutD (d_in pad dilation k stride : Nat) : Option Nat :=
  conv1dOut d_in pad dilation k stride

/-! ## MaxPool2d / AvgPool2d output formulas -/

def maxpool2dOutH (h_in pad k stride : Nat) : Option Nat :=
  conv1dOut h_in pad 1 k stride

def maxpool2dOutW (w_in pad k stride : Nat) : Option Nat :=
  conv1dOut w_in pad 1 k stride

def avgpool2dOutH := maxpool2dOutH
def avgpool2dOutW := maxpool2dOutW

/-! ## Cat (concatenation along axis) -/

def catAlong (shapes : List (List Nat)) (axis : Nat) : Option (List Nat) :=
  match shapes with
  | [] => none
  | s :: ss =>
      if axis >= s.length then none
      else if ss.all (fun t => 
        s.length = t.length ∧ 
        (List.range s.length).all (fun k => 
          k = axis ∨ s.get? k = t.get? k)) then
        let out_axis_dim := shapes.foldl (fun acc sh => 
          match sh.get? axis with
          | none => acc
          | some d => acc + d) 0;
        some (s.enum.map (fun (i, d) => if i = axis then out_axis_dim else d))
      else none

/-! ## Stack (insert new axis) -/

def stack (shapes : List (List Nat)) (axis : Nat) : Option (List Nat) :=
  match shapes with
  | [] => none
  | s :: ss =>
      if ss.all (· = s) then
        let n := shapes.length;
        some (s.take axis ++ [n] ++ s.drop axis)
      else none

/-! ## Squeeze (drop dim of size 1) -/

def squeeze (shape : List Nat) : List Nat :=
  shape.filter (· ≠ 1)

theorem squeeze_removes_ones (shape : List Nat) (d : Nat) (h : d ∈ squeeze shape) : d ≠ 1 := by
  simp [squeeze, List.mem_filter] at h
  exact h.2

/-! ## Unsqueeze (insert dim of size 1) -/

def unsqueeze (shape : List Nat) (axis : Nat) : List Nat :=
  shape.take axis ++ [1] ++ shape.drop axis

/-! ## Flatten (range start..end) -/

def flatten (shape : List Nat) (start end_ : Nat) : Option (List Nat) :=
  if end_ ≤ start ∨ end_ > shape.length then none
  else
    let middle := shape.drop start |>.take (end_ - start);
    let flat_dim := middle.foldl (· * ·) 1;
    some (shape.take start ++ [flat_dim] ++ shape.drop end_)

/-! ## Split (along axis, equal chunks) -/

def split (shape : List Nat) (axis chunks : Nat) : Option (List Nat) :=
  match shape.get? axis with
  | none => none
  | some d =>
      if chunks = 0 ∨ d % chunks ≠ 0 then none
      else
        let new_d := d / chunks;
        some (shape.enum.map (fun (i, dim) => if i = axis then new_d else dim))

/-! ## Chunk (similar to split) -/

def chunk (shape : List Nat) (axis chunk_size : Nat) : Option (List Nat) :=
  match shape.get? axis with
  | none => none
  | some _d =>
      if chunk_size = 0 then none
      else some shape  -- For simplicity, model as identity

/-! ## LayerNorm shape (identity) -/

def layerNormShape (shape : List Nat) (normalized_dims : Nat) : Option (List Nat) :=
  if normalized_dims ≤ shape.length then some shape
  else none

theorem layerNorm_identity
    (shape : List Nat) (normalized_dims : Nat) (out : List Nat)
    (h : layerNormShape shape normalized_dims = some out) :
    out = shape := by
  simp only [layerNormShape] at h
  by_cases hc : normalized_dims ≤ shape.length
  · simp [hc] at h; exact h.symm
  · simp [hc] at h

/-! ## Linear with k-rank input -/

def linearShape (shape : List Nat) (in_features out_features : Nat) : Option (List Nat) :=
  match shape.reverse with
  | [] => none
  | last :: rest =>
      if last = in_features then
        some ((out_features :: rest).reverse)
      else none

/-! ## Embedding (input ids → +embed_dim) -/

def embeddingShape (input_shape : List Nat) (embed_dim : Nat) : List Nat :=
  input_shape ++ [embed_dim]

theorem embedding_appends
    (input_shape : List Nat) (embed_dim : Nat) :
    (embeddingShape input_shape embed_dim).length = input_shape.length + 1 := by
  simp [embeddingShape, List.length_append]

end TensorGuard
