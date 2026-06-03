/-
TensorGuard grid_sample / affine_grid shape rules, machine-checked in Lean 4
(Step 233).

`src/grid_sample_verify.py` models the concrete PyTorch contracts for
`torch.nn.functional.grid_sample` and `affine_grid`:

  * `grid_sample` has two spatial variants: 2-D sampling over rank-4 inputs
    `(N,C,H,W)` with grids `(N,H_out,W_out,2)`, and 3-D sampling over rank-5
    inputs `(N,C,D,H,W)` with grids `(N,D_out,H_out,W_out,3)`;
  * input spatial axes must be positive, while batch, channels, and output-grid
    extents may be zero;
  * `affine_grid` receives a rank-3 theta matrix, a positive rank-4 or rank-5
    output size, and returns the matching coordinate grid.

This file mechanizes the concrete shape algebra.  Dtype/mode checks remain in
the Python verifier; symbolic dimensions are intentionally outside this
`List Nat` model, matching the other concrete Lean operator rules.

Pure Lean 4 core (no mathlib).
-/

namespace TensorGuard
namespace GridSample

/- ===================================================================== -/
/- 1. grid_sample shape contract                                         -/
/- ===================================================================== -/

def gridSample2DValid (n h w gn coord : Nat) : Bool :=
  decide (0 < h) && decide (0 < w) && decide (gn = n) && decide (coord = 2)

def gridSample3DValid (n d h w gn coord : Nat) : Bool :=
  decide (0 < d) && decide (0 < h) && decide (0 < w) &&
    decide (gn = n) && decide (coord = 3)

/-- Shape-only `grid_sample` transfer over concrete dimensions. -/
def gridSample? : List Nat → List Nat → Option (List Nat)
  | n :: c :: h :: w :: [], gn :: oh :: ow :: coord :: [] =>
      if gridSample2DValid n h w gn coord then
        some [n, c, oh, ow]
      else
        none
  | n :: c :: d :: h :: w :: [], gn :: od :: oh :: ow :: coord :: [] =>
      if gridSample3DValid n d h w gn coord then
        some [n, c, od, oh, ow]
      else
        none
  | _, _ => none

theorem gridSample2DValid_iff (n h w gn coord : Nat) :
    gridSample2DValid n h w gn coord = true ↔
      (((0 < h ∧ 0 < w) ∧ gn = n) ∧ coord = 2) := by
  unfold gridSample2DValid
  simp [Bool.and_eq_true]

theorem gridSample3DValid_iff (n d h w gn coord : Nat) :
    gridSample3DValid n d h w gn coord = true ↔
      ((((0 < d ∧ 0 < h) ∧ 0 < w) ∧ gn = n) ∧ coord = 3) := by
  unfold gridSample3DValid
  simp [Bool.and_eq_true]

theorem gridSample2D_valid_link
    (n c h w gn oh ow coord : Nat)
    (hvalid : gridSample2DValid n h w gn coord = true) :
    gridSample? [n, c, h, w] [gn, oh, ow, coord] = some [n, c, oh, ow] := by
  simp [gridSample?, hvalid]

theorem gridSample3D_valid_link
    (n c d h w gn od oh ow coord : Nat)
    (hvalid : gridSample3DValid n d h w gn coord = true) :
    gridSample? [n, c, d, h, w] [gn, od, oh, ow, coord] =
      some [n, c, od, oh, ow] := by
  simp [gridSample?, hvalid]

theorem gridSample2D_invalid_link
    (n c h w gn oh ow coord : Nat)
    (hvalid : gridSample2DValid n h w gn coord = false) :
    gridSample? [n, c, h, w] [gn, oh, ow, coord] = none := by
  simp [gridSample?, hvalid]

theorem gridSample3D_invalid_link
    (n c d h w gn od oh ow coord : Nat)
    (hvalid : gridSample3DValid n d h w gn coord = false) :
    gridSample? [n, c, d, h, w] [gn, od, oh, ow, coord] = none := by
  simp [gridSample?, hvalid]

theorem gridSample2D_output_shape
    (n c h w oh ow : Nat) (hh : 0 < h) (hw : 0 < w) :
    gridSample? [n, c, h, w] [n, oh, ow, 2] = some [n, c, oh, ow] := by
  simp [gridSample?, gridSample2DValid, hh, hw]

theorem gridSample3D_output_shape
    (n c d h w od oh ow : Nat) (hd : 0 < d) (hh : 0 < h) (hw : 0 < w) :
    gridSample? [n, c, d, h, w] [n, od, oh, ow, 3] =
      some [n, c, od, oh, ow] := by
  simp [gridSample?, gridSample3DValid, hd, hh, hw]

theorem gridSample2D_output_rank (n c oh ow : Nat) :
    [n, c, oh, ow].length = 4 := by
  rfl

theorem gridSample3D_output_rank (n c od oh ow : Nat) :
    [n, c, od, oh, ow].length = 5 := by
  rfl

theorem gridSample_wrong_input_rank_rejected (n c h gn gh coord : Nat) :
    gridSample? [n, c, h] [gn, gh, coord] = none := by
  rfl

theorem gridSample_grid_rank_mismatch_rejected
    (n c h w gn oh ow coord extra : Nat) :
    gridSample? [n, c, h, w] [gn, oh, ow, coord, extra] = none := by
  rfl

theorem gridSample2D_coord_dim_flagged (n c h w oh ow : Nat) :
    gridSample? [n, c, h, w] [n, oh, ow, 3] = none := by
  simp [gridSample?, gridSample2DValid]

theorem gridSample3D_coord_dim_flagged (n c d h w od oh ow : Nat) :
    gridSample? [n, c, d, h, w] [n, od, oh, ow, 2] = none := by
  simp [gridSample?, gridSample3DValid]

theorem gridSample2D_zero_height_flagged (n c w oh ow : Nat) :
    gridSample? [n, c, 0, w] [n, oh, ow, 2] = none := by
  simp [gridSample?, gridSample2DValid]

theorem gridSample2D_zero_width_flagged (n c h oh ow : Nat) :
    gridSample? [n, c, h, 0] [n, oh, ow, 2] = none := by
  simp [gridSample?, gridSample2DValid]

theorem gridSample3D_zero_depth_flagged (n c h w od oh ow : Nat) :
    gridSample? [n, c, 0, h, w] [n, od, oh, ow, 3] = none := by
  simp [gridSample?, gridSample3DValid]

theorem gridSample3D_zero_height_flagged (n c d w od oh ow : Nat) :
    gridSample? [n, c, d, 0, w] [n, od, oh, ow, 3] = none := by
  simp [gridSample?, gridSample3DValid]

theorem gridSample3D_zero_width_flagged (n c d h od oh ow : Nat) :
    gridSample? [n, c, d, h, 0] [n, od, oh, ow, 3] = none := by
  simp [gridSample?, gridSample3DValid]

theorem gridSample2D_batch_mismatch_flagged
    (n gn c h w oh ow : Nat) (hne : gn ≠ n) :
    gridSample? [n, c, h, w] [gn, oh, ow, 2] = none := by
  simp [gridSample?, gridSample2DValid, hne]

theorem gridSample_accepts_empty_output_grid
    (n c h w ow : Nat) (hh : 0 < h) (hw : 0 < w) :
    gridSample? [n, c, h, w] [n, 0, ow, 2] = some [n, c, 0, ow] := by
  simp [gridSample?, gridSample2DValid, hh, hw]

/- ===================================================================== -/
/- 2. affine_grid shape contract                                         -/
/- ===================================================================== -/

def affineGrid2DValid (tn rows cols n c h w : Nat) : Bool :=
  decide (0 < n) && decide (0 < c) && decide (0 < h) && decide (0 < w) &&
    decide (0 < tn) && decide (tn = n) && decide (rows = 2) && decide (cols = 3)

def affineGrid3DValid (tn rows cols n c d h w : Nat) : Bool :=
  decide (0 < n) && decide (0 < c) && decide (0 < d) &&
    decide (0 < h) && decide (0 < w) && decide (0 < tn) &&
    decide (tn = n) && decide (rows = 3) && decide (cols = 4)

/-- Shape-only `affine_grid` transfer over concrete dimensions. -/
def affineGrid? : List Nat → List Nat → Option (List Nat)
  | tn :: rows :: cols :: [], n :: c :: h :: w :: [] =>
      if affineGrid2DValid tn rows cols n c h w then
        some [n, h, w, 2]
      else
        none
  | tn :: rows :: cols :: [], n :: c :: d :: h :: w :: [] =>
      if affineGrid3DValid tn rows cols n c d h w then
        some [n, d, h, w, 3]
      else
        none
  | _, _ => none

theorem affineGrid2DValid_iff (tn rows cols n c h w : Nat) :
    affineGrid2DValid tn rows cols n c h w = true ↔
      (((((((0 < n ∧ 0 < c) ∧ 0 < h) ∧ 0 < w) ∧
        0 < tn) ∧ tn = n) ∧ rows = 2) ∧ cols = 3) := by
  unfold affineGrid2DValid
  simp [Bool.and_eq_true]

theorem affineGrid3DValid_iff (tn rows cols n c d h w : Nat) :
    affineGrid3DValid tn rows cols n c d h w = true ↔
      ((((((((0 < n ∧ 0 < c) ∧ 0 < d) ∧ 0 < h) ∧ 0 < w) ∧
        0 < tn) ∧ tn = n) ∧ rows = 3) ∧ cols = 4) := by
  unfold affineGrid3DValid
  simp [Bool.and_eq_true]

theorem affineGrid2D_valid_link
    (tn rows cols n c h w : Nat)
    (hvalid : affineGrid2DValid tn rows cols n c h w = true) :
    affineGrid? [tn, rows, cols] [n, c, h, w] = some [n, h, w, 2] := by
  simp [affineGrid?, hvalid]

theorem affineGrid3D_valid_link
    (tn rows cols n c d h w : Nat)
    (hvalid : affineGrid3DValid tn rows cols n c d h w = true) :
    affineGrid? [tn, rows, cols] [n, c, d, h, w] = some [n, d, h, w, 3] := by
  simp [affineGrid?, hvalid]

theorem affineGrid2D_invalid_link
    (tn rows cols n c h w : Nat)
    (hvalid : affineGrid2DValid tn rows cols n c h w = false) :
    affineGrid? [tn, rows, cols] [n, c, h, w] = none := by
  simp [affineGrid?, hvalid]

theorem affineGrid3D_invalid_link
    (tn rows cols n c d h w : Nat)
    (hvalid : affineGrid3DValid tn rows cols n c d h w = false) :
    affineGrid? [tn, rows, cols] [n, c, d, h, w] = none := by
  simp [affineGrid?, hvalid]

theorem affineGrid2D_output_shape
    (n c h w : Nat) (hn : 0 < n) (hc : 0 < c) (hh : 0 < h) (hw : 0 < w) :
    affineGrid? [n, 2, 3] [n, c, h, w] = some [n, h, w, 2] := by
  simp [affineGrid?, affineGrid2DValid, hn, hc, hh, hw]

theorem affineGrid3D_output_shape
    (n c d h w : Nat)
    (hn : 0 < n) (hc : 0 < c) (hd : 0 < d) (hh : 0 < h) (hw : 0 < w) :
    affineGrid? [n, 3, 4] [n, c, d, h, w] = some [n, d, h, w, 3] := by
  simp [affineGrid?, affineGrid3DValid, hn, hc, hd, hh, hw]

theorem affineGrid2D_output_rank (n h w : Nat) :
    [n, h, w, 2].length = 4 := by
  rfl

theorem affineGrid3D_output_rank (n d h w : Nat) :
    [n, d, h, w, 3].length = 5 := by
  rfl

theorem affineGrid_size_rank_rejected (tn rows cols n c h : Nat) :
    affineGrid? [tn, rows, cols] [n, c, h] = none := by
  rfl

theorem affineGrid_theta_rank_rejected (n c h w : Nat) :
    affineGrid? [n, 2] [n, c, h, w] = none := by
  rfl

theorem affineGrid2D_theta_rows_flagged (n c h w : Nat) :
    affineGrid? [n, 3, 3] [n, c, h, w] = none := by
  simp [affineGrid?, affineGrid2DValid]

theorem affineGrid2D_theta_cols_flagged (n c h w : Nat) :
    affineGrid? [n, 2, 4] [n, c, h, w] = none := by
  simp [affineGrid?, affineGrid2DValid]

theorem affineGrid3D_theta_rows_flagged (n c d h w : Nat) :
    affineGrid? [n, 2, 4] [n, c, d, h, w] = none := by
  simp [affineGrid?, affineGrid3DValid]

theorem affineGrid3D_theta_cols_flagged (n c d h w : Nat) :
    affineGrid? [n, 3, 3] [n, c, d, h, w] = none := by
  simp [affineGrid?, affineGrid3DValid]

theorem affineGrid2D_size_batch_positive_required (c h w : Nat) :
    affineGrid? [0, 2, 3] [0, c, h, w] = none := by
  simp [affineGrid?, affineGrid2DValid]

theorem affineGrid2D_size_channel_positive_required (n h w : Nat) :
    affineGrid? [n, 2, 3] [n, 0, h, w] = none := by
  simp [affineGrid?, affineGrid2DValid]

theorem affineGrid2D_size_height_positive_required (n c w : Nat) :
    affineGrid? [n, 2, 3] [n, c, 0, w] = none := by
  simp [affineGrid?, affineGrid2DValid]

theorem affineGrid2D_size_width_positive_required (n c h : Nat) :
    affineGrid? [n, 2, 3] [n, c, h, 0] = none := by
  simp [affineGrid?, affineGrid2DValid]

theorem affineGrid3D_size_depth_positive_required (n c h w : Nat) :
    affineGrid? [n, 3, 4] [n, c, 0, h, w] = none := by
  simp [affineGrid?, affineGrid3DValid]

theorem affineGrid_theta_batch_positive_required (c h w : Nat) :
    affineGrid? [0, 2, 3] [1, c, h, w] = none := by
  simp [affineGrid?, affineGrid2DValid]

theorem affineGrid2D_batch_mismatch_flagged
    (tn n c h w : Nat) (hne : tn ≠ n) :
    affineGrid? [tn, 2, 3] [n, c, h, w] = none := by
  simp [affineGrid?, affineGrid2DValid, hne]

end GridSample
end TensorGuard
