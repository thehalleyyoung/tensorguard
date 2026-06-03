/-
TensorGuard named-tensor refine/align rules, machine-checked in Lean 4
(Step 232).

`src/named_tensor_verify.py` models the PyTorch contracts for
`Tensor.refine_names` and `Tensor.align_to`:

  * `refine_names` may fill an unnamed dimension, but an existing concrete name
    must be preserved exactly;
  * concrete output names must be unique;
  * the no-ellipsis core of `align_to` permutes existing named dimensions by
    name, inserts singleton axes for fresh target names (and explicit unnamed
    targets), and rejects unnamed input dimensions because PyTorch requires an
    ellipsis to carry them through.

This Lean file models that no-ellipsis core over integer name identifiers.  The
companion test `tests/test_namedtensor_lean_conformance.py` maps those theorem
shapes to valid PyTorch string names and differentially checks the Python helper
against live named tensors.

Pure Lean 4 core (no mathlib).
-/

namespace TensorGuard
namespace NamedTensor

/- ===================================================================== -/
/- 1. Name and shape model                                                -/
/- ===================================================================== -/

/-- Named-tensor axis names.  `anon` corresponds to PyTorch's `None`; concrete
    names are modeled by natural-number identifiers. -/
inductive AxisName where
  | anon
  | named (id : Nat)
deriving DecidableEq, Repr

open AxisName

/-- A concrete shape paired with one optional name per axis. -/
structure NamedShape where
  shape : List Nat
  names : List AxisName
deriving DecidableEq, Repr

/-- Whether a concrete name occurs in a name list.  Anonymous axes are ignored. -/
def containsNamed (needle : Nat) : List AxisName → Bool
  | [] => false
  | anon :: rest => containsNamed needle rest
  | named n :: rest => decide (n = needle) || containsNamed needle rest

/-- PyTorch named tensors reject duplicate concrete names; duplicate `None`
    axes are allowed and therefore ignored here. -/
def uniqueNamed : List AxisName → Bool
  | [] => true
  | anon :: rest => uniqueNamed rest
  | named n :: rest => !containsNamed n rest && uniqueNamed rest

theorem containsNamed_head (n : Nat) (rest : List AxisName) :
    containsNamed n (named n :: rest) = true := by
  unfold containsNamed
  simp

theorem unique_named_duplicate_head (n : Nat) (rest : List AxisName) :
    uniqueNamed (named n :: named n :: rest) = false := by
  unfold uniqueNamed containsNamed
  simp

theorem uniqueNamed_allows_repeated_anon (rest : List AxisName) :
    uniqueNamed (anon :: anon :: rest) = uniqueNamed rest := by
  rfl

/- ===================================================================== -/
/- 2. refine_names                                                        -/
/- ===================================================================== -/

/-- Per-axis `refine_names`: anonymous axes may be filled; concrete axes must be
    preserved exactly and cannot be demoted to anonymous. -/
def refineAxis? : AxisName → AxisName → Option AxisName
  | anon, new => some new
  | named old, named new => if old = new then some (named old) else none
  | named _, anon => none

/-- Pointwise refine over a full name tuple. -/
def refineList? : List AxisName → List AxisName → Option (List AxisName)
  | [], [] => some []
  | old :: olds, new :: news =>
      match refineAxis? old new, refineList? olds news with
      | some out, some rest => some (out :: rest)
      | _, _ => none
  | _, _ => none

/-- No-ellipsis `refine_names` contract: rank must match, concrete names must be
    unique, and every pre-existing concrete name must be preserved. -/
def refineNames? (shape : List Nat) (current requested : List AxisName) : Option NamedShape :=
  if shape.length = current.length then
    if current.length = requested.length then
      if uniqueNamed current then
        if uniqueNamed requested then
          match refineList? current requested with
          | some out =>
              if uniqueNamed out then
                some { shape := shape, names := out }
              else
                none
          | none => none
        else
          none
      else
        none
    else
      none
  else
    none

theorem refine_existing_name_preserved (n : Nat) :
    refineAxis? (named n) (named n) = some (named n) := by
  simp [refineAxis?]

theorem refine_rename_rejected (old new : Nat) (h : old ≠ new) :
    refineAxis? (named old) (named new) = none := by
  unfold refineAxis?
  simp [h]

theorem refine_demotion_rejected (n : Nat) :
    refineAxis? (named n) anon = none := by
  rfl

theorem refine_duplicate_requested_rejected
    (shape : List Nat) (current requested : List AxisName)
    (hdup : uniqueNamed requested = false) :
    refineNames? shape current requested = none := by
  unfold refineNames?
  by_cases hlen₁ : shape.length = current.length <;> simp [hlen₁]
  by_cases hlen₂ : current.length = requested.length <;> simp [hlen₂]
  by_cases hcur : uniqueNamed current <;> simp [hcur, hdup]

theorem refine_duplicate_current_rejected
    (shape : List Nat) (current requested : List AxisName)
    (hdup : uniqueNamed current = false) :
    refineNames? shape current requested = none := by
  unfold refineNames?
  by_cases hlen₁ : shape.length = current.length <;> simp [hlen₁]
  by_cases hlen₂ : current.length = requested.length <;> simp [hlen₂, hdup]

theorem refine_shape_preserved
    (shape : List Nat) (current requested : List AxisName) (out : NamedShape)
    (h : refineNames? shape current requested = some out) :
    out.shape = shape := by
  unfold refineNames? at h
  by_cases hlen₁ : shape.length = current.length <;> simp [hlen₁] at h
  by_cases hlen₂ : current.length = requested.length <;> simp [hlen₂] at h
  by_cases hcur : uniqueNamed current <;> simp [hcur] at h
  by_cases hreq : uniqueNamed requested <;> simp [hreq] at h
  cases hlist : refineList? current requested with
  | none =>
      simp [hlist] at h
  | some outNames =>
      simp [hlist] at h
      by_cases hout : uniqueNamed outNames <;> simp [hout] at h
      cases h
      rfl

theorem refine_fill_anon_example :
    refineNames? [2, 3] [anon, anon] [named 0, named 1] =
      some { shape := [2, 3], names := [named 0, named 1] } := by
  rfl

theorem refine_preserve_existing_example :
    refineNames? [2, 3] [named 0, named 1] [named 0, named 1] =
      some { shape := [2, 3], names := [named 0, named 1] } := by
  rfl

theorem refine_duplicate_names_rejected :
    refineNames? [2, 3] [anon, anon] [named 0, named 0] = none := by
  rfl

/- ===================================================================== -/
/- 3. align_to                                                            -/
/- ===================================================================== -/

/-- Lookup the dimension carried by a concrete axis name. -/
def lookupDimByName (needle : Nat) : List Nat → List AxisName → Option Nat
  | [], _ => none
  | _, [] => none
  | _ :: dims, anon :: names => lookupDimByName needle dims names
  | d :: dims, named n :: names =>
      if n = needle then some d else lookupDimByName needle dims names

/-- The no-ellipsis `align_to` fragment must mention every input axis by a
    concrete name.  An anonymous input axis is rejected here; Python/PyTorch need
    an ellipsis to carry it through. -/
def allCurrentNamedMentioned : List AxisName → List AxisName → Bool
  | [], _ => true
  | anon :: _, _ => false
  | named n :: rest, target =>
      containsNamed n target && allCurrentNamedMentioned rest target

/-- Dimension contributed by a target axis.  Existing names preserve their
    original dimension; fresh concrete names and explicit anonymous targets
    insert singleton dimensions. -/
def targetDim (shape : List Nat) (current : List AxisName) : AxisName → Nat
  | anon => 1
  | named n =>
      match lookupDimByName n shape current with
      | some d => d
      | none => 1

def alignShape (shape : List Nat) (current target : List AxisName) : List Nat :=
  target.map (targetDim shape current)

/-- No-ellipsis `align_to`: reorder existing named dimensions, insert singleton
    axes for fresh target names / explicit anonymous targets, and reject missing
    existing names, duplicate concrete target names, duplicate input names, rank
    mismatches, and anonymous input axes. -/
def alignTo? (shape : List Nat) (current target : List AxisName) : Option NamedShape :=
  if shape.length = current.length then
    if uniqueNamed current then
      if uniqueNamed target then
        if allCurrentNamedMentioned current target then
          some { shape := alignShape shape current target, names := target }
        else
          none
      else
        none
    else
      none
  else
    none

theorem existing_name_dim_preserved
    (shape : List Nat) (current : List AxisName) (n d : Nat)
    (h : lookupDimByName n shape current = some d) :
    targetDim shape current (named n) = d := by
  simp [targetDim, h]

theorem fresh_name_inserts_singleton
    (shape : List Nat) (current : List AxisName) (n : Nat)
    (h : lookupDimByName n shape current = none) :
    targetDim shape current (named n) = 1 := by
  simp [targetDim, h]

theorem anon_target_inserts_singleton
    (shape : List Nat) (current : List AxisName) :
    targetDim shape current anon = 1 := by
  rfl

theorem align_names_preserved
    (shape : List Nat) (current target : List AxisName) (out : NamedShape)
    (h : alignTo? shape current target = some out) :
    out.names = target := by
  unfold alignTo? at h
  by_cases hlen : shape.length = current.length <;> simp [hlen] at h
  by_cases hcur : uniqueNamed current <;> simp [hcur] at h
  by_cases htarget : uniqueNamed target <;> simp [htarget] at h
  by_cases hall : allCurrentNamedMentioned current target <;> simp [hall] at h
  cases h
  rfl

theorem align_duplicate_target_rejected
    (shape : List Nat) (current target : List AxisName)
    (hdup : uniqueNamed target = false) :
    alignTo? shape current target = none := by
  unfold alignTo?
  by_cases hlen : shape.length = current.length <;> simp [hlen]
  by_cases hcur : uniqueNamed current <;> simp [hcur, hdup]

theorem align_duplicate_current_rejected
    (shape : List Nat) (current target : List AxisName)
    (hdup : uniqueNamed current = false) :
    alignTo? shape current target = none := by
  unfold alignTo?
  by_cases hlen : shape.length = current.length <;> simp [hlen, hdup]

theorem align_unnamed_input_rejected
    (d : Nat) (restShape : List Nat) (currentTail target : List AxisName) :
    alignTo? (d :: restShape) (anon :: currentTail) target = none := by
  unfold alignTo?
  by_cases hlen : (d :: restShape).length = (anon :: currentTail).length <;> simp [hlen]
  by_cases hcur : uniqueNamed (anon :: currentTail) <;> simp [hcur, allCurrentNamedMentioned]
  by_cases htarget : uniqueNamed target <;> simp [htarget, allCurrentNamedMentioned]

theorem align_permute_example :
    alignTo? [2, 3] [named 0, named 1] [named 1, named 0] =
      some { shape := [3, 2], names := [named 1, named 0] } := by
  rfl

theorem align_singleton_insert_example :
    alignTo? [2, 3] [named 0, named 1] [named 0, named 2, named 1] =
      some { shape := [2, 1, 3], names := [named 0, named 2, named 1] } := by
  rfl

theorem align_anon_target_insert_example :
    alignTo? [2, 3] [named 0, named 1] [named 0, anon, named 1] =
      some { shape := [2, 1, 3], names := [named 0, anon, named 1] } := by
  rfl

theorem align_missing_name_rejected :
    alignTo? [2, 3] [named 0, named 1] [named 0] = none := by
  rfl

theorem align_duplicate_names_rejected :
    alignTo? [2, 3] [named 0, named 1] [named 0, named 0, named 1] = none := by
  rfl

end NamedTensor
end TensorGuard
