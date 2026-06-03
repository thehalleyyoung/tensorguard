/-
TensorGuard CEGAR predicate-record soundness (Step 241).

`src/shape_cegar.py` serializes discovered CEGAR predicates as versioned Python
records with fields

  schema, kind, tensor, axis, value, match_tensor, match_axis, divisor, provenance

and kind names matching `PredicateKind.name`.  This file mirrors that record
shape, proves that CEGAR history grows monotonically under append-only
refinement, and pins the terminal decision used when replayed serialized
contracts are jointly infeasible: abstain rather than report SAFE.

The proof is intentionally about the serialization boundary, not Python object
identity.  The Python tests round-trip real `ShapePredicate` objects through this
format and check the field/kind lists below against the live enum.

Pure Lean 4 core (no mathlib).
-/

namespace TensorGuard
namespace CegarSerialized

def schemaV1 : String := "tensorguard.shape_predicate.v1"

def pythonRecordKeys : List String :=
  ["schema", "kind", "tensor", "axis", "value",
   "match_tensor", "match_axis", "divisor", "provenance"]

inductive PredKind where
  | DIM_EQ
  | DIM_GT
  | DIM_GE
  | DIM_DIVISIBLE
  | DIM_MATCH
  | NDIM_EQ
  | SHAPE_EQ
  deriving DecidableEq, Repr

def predKindName : PredKind → String
  | PredKind.DIM_EQ => "DIM_EQ"
  | PredKind.DIM_GT => "DIM_GT"
  | PredKind.DIM_GE => "DIM_GE"
  | PredKind.DIM_DIVISIBLE => "DIM_DIVISIBLE"
  | PredKind.DIM_MATCH => "DIM_MATCH"
  | PredKind.NDIM_EQ => "NDIM_EQ"
  | PredKind.SHAPE_EQ => "SHAPE_EQ"

def pythonKindNames : List String :=
  ["DIM_EQ", "DIM_GT", "DIM_GE", "DIM_DIVISIBLE",
   "DIM_MATCH", "NDIM_EQ", "SHAPE_EQ"]

/-- JSON-compatible values carried by the Python record.  Exact-shape tuples are
    encoded by Python as lists and represented here by `list`. -/
inductive FieldValue where
  | null
  | int (value : Int)
  | string (value : String)
  | list (values : List FieldValue)

/-- Mirror of `shape_predicate_to_record` in `src/shape_cegar.py`. -/
structure SerializedPredicate where
  schema : String
  kind : PredKind
  tensor : String
  axis : Option Int
  value : FieldValue
  match_tensor : Option String
  match_axis : Option Int
  divisor : Option Int
  provenance : String

/-- Every predicate that was present before a refinement step is present after
    the step. -/
def Extends (old new : List SerializedPredicate) : Prop :=
  ∀ p, p ∈ old → p ∈ new

/-- The Lean record-key list is exactly the Python schema-v1 key order. -/
theorem python_record_keys_match_v1 :
    pythonRecordKeys =
      ["schema", "kind", "tensor", "axis", "value",
       "match_tensor", "match_axis", "divisor", "provenance"] := rfl

/-- The Lean kind list is exactly the live Python `PredicateKind.name` set. -/
theorem python_kind_names_cover_v1 :
    pythonKindNames =
      ["DIM_EQ", "DIM_GT", "DIM_GE", "DIM_DIVISIBLE",
       "DIM_MATCH", "NDIM_EQ", "SHAPE_EQ"] := rfl

/-- Append-only CEGAR refinement preserves all previously serialized
    predicates. -/
theorem serialized_append_preserves_membership
    (old fresh : List SerializedPredicate) :
    Extends old (old ++ fresh) := by
  intro p hp
  exact List.mem_append_left fresh hp

/-- Append-only CEGAR refinement cannot shrink the serialized predicate set. -/
theorem serialized_append_length_mono
    (old fresh : List SerializedPredicate) :
    old.length ≤ (old ++ fresh).length := by
  simp

/-- One CEGAR history transition is a prefix-preserving append step. -/
theorem serialized_history_step_prefix
    (previous added : List SerializedPredicate) :
    Extends previous (previous ++ added) :=
  serialized_append_preserves_membership previous added

inductive Verdict where
  | safe
  | bug
  | abstain
  deriving DecidableEq, Repr

/-- A serialized contract replay can report safe only if the replayed predicate
    set is feasible; infeasible replay abstains. -/
def decideSerialized (feasible : Bool) : Verdict :=
  if feasible then Verdict.safe else Verdict.abstain

def safeSound (v : Verdict) (realBug : Bool) : Prop :=
  v = Verdict.safe → realBug = false

def feasibleSerializedJustifiesSafety (feasible realBug : Bool) : Prop :=
  feasible = true → realBug = false

/-- Replaying a jointly infeasible serialized contract abstains, matching the
    Python `CEGARStatus.INFEASIBLE_REFINEMENT` branch. -/
theorem serialized_infeasible_abstains :
    decideSerialized false = Verdict.abstain := rfl

theorem serialized_feasible_safe :
    decideSerialized true = Verdict.safe := rfl

/-- If the feasible branch is sound, the serialized-record decision is sound:
    infeasible records never report SAFE, and feasible records rely exactly on
    the feasible-branch guarantee. -/
theorem decideSerialized_safeSound (feasible realBug : Bool)
    (h : feasibleSerializedJustifiesSafety feasible realBug) :
    safeSound (decideSerialized feasible) realBug := by
  unfold safeSound decideSerialized feasibleSerializedJustifiesSafety at *
  intro hsafe
  cases feasible with
  | false => simp at hsafe
  | true => exact h rfl

theorem infeasible_serialized_safeSound_any_bug (realBug : Bool) :
    safeSound (decideSerialized false) realBug := by
  unfold safeSound decideSerialized
  intro hsafe
  simp at hsafe

end CegarSerialized
end TensorGuard
