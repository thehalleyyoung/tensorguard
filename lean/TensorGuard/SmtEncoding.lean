/-
TensorGuard SMT-encoding faithfulness, machine-checked in Lean 4 (Step 135).

The verifier discharges every device / phase / gradient check by handing Z3 an
**enumeration-sort equality** constraint (`model_checker.py`:
`encode_device_constraint` returns `dev_a == dev_b`; `encode_phase_constraint`
compares against `PHASE_TRAIN`/`PHASE_EVAL`; `encode_gradient_constraint`
returns `grad_out == BoolVal(requires_grad)`).  A *mismatch* is reported exactly
when, after pinning the two endpoints to concrete sort elements, that same-value
constraint is **unsatisfiable**.

This file proves the *faithfulness* of that encoding: for any decidable sort,
the pinned equality formula is satisfiable **iff** the two pinned values are
equal — so the SMT verdict (UNSAT) coincides exactly with the abstract algebra's
`*Bug` predicate (`DeviceDtype.lean`).  The faithfulness is therefore independent
of any particular concrete sort and lifts uniformly to the device sort, the
phase sort and the gradient `Bool`.

The companion test `tests/test_smt_encoding_faithful.py` runs the **real Z3
encoder** on every concrete device / phase / gradient endpoint pair and asserts
the live SAT/UNSAT verdict equals the Lean-modeled prediction — so the proved
faithfulness holds for the actual solver calls the verifier makes.

Pure Lean 4 core (no mathlib).
-/

import TensorGuard.BroadcastChain
import TensorGuard.ChunkSplit
import TensorGuard.DeviceDtype
import TensorGuard.DtypePromoteChain
import TensorGuard.ReshapeInfer

namespace TensorGuard
namespace SmtEncoding

open TensorGuard.DevDtype

/- ===================================================================== -/
/- 1. Generic enum-equality encoding faithfulness                        -/
/- ===================================================================== -/

/-- The SMT encoding TensorGuard emits for a "same value" check: two fresh
    sort variables `a b`, pinned to the concrete endpoints `ca cb`, are
    constrained equal.  `Sat ca cb` says this conjunction has a model. -/
def Sat {α : Type} (ca cb : α) : Prop :=
  ∃ a b : α, a = ca ∧ b = cb ∧ a = b

/-- **Faithfulness (SAT direction).** The pinned same-value constraint is
    satisfiable iff the two endpoints are actually equal. -/
theorem sat_iff_eq {α : Type} (ca cb : α) : Sat ca cb ↔ ca = cb := by
  constructor
  · rintro ⟨a, b, rfl, rfl, hab⟩; exact hab
  · intro h; exact ⟨ca, cb, rfl, rfl, h⟩

/-- **Faithfulness (UNSAT direction).** The constraint is *unsatisfiable* — i.e.
    the verifier reports a mismatch — iff the endpoints genuinely differ. -/
theorem unsat_iff_ne {α : Type} (ca cb : α) : ¬ Sat ca cb ↔ ca ≠ cb := by
  rw [sat_iff_eq]

/-- A reported mismatch (UNSAT) is *sound*: it implies a genuine inequality. -/
theorem unsat_sound {α : Type} (ca cb : α) (h : ¬ Sat ca cb) : ca ≠ cb :=
  (unsat_iff_ne ca cb).1 h

/-- A satisfiable encoding never fires a mismatch on equal endpoints (no false
    positive at the SMT layer). -/
theorem eq_is_sat {α : Type} (c : α) : Sat c c :=
  (sat_iff_eq c c).2 rfl

/- ===================================================================== -/
/- 2. Device sort: SMT verdict coincides with `devBug`                   -/
/- ===================================================================== -/

/-- For two **known** devices the real solver's UNSAT verdict on the
    `encode_device_constraint` formula is exactly the abstract `devBug`. -/
theorem device_smt_matches_devBug (ca cb : Dev)
    (ha : ca ≠ Dev.unknown) (hb : cb ≠ Dev.unknown) :
    (¬ Sat ca cb) ↔ devBug ca cb = true := by
  rw [unsat_iff_ne]
  constructor
  · intro hne
    have hu : ¬ (ca = Dev.unknown ∨ cb = Dev.unknown) := by
      rintro (h | h); exact ha h; exact hb h
    simp [devBug, hu, hne]
  · intro hbug
    have hu : ¬ (ca = Dev.unknown ∨ cb = Dev.unknown) := by
      rintro (h | h); exact ha h; exact hb h
    have : decide (ca ≠ cb) = true := by simpa [devBug, hu] using hbug
    simpa using this

/- ===================================================================== -/
/- 3. Phase sort: equality encoding is faithful                          -/
/- ===================================================================== -/

/-- The phase encoder compares a phase variable against a concrete phase const;
    pinning both endpoints, the constraint is UNSAT iff the phases differ. -/
theorem phase_smt_unsat_iff_ne (pa pb : Phase) :
    (¬ Sat pa pb) ↔ pa ≠ pb :=
  unsat_iff_ne pa pb

/-- Same phase ⇒ the phase-equality constraint is satisfiable (no spurious
    phase inconsistency). -/
theorem phase_same_sat (p : Phase) : Sat p p := eq_is_sat p

/- ===================================================================== -/
/- 4. Gradient Bool: `grad_out == BoolVal(requires_grad)` is faithful    -/
/- ===================================================================== -/

/-- `encode_gradient_constraint` pins a gradient boolean to a required value;
    the resulting equality is UNSAT iff the demanded status disagrees with the
    encoded one. -/
theorem grad_smt_unsat_iff_ne (gset greq : Bool) :
    (¬ Sat gset greq) ↔ gset ≠ greq :=
  unsat_iff_ne gset greq

theorem grad_consistent_sat (g : Bool) : Sat g g := eq_is_sat g

/- ===================================================================== -/
/- 5. Dtype sort: matmul's exact-dtype encoding coincides with           -/
/-    `dtMatmulBug` (Step 146)                                           -/
/- ===================================================================== -/

/-- The dtype-equality encoder (`mm`/`bmm`/`matmul` require identical operand
    dtypes) pins two dtype variables to concrete endpoints and constrains them
    equal; it is UNSAT iff the dtypes differ. -/
theorem dtype_smt_unsat_iff_ne (da db : Dt) :
    (¬ Sat da db) ↔ da ≠ db :=
  unsat_iff_ne da db

/-- **Faithfulness for the matmul dtype check.** For two **known** dtypes the
    real solver's UNSAT verdict on the dtype-equality formula is exactly the
    abstract `dtMatmulBug` — the SMT layer flags a matmul dtype error iff the
    abstract rule does. -/
theorem dtype_smt_matches_dtMatmulBug (da db : Dt)
    (ha : da ≠ Dt.unknown) (hb : db ≠ Dt.unknown) :
    (¬ Sat da db) ↔ dtMatmulBug da db = true := by
  rw [unsat_iff_ne]
  constructor
  · intro hne
    have hu : ¬ (da = Dt.unknown ∨ db = Dt.unknown) := by
      rintro (h | h); exact ha h; exact hb h
    simp [dtMatmulBug, hu, hne]
  · intro hbug
    have hu : ¬ (da = Dt.unknown ∨ db = Dt.unknown) := by
      rintro (h | h); exact ha h; exact hb h
    have : decide (da ≠ db) = true := by simpa [dtMatmulBug, hu] using hbug
    simpa using this

/-- Equal dtypes ⇒ the dtype-equality constraint is satisfiable (no spurious
    matmul dtype error). -/
theorem dtype_same_sat (d : Dt) : Sat d d := eq_is_sat d

/- ===================================================================== -/
/- 6. Broadcast compatibility SMT formula (Step 242)                    -/
/- ===================================================================== -/

/-- One dimension of the broadcast encoding handed to Z3: with the two concrete
    endpoints pinned, a witness output dimension must satisfy exactly one of the
    PyTorch/NumPy compatibility branches (`a == 1`, `b == 1`, or `a == b`). -/
def BroadcastDimFormula (a b out : Nat) : Prop :=
  (a = 1 ∧ out = b) ∨ (b = 1 ∧ out = a) ∨ (a = b ∧ out = a)

/-- Satisfiability of the one-dimensional broadcast SMT formula. -/
def BroadcastSat (a b : Nat) : Prop :=
  ∃ out : Nat, BroadcastDimFormula a b out

/-- **Broadcast SMT faithfulness (SAT).** The disjunctive SMT formula is
    satisfiable iff the Lean/PyTorch broadcast transfer returns some output
    dimension. -/
theorem broadcast_smt_sat_iff_bcDim_some (a b : Nat) :
    BroadcastSat a b ↔ ∃ out, BroadcastChain.bcDim a b = some out := by
  unfold BroadcastSat BroadcastDimFormula BroadcastChain.bcDim
  by_cases ha : a = 1
  · subst ha
    simp
  · by_cases hb : b = 1
    · subst hb
      simp [ha]
    · by_cases hab : a = b
      · subst hab
        simp [ha]
      · simp [ha, hb, hab]

/-- **Broadcast SMT faithfulness (UNSAT).** The formula is unsatisfiable exactly
    when the broadcast rule returns `none`, i.e. the endpoints are genuinely
    incompatible. -/
theorem broadcast_smt_unsat_iff_bcDim_none (a b : Nat) :
    ¬ BroadcastSat a b ↔ BroadcastChain.bcDim a b = none := by
  unfold BroadcastSat BroadcastDimFormula BroadcastChain.bcDim
  by_cases ha : a = 1
  · subst ha
    simp
  · by_cases hb : b = 1
    · subst hb
      simp [ha]
    · by_cases hab : a = b
      · subst hab
        simp [ha]
      · simp [ha, hb, hab]

/-- Corollary tied to the existing broadcast theorem: UNSAT iff both dimensions
    are non-unit and unequal. -/
theorem broadcast_smt_unsat_iff_incompatible (a b : Nat) :
    ¬ BroadcastSat a b ↔ (a ≠ 1 ∧ b ≠ 1 ∧ a ≠ b) := by
  rw [broadcast_smt_unsat_iff_bcDim_none, BroadcastChain.bcDim_none_iff]

/- ===================================================================== -/
/- 7. Reshape divisibility SMT formula (Step 242)                        -/
/- ===================================================================== -/

/-- SMT model for `reshape(..., -1)` divisibility: the known output product is
    positive and there exists an inferred dimension whose product reconstructs the
    input element count. -/
def DivisibilitySat (total : Nat) (known : List Nat) : Prop :=
  0 < ReshapeInfer.prod known ∧
    ∃ inferred : Nat, ReshapeInfer.prod known * inferred = total

/-- **Divisibility SMT faithfulness.** The existential product constraint is
    satisfiable iff the Lean reshape guard admits the target spec. -/
theorem divisibility_smt_sat_iff_reshapeValid (total : Nat) (known : List Nat) :
    DivisibilitySat total known ↔
      ReshapeInfer.reshapeValid total known = true := by
  rw [ReshapeInfer.reshapeValid_iff]
  constructor
  · rintro ⟨hpos, inferred, hmul⟩
    exact ⟨hpos, ⟨inferred, hmul.symm⟩⟩
  · rintro ⟨hpos, inferred, hmul⟩
    exact ⟨hpos, ⟨inferred, hmul.symm⟩⟩

/-- The reshape divisibility formula is UNSAT exactly when the Lean guard rejects
    the spec. -/
theorem divisibility_smt_unsat_iff_invalid (total : Nat) (known : List Nat) :
    ¬ DivisibilitySat total known ↔
      ReshapeInfer.reshapeValid total known = false := by
  rw [divisibility_smt_sat_iff_reshapeValid]
  cases ReshapeInfer.reshapeValid total known <;> simp

/- ===================================================================== -/
/- 8. Split/chunk partition SMT formula (Step 242)                       -/
/- ===================================================================== -/

/-- SMT model for concrete split/chunk reconstruction: the pinned input axis must
    equal the sum of emitted section sizes. -/
def PartitionSat (axisSize : Nat) (sections : List Nat) : Prop :=
  ∃ axis recon : Nat,
    axis = axisSize ∧ recon = ChunkSplit.sum sections ∧ axis = recon

/-- **Partition SMT faithfulness.** The section-sum equality is satisfiable iff
    the sections reconstruct the original axis. -/
theorem partition_smt_sat_iff_sum_eq (axisSize : Nat) (sections : List Nat) :
    PartitionSat axisSize sections ↔ ChunkSplit.sum sections = axisSize := by
  unfold PartitionSat
  constructor
  · rintro ⟨axis, recon, haxis, hrecon, heq⟩
    rw [← haxis, ← hrecon]
    exact heq.symm
  · intro h
    exact ⟨axisSize, ChunkSplit.sum sections, rfl, rfl, h.symm⟩

/-- The SMT section-sum formula matches the Lean `splitSectionsValid` guard for
    any axis-factored shape. -/
theorem partition_smt_matches_splitSectionsValid
    (s : ChunkSplit.AxisShape) (sections : List Nat) :
    PartitionSat s.axisSize sections ↔
      ChunkSplit.splitSectionsValid s sections = true := by
  rw [partition_smt_sat_iff_sum_eq, ChunkSplit.splitValid_iff]

/-- UNSAT is exactly a section-sum mismatch. -/
theorem partition_smt_unsat_iff_mismatch
    (s : ChunkSplit.AxisShape) (sections : List Nat) :
    ¬ PartitionSat s.axisSize sections ↔
      ChunkSplit.splitSectionsValid s sections = false := by
  rw [partition_smt_matches_splitSectionsValid]
  cases ChunkSplit.splitSectionsValid s sections <;> simp

/- ===================================================================== -/
/- 9. Dtype-promotion SMT formula (Step 242)                             -/
/- ===================================================================== -/

/-- SMT model for a pairwise dtype-promotion transfer: input dtypes and the
    claimed output dtype are pinned, while the formula enforces equality with
    the finite promotion table. -/
def DtypePromoteSat (a b out : Dt) : Prop :=
  ∃ da db dout : Dt,
    da = a ∧ db = b ∧ dout = out ∧ dout = dtPromote da db

/-- **Dtype-promotion SMT faithfulness (pair).** The finite-table equality
    formula is satisfiable iff the claimed output is exactly `dtPromote a b`. -/
theorem dtype_promote_smt_sat_iff (a b out : Dt) :
    DtypePromoteSat a b out ↔ out = dtPromote a b := by
  unfold DtypePromoteSat
  constructor
  · rintro ⟨da, db, dout, rfl, rfl, rfl, h⟩
    exact h
  · intro h
    exact ⟨a, b, out, rfl, rfl, rfl, h⟩

/-- A pinned dtype-promotion formula is UNSAT exactly when the claimed output
    differs from the promotion table. -/
theorem dtype_promote_smt_unsat_iff_mismatch (a b out : Dt) :
    ¬ DtypePromoteSat a b out ↔ out ≠ dtPromote a b := by
  rw [dtype_promote_smt_sat_iff]

/-- SMT model for a dtype-promotion chain: the pinned final dtype must equal the
    `promoteRun` fold used by the abstract transfer. -/
def DtypePromoteChainSat (acc : Dt) (xs : List Dt) (out : Dt) : Prop :=
  ∃ promoted : Dt,
    promoted = DtypePromoteChain.promoteRun acc xs ∧ out = promoted

/-- **Dtype-promotion SMT faithfulness (chain).** The chained equality formula is
    satisfiable iff the claimed final dtype is exactly the Lean promotion fold. -/
theorem dtype_promote_chain_smt_sat_iff (acc : Dt) (xs : List Dt) (out : Dt) :
    DtypePromoteChainSat acc xs out ↔
      out = DtypePromoteChain.promoteRun acc xs := by
  unfold DtypePromoteChainSat
  constructor
  · rintro ⟨promoted, hpromoted, hout⟩
    rw [hout, hpromoted]
  · intro h
    exact ⟨DtypePromoteChain.promoteRun acc xs, rfl, h⟩

/-- A chained dtype-promotion formula is UNSAT exactly when the claimed final
    dtype differs from the promotion fold. -/
theorem dtype_promote_chain_smt_unsat_iff_mismatch
    (acc : Dt) (xs : List Dt) (out : Dt) :
    ¬ DtypePromoteChainSat acc xs out ↔
      out ≠ DtypePromoteChain.promoteRun acc xs := by
  rw [dtype_promote_chain_smt_sat_iff]

end SmtEncoding
end TensorGuard
