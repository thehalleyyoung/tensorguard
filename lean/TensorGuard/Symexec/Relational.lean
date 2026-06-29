/-
TensorGuard.Symexec.Relational

Machine-checked soundness of the **relational constraint layer** that the
symbolic executor uses for path conditions, branch refinement, feasibility-gated
reporting and control-flow joins (`src/symexec/smt_bridge.py` +
`src/symexec/relational.py`).

This sits directly on top of the affine-dimension semantics proved in
`TensorGuard.Symexec.Affine`: a *constraint* is a relation (`==`, `!=`, `<`,
`<=`, `>`, `>=`) between two affine forms, or a divisibility predicate
(`k ∣ e` / `¬ k ∣ e`).  A `RelationalDomain` is a conjunction (list) of such
constraints; its concretization `γ(D)` is the set of integer assignments
satisfying **all** of them.

The engine's whole relational machinery rests on three soundness obligations,
each proved here against that concrete semantics:

  1. **`negate` is an *exact* logical complement** (`negate_sound`).
     This is the linchpin: `entails`, branch-condition refinement, and
     feasibility/unsat refutation all reduce a question to
     "`facts ∧ ¬c` is unsatisfiable".  That reduction is sound **iff** `¬c` is
     the true complement of `c` on every assignment — which is what
     `negate_sound` certifies, operator by operator (including the divisibility
     pair).

  2. **The entailment reduction is sound** (`entails_of_unsat`).
     If `facts ∧ negate c` has no model, then every model of `facts` satisfies
     `c`.  This is exactly the inference `RelationalDomain.entails` performs once
     Z3 reports `unsat` (Z3 is the trusted oracle for the *unsat* fact itself;
     this theorem certifies the surrounding logic that turns that fact into an
     entailment).

  3. **The lattice operations are sound w.r.t. `γ`** — `meet` is exactly
     conjunction (`meet_sound`), `join` over-approximates each branch
     (`join_sound_left` / `join_sound_right`, so it over-approximates the union
     of the two incoming path-states), and `widen` only ever drops constraints
     so it can only *grow* `γ` (`widen_sound`).  Over-approximation of `γ` is the
     soundness condition for a merge/fixpoint step.

Together with `Affine`, this extends the machine-checked frontier across the
full relational/SMT-feasibility substrate, not just the op-level forced-failure
detectors.

Pure Lean 4 core (no mathlib).
-/
import TensorGuard.Symexec.Affine

namespace TensorGuard
namespace Symexec
namespace Relational

open TensorGuard.Symexec.Affine

/-- The six relational operators between two affine forms. -/
inductive RelOp
  | eq | ne | lt | le | gt | ge
  deriving DecidableEq, Repr

/-- Concrete meaning of a relational operator on two integer values. -/
def relSat : RelOp → Int → Int → Prop
  | .eq, x, y => x = y
  | .ne, x, y => x ≠ y
  | .lt, x, y => x < y
  | .le, x, y => x ≤ y
  | .gt, x, y => x > y
  | .ge, x, y => x ≥ y

/-- Logical-complement operator, matching `smt_bridge._OP_NEGATE`
(`==`↔`!=`, `<`↔`>=`, `<=`↔`>`, `>`↔`<=`, `>=`↔`<`). -/
def relNeg : RelOp → RelOp
  | .eq => .ne
  | .ne => .eq
  | .lt => .ge
  | .le => .gt
  | .gt => .le
  | .ge => .lt

/-- **Relational negation is exact**: the complemented operator holds iff the
original fails, on every pair of integers. -/
theorem relNeg_sound (op : RelOp) (x y : Int) :
    relSat (relNeg op) x y ↔ ¬ relSat op x y := by
  cases op <;> simp [relSat, relNeg] <;> omega

/-- A single dimension constraint over affine forms.  `dvd`/`ndvd` carry the
(constant, non-zero) modulus exactly as `smt_bridge.DimConstraint` does. -/
inductive Constraint
  | rel  (lhs : Affine) (op : RelOp) (rhs : Affine)
  | dvd  (e : Affine) (k : Int)
  | ndvd (e : Affine) (k : Int)

/-- Concrete satisfaction of a constraint under an assignment. -/
def sat (env : Env) : Constraint → Prop
  | .rel lhs op rhs => relSat op (eval lhs env) (eval rhs env)
  | .dvd e k        => k ∣ eval e env
  | .ndvd e k       => ¬ (k ∣ eval e env)

/-- The logical negation of a constraint (mirrors `smt_bridge.negate`). -/
def negate : Constraint → Constraint
  | .rel lhs op rhs => .rel lhs (relNeg op) rhs
  | .dvd e k        => .ndvd e k
  | .ndvd e k       => .dvd e k

/-- **`negate` is an exact logical complement** — the soundness linchpin for
`entails`, branch refinement and feasibility refutation. -/
theorem negate_sound (env : Env) (c : Constraint) :
    sat env (negate c) ↔ ¬ sat env c := by
  cases c with
  | rel lhs op rhs => simpa [sat, negate] using relNeg_sound op (eval lhs env) (eval rhs env)
  | dvd e k        => simp [sat, negate]
  | ndvd e k       => simp only [sat, negate]; exact (Classical.not_not).symm

/-- Double negation is the identity (mirrors `negate(negate c) = c`). -/
theorem negate_negate (env : Env) (c : Constraint) :
    sat env (negate (negate c)) ↔ sat env c := by
  rw [negate_sound, negate_sound, Classical.not_not]

-- --------------------------------------------------------------------------- --
-- The relational domain: a conjunction of constraints with `γ` = its models.  --
-- --------------------------------------------------------------------------- --

/-- A `RelationalDomain` value: a finite conjunction of constraints. -/
abbrev Domain := List Constraint

/-- `γ` membership: an assignment is in the concretization of a domain iff it
satisfies every constraint. -/
def models (env : Env) (D : Domain) : Prop :=
  ∀ c ∈ D, sat env c

/-- The greatest lower bound: the conjunction of both constraint sets
(`RelationalDomain.meet`). -/
def meet (A B : Domain) : Domain := A ++ B

/-- Semantic entailment: every model of the domain satisfies `c`
(the property `RelationalDomain.entails` decides). -/
def Entails (D : Domain) (c : Constraint) : Prop :=
  ∀ env, models env D → sat env c

/-- A domain is unsatisfiable when no assignment models it. -/
def Unsat (D : Domain) : Prop := ∀ env, ¬ models env D

-- --------------------------------------------------------------------------- --
-- Soundness of the lattice operations.                                        --
-- --------------------------------------------------------------------------- --

/-- **`meet` is exactly conjunction.** `γ(meet A B) = γ(A) ∩ γ(B)`. -/
theorem meet_sound (env : Env) (A B : Domain) :
    models env (meet A B) ↔ models env A ∧ models env B := by
  constructor
  · intro h
    exact ⟨fun c hc => h c (List.mem_append_left B hc),
           fun c hc => h c (List.mem_append_right A hc)⟩
  · intro ⟨hA, hB⟩ c hc
    rcases List.mem_append.1 hc with h | h
    · exact hA c h
    · exact hB c h

/-- **The entailment reduction is sound.** If `facts ∧ ¬c` is unsatisfiable then
`facts` entails `c` — the exact inference `entails` draws from a Z3 `unsat`
verdict (Z3 supplies the `Unsat` premise; this certifies the surrounding logic).

The proof uses only `negate_sound`: were some model of `facts` to violate `c`,
it would satisfy `negate c`, hence model `negate c :: facts`, contradicting
`Unsat`. -/
theorem entails_of_unsat {D : Domain} {c : Constraint}
    (h : Unsat (negate c :: D)) : Entails D c := by
  intro env hD
  refine Classical.byContradiction (fun hc => ?_)
  have hneg : sat env (negate c) := (negate_sound env c).2 hc
  exact h env (by
    intro c' hc'
    rcases List.mem_cons.1 hc' with rfl | hmem
    · exact hneg
    · exact hD c' hmem)

/-- Conversely, a sound `entails` verdict really does make `facts ∧ ¬c`
unsatisfiable — so the reduction is an equivalence, not just one direction. -/
theorem unsat_of_entails {D : Domain} {c : Constraint}
    (h : Entails D c) : Unsat (negate c :: D) := by
  intro env hmod
  have hD : models env D := fun c' hc' => hmod c' (List.mem_cons_of_mem _ hc')
  have hneg : sat env (negate c) := hmod (negate c) (List.mem_cons_self _ _)
  exact (negate_sound env c).1 hneg (h env hD)

/-- **`join` over-approximates the left branch.** The engine keeps a candidate
constraint only when *both* operands entail it; any such kept set is therefore
modeled by every assignment that models `A` — so `γ(A) ⊆ γ(join A B)`. -/
theorem join_sound_left {A kept : Domain} (env : Env)
    (hk : ∀ c ∈ kept, Entails A c) (hA : models env A) :
    models env kept := by
  intro c hc
  exact hk c hc env hA

/-- **`join` over-approximates the right branch** (symmetric). Together with
`join_sound_left`, `γ(A) ∪ γ(B) ⊆ γ(join A B)`: the merge of two paths keeps
only facts true on both, so it loses no concrete state of either incoming
branch. -/
theorem join_sound_right {B kept : Domain} (env : Env)
    (hk : ∀ c ∈ kept, Entails B c) (hB : models env B) :
    models env kept := by
  intro c hc
  exact hk c hc env hB

/-- **`widen` is sound.** Widening keeps only a subset of `self`'s constraints,
so it can only enlarge `γ`: every model of `self` still models the widened
domain.  (Dropping constraints is the over-approximation that guarantees
fixpoint termination without losing soundness.) -/
theorem widen_sound {self kept : Domain} (env : Env)
    (hsub : ∀ c ∈ kept, c ∈ self) (hself : models env self) :
    models env kept := by
  intro c hc
  exact hself c (hsub c hc)

/-- Adding the dimension well-formedness floor (each variable `≥ floor`, encoded
in `SymDimSolver._wellformed`) is just conjoining more constraints, so it can
only *shrink* `γ` — it never makes a satisfiable report spuriously infeasible in
a way that would suppress a true bug.  (Modeled as: extending the domain only
removes models.) -/
theorem extend_only_shrinks (env : Env) (D extra : Domain)
    (h : models env (D ++ extra)) : models env D :=
  ((meet_sound env D extra).1 h).1

end Relational
end Symexec
end TensorGuard
