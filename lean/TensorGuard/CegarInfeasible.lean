/-
TensorGuard known-unsoundness U2 — "SAFE-on-infeasible" — closed and
machine-checked in Lean 4 (Step 132).

`src/soundness_contract.py` recorded gap **U2**: the shape-CEGAR loop
(`src/shape_cegar.py`) could return `SAFE` when the *accumulated refined
predicates are jointly infeasible*. Infeasible accumulated predicates mean the
loop eliminated its counterexamples using mutually contradictory assumptions, so
the elimination is spurious and carries **no** information about whether the
program really has a shape bug. Reporting `SAFE` there is unsound.

This file models the terminal decision and proves the fix correct:

* `decideOld` (the buggy behaviour) returns `safe` regardless of feasibility;
* `decideNew` (the fix) abstains with `abstain` exactly when the accumulated
  predicates are infeasible, and only reports `safe` on the feasible branch.

Under the genuine soundness guarantee of the feasible branch
(`feasibleJustifiesSafety`: a feasible over-approximation that eliminated every
counterexample really is bug-free), we prove `decideNew` never reports `safe`
for a buggy program (`decideNew_safeSound`), while `decideOld` admits a reachable
counterexample (`decideOld_unsound`). `fix_abstains_on_infeasible` pins that the
behaviour changes exactly on the infeasible branch.

Pure Lean 4 core (no mathlib).
-/

namespace TensorGuard
namespace CegarU2

inductive Verdict
  | safe
  | bug
  | abstain
  deriving DecidableEq, Repr

/-- A `safe` verdict is sound only if the program is actually bug-free. -/
def safeSound (v : Verdict) (realBug : Bool) : Prop :=
  v = Verdict.safe → realBug = false

/-- The only sound knowledge available at this termination point: when the
    accumulated predicates are *feasible*, eliminating every counterexample is a
    genuine over-approximation, hence no real bug remains. When they are
    *infeasible*, the elimination is spurious and says nothing about `realBug`. -/
def feasibleJustifiesSafety (infeasible realBug : Bool) : Prop :=
  infeasible = false → realBug = false

/-- OLD (buggy) decision — report `safe` regardless of feasibility. -/
def decideOld (_infeasible : Bool) : Verdict := Verdict.safe

/-- NEW (fixed) decision — abstain when the accumulated
    predicates are infeasible, otherwise report `safe`. -/
def decideNew (infeasible : Bool) : Verdict :=
  if infeasible then Verdict.abstain else Verdict.safe

/-- **The fix is sound.** Under the feasible-branch guarantee, `decideNew` never
    reports `safe` for a buggy program — on the infeasible branch it abstains. -/
theorem decideNew_safeSound (infeasible realBug : Bool)
    (h : feasibleJustifiesSafety infeasible realBug) :
    safeSound (decideNew infeasible) realBug := by
  unfold safeSound decideNew feasibleJustifiesSafety at *
  intro hsafe
  cases infeasible with
  | true => simp at hsafe
  | false => exact h rfl

/-- **The old behaviour is unsound.** There is a reachable scenario — infeasible
    accumulated predicates together with a real bug — consistent with the
    analyzer's knowledge, in which `decideOld` reports `safe` for a buggy
    program. -/
theorem decideOld_unsound :
    ∃ infeasible realBug,
      feasibleJustifiesSafety infeasible realBug ∧
      decideOld infeasible = Verdict.safe ∧ realBug = true := by
  refine ⟨true, true, ?_, rfl, rfl⟩
  intro hc
  exact hc

/-- The fix changes the verdict **exactly** on the infeasible branch: it abstains
    (`abstain`) when infeasible and is unchanged (`safe`) when feasible, whereas
    the old decision is unconditionally `safe`. -/
theorem fix_abstains_on_infeasible : decideNew true = Verdict.abstain := rfl

theorem fix_keeps_safe_when_feasible : decideNew false = Verdict.safe := rfl

theorem old_always_safe (b : Bool) : decideOld b = Verdict.safe := rfl

end CegarU2
end TensorGuard
