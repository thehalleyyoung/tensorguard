/-
TensorGuard shape-CEGAR termination & iteration bound, machine-checked in Lean 4
(Steps 129–130).

The Python theory lives in `src/cegar_convergence_theory.py` and is measured by
`reproducibility/cegar_convergence.py`, which empirically confirms, per model,

    iterations ≤ 1 + |discovered_predicates|              (the *tight* bound)

for the shape-predicate CEGAR loop. The argument is Houdini-style monotone
predicate accumulation (Flanagan & Leino 2001): the loop maintains a set of
discovered shape predicates that only ever grows, each *productive* refinement
iteration discovers at least one new predicate drawn from a finite universe, and
one final iteration confirms SAFE or exhibits a real bug. Hence the loop
terminates, in at most `|P_final \ P_seed|` refinement iterations.

This file mechanizes that argument abstractly and soundly. A run is modelled by
its list of per-iteration *gains* — `gains[i] ≥ 1` is the number of new
predicates discovered at refinement iteration `i` (productivity). Then

* `discovered gains = Σ gains`     — total predicates learned,
* `totalIters gains = gains.length + 1` — refinement iterations plus the
  terminal confirming iteration,

and we prove the harness inequality `totalIters ≤ 1 + discovered` together with
termination inside any finite predicate universe `U` (`discovered ≤ U` implies a
finite, `U`-bounded run).

Pure Lean 4 core (no mathlib).
-/

namespace TensorGuard
namespace Cegar

/-- Sum of a list of naturals (the total number of predicates discovered across
    all refinement iterations). Defined locally to avoid any mathlib dependency. -/
def lsum : List Nat → Nat
  | []      => 0
  | x :: xs => x + lsum xs

/-- **Productivity ⇒ progress.** If every refinement iteration discovers at least
    one new predicate, the number of iterations is bounded by the number of
    predicates discovered. This is the heart of the Houdini termination argument:
    a strictly productive loop over a finite predicate set must stop. -/
theorem length_le_lsum (l : List Nat) (h : ∀ g ∈ l, 1 ≤ g) :
    l.length ≤ lsum l := by
  induction l with
  | nil => simp [lsum]
  | cons x xs ih =>
    have hx : 1 ≤ x := h x (List.mem_cons_self x xs)
    have hxs : ∀ g ∈ xs, 1 ≤ g := fun g hg => h g (List.mem_cons_of_mem x hg)
    have hlen := ih hxs
    simp only [List.length_cons, lsum]
    omega

/-- Total predicates discovered over a run (Σ of per-iteration gains). -/
def discovered (gains : List Nat) : Nat := lsum gains

/-- Total CEGAR iterations: one per refinement plus a terminal confirming
    iteration. -/
def totalIters (gains : List Nat) : Nat := gains.length + 1

/-- **Step 130 — the tight iteration bound.** For any productive run,

        iterations ≤ 1 + |discovered predicates|,

exactly the inequality `reproducibility/cegar_convergence.py` checks per model. -/
theorem cegar_iter_bound (gains : List Nat) (h : ∀ g ∈ gains, 1 ≤ g) :
    totalIters gains ≤ 1 + discovered gains := by
  have hlen := length_le_lsum gains h
  simp only [totalIters, discovered]
  omega

/-- **Step 129 — termination.** Inside a finite predicate universe of size `U`
    (no predicate is discovered twice, so the cumulative discoveries are
    `≤ U`), a productive run performs at most `U` refinement iterations and at
    most `U + 1` total iterations — the loop cannot run forever. -/
theorem cegar_terminates (gains : List Nat) (U : Nat)
    (h : ∀ g ∈ gains, 1 ≤ g) (hU : discovered gains ≤ U) :
    gains.length ≤ U ∧ totalIters gains ≤ U + 1 := by
  have hlen := length_le_lsum gains h
  simp only [discovered] at hU
  simp only [totalIters]
  omega

/-- The tight bound dominates the naive predicate-universe bound: whenever the
    discovered set is a strict subset of the universe (`discovered < U`), the
    tight total-iteration bound `1 + discovered` is strictly below the naive
    `U + 1`. Quantifies *why* the measured improvement factor is large. -/
theorem tight_below_naive (gains : List Nat) (U : Nat)
    (hU : discovered gains < U) :
    1 + discovered gains < U + 1 := by
  omega

end Cegar
end TensorGuard
