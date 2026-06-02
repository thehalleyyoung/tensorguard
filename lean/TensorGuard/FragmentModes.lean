/-
TensorGuard known-unsoundness U1 — the verifiable-fragment boundary is
**mode-dependent** — re-audited and machine-checked in Lean 4 (Step 133).

`src/soundness_contract.py` records gap **U1**: in the permissive
`balanced`/`heuristic` modes, `verify_architecture` does *not* abstain on a
static *fragment violation* (a construct outside the verifiable fragment V_TG,
e.g. data-dependent control flow or a tensor→scalar `.item()` coercion). Such an
out-of-fragment module is reported `SAFE`, so a **real bug hidden by the
unmodeled construct can be missed**. The `sound` mode CLOSES this by abstaining
(`UNKNOWN`) on any fragment violation.

This file models the three-mode terminal decision and proves the exact U1 shape:

* the verified core is sound only on the *modeled* fragment: when a module is
  in-fragment and the core found no in-fragment bug, there is genuinely no bug
  (`coreSoundOnFragment`). Out of fragment, the core knows nothing;
* `decide sound` abstains on every fragment violation, so a `safe` verdict in
  `sound` mode implies no real bug — `sound_safeSound`;
* `decide balanced` / `decide heuristic` report `safe` on a fragment violation,
  so they admit a reachable scenario (out-of-fragment construct hiding a real
  bug) in which they report `safe` for a buggy program — `balanced_unsound`,
  `heuristic_unsound`;
* the three modes agree exactly on in-fragment modules and differ exactly on
  fragment violations — `modes_agree_in_fragment`, `modes_differ_iff_violation`.

Pure Lean 4 core (no mathlib).
-/

namespace TensorGuard
namespace FragmentU1

inductive Mode
  | sound
  | balanced
  | heuristic
  deriving DecidableEq, Repr

inductive Verdict
  | safe
  | bug
  | abstain
  deriving DecidableEq, Repr

/-- A `safe` verdict is sound only if the program is actually bug-free. -/
def safeSound (v : Verdict) (realBug : Bool) : Prop :=
  v = Verdict.safe → realBug = false

/-- The only sound knowledge the verified core provides: on an **in-fragment**
    module for which the core found **no** in-fragment bug, there is genuinely no
    real bug. Out of fragment the core is silent (`realBug` is unconstrained). -/
def coreSoundOnFragment (outOfFragment coreFoundBug realBug : Bool) : Prop :=
  outOfFragment = false → coreFoundBug = false → realBug = false

/-- The three-mode terminal decision.

    * If the verified core found an in-fragment bug, report `bug` (every mode).
    * Else if the module is out of fragment: `sound` mode abstains, while
      `balanced`/`heuristic` optimistically report `safe` (the U1 recall
      trade-off).
    * Else (in-fragment, no bug found) report `safe`. -/
def decide (mode : Mode) (outOfFragment coreFoundBug : Bool) : Verdict :=
  if coreFoundBug then
    Verdict.bug
  else if outOfFragment then
    match mode with
    | Mode.sound => Verdict.abstain
    | _          => Verdict.safe
  else
    Verdict.safe

/-- **`sound` mode is sound.** Under the core guarantee, a `safe` verdict in
    `sound` mode implies the program is bug-free — it abstains on every fragment
    violation, so `safe` can only come from the in-fragment, no-bug branch. -/
theorem sound_safeSound (outOfFragment coreFoundBug realBug : Bool)
    (h : coreSoundOnFragment outOfFragment coreFoundBug realBug) :
    safeSound (decide Mode.sound outOfFragment coreFoundBug) realBug := by
  intro hsafe
  cases hb : coreFoundBug with
  | true => simp [decide, hb] at hsafe
  | false =>
    cases hf : outOfFragment with
    | true => simp [decide, hb, hf] at hsafe
    | false => exact h hf hb

/-- **`balanced` mode is unsound.** There is a reachable scenario — an
    out-of-fragment construct hiding a real bug, with no in-fragment bug found —
    consistent with the core's knowledge, in which `balanced` mode reports `safe`
    for a buggy program. -/
theorem balanced_unsound :
    ∃ outOfFragment coreFoundBug realBug,
      coreSoundOnFragment outOfFragment coreFoundBug realBug ∧
      decide Mode.balanced outOfFragment coreFoundBug = Verdict.safe ∧
      realBug = true := by
  refine ⟨true, false, true, ?_, rfl, rfl⟩
  intro hc
  exact absurd hc (by decide)

/-- **`heuristic` mode is unsound**, by the same reachable scenario. -/
theorem heuristic_unsound :
    ∃ outOfFragment coreFoundBug realBug,
      coreSoundOnFragment outOfFragment coreFoundBug realBug ∧
      decide Mode.heuristic outOfFragment coreFoundBug = Verdict.safe ∧
      realBug = true := by
  refine ⟨true, false, true, ?_, rfl, rfl⟩
  intro hc
  exact absurd hc (by decide)

/-- On **in-fragment** modules the three modes are indistinguishable: U1 is
    entirely about the fragment boundary. -/
theorem modes_agree_in_fragment (mode mode' : Mode) (coreFoundBug : Bool) :
    decide mode false coreFoundBug = decide mode' false coreFoundBug := by
  cases coreFoundBug <;> rfl

/-- The modes differ **iff** there is a fragment violation with no in-fragment
    bug: precisely the U1 region. (Stated for `sound` vs `balanced`.) -/
theorem modes_differ_iff_violation (outOfFragment coreFoundBug : Bool) :
    (decide Mode.sound outOfFragment coreFoundBug
       ≠ decide Mode.balanced outOfFragment coreFoundBug)
    ↔ (outOfFragment = true ∧ coreFoundBug = false) := by
  cases outOfFragment <;> cases coreFoundBug <;> simp [decide]

end FragmentU1
end TensorGuard
