/-
TensorGuard.Symexec.Lattice

The abstract-value lattice that underlies the torch-free symbolic-execution
engine (`src/symexec/values.py`, `src/symexec/symdim.py`).  We model the part of
the domain that every shape/axis bug check actually inspects: a *dimension*
abstraction `Dim` (a known concrete size, or ⊤ "unknown"), lifted pointwise to
abstract *shapes* `AShape`.

The concretization `γ` is given as a membership predicate `dimModels`/`Models`
relating a concrete value to the abstract value that over-approximates it.  This
is the exact object the soundness theorems quantify over: a report is sound iff
*every* concrete value modeled by the abstract operands makes the operator
ill-typed.

Pure Lean 4 core (no mathlib).
-/

namespace TensorGuard
namespace Symexec

/-- A dimension abstraction: a known concrete size, or ⊤ (unknown). -/
inductive Dim
  | known : Nat → Dim
  | unk   : Dim
  deriving DecidableEq, Repr

namespace Dim

/-- Concretization membership: `dimModels c d` iff the concrete size `c` is one
of the sizes the abstract dim `d` stands for.  `γ(unk)` is everything. -/
def dimModels (c : Nat) : Dim → Prop
  | .known n => c = n
  | .unk     => True

/-- The abstract order `a ⊑ b` ("a is at least as precise as b"). -/
def dle : Dim → Dim → Bool
  | _,        .unk     => true
  | .known n, .known m => n = m
  | .unk,     .known _ => false

/-- Join (least upper bound): equal knowns stay known, everything else ⊤. -/
def join : Dim → Dim → Dim
  | .known n, .known m => if n = m then .known n else .unk
  | _,        _        => .unk

/-- Widening coincides with join on this height-2 lattice (it already loses to
⊤ in one step), so it trivially terminates. -/
def widen : Dim → Dim → Dim := join

@[simp] theorem dle_unk_right (a : Dim) : dle a .unk = true := by
  cases a <;> rfl

/-- ⊤ concretizes to everything. -/
@[simp] theorem dimModels_unk (c : Nat) : dimModels c .unk := by
  trivial

/-- **γ is monotone w.r.t. ⊑.**  If `c` is modeled by `a` and `a ⊑ b`, then `c`
is modeled by `b`: concretization only grows as we go up the lattice. -/
theorem dimModels_mono {c : Nat} {a b : Dim}
    (hm : dimModels c a) (hle : dle a b = true) : dimModels c b := by
  cases a <;> cases b <;>
    simp_all [dimModels, dle]

/-- `join` is an upper bound of its left argument. -/
theorem dle_join_left (a b : Dim) : dle a (join a b) = true := by
  cases a with
  | unk => rfl
  | known n =>
    cases b with
    | unk => rfl
    | known m =>
      simp only [join]
      by_cases h : n = m
      · subst h; simp [dle]
      · simp [h, dle]

/-- `join` is an upper bound of its right argument. -/
theorem dle_join_right (a b : Dim) : dle b (join a b) = true := by
  cases a with
  | unk => cases b <;> rfl
  | known n =>
    cases b with
    | unk => rfl
    | known m =>
      simp only [join]
      by_cases h : n = m
      · subst h; simp [dle]
      · simp [h, dle]

/-- A concrete value modeled by either operand is modeled by their join
(soundness of control-flow merge at the dimension level). -/
theorem dimModels_join_left {c : Nat} {a b : Dim}
    (h : dimModels c a) : dimModels c (join a b) :=
  dimModels_mono h (dle_join_left a b)

theorem dimModels_join_right {c : Nat} {a b : Dim}
    (h : dimModels c b) : dimModels c (join a b) :=
  dimModels_mono h (dle_join_right a b)

end Dim

/-- An abstract shape is a list of dimension abstractions. -/
abbrev AShape := List Dim

/-- A concrete shape is a list of sizes. -/
abbrev CShape := List Nat

/-- `Models cs as` iff `cs` is a concretization of the abstract shape `as`:
same rank, pointwise modeled. -/
def Models : CShape → AShape → Prop
  | [],      []      => True
  | c :: cs, d :: ds => Dim.dimModels c d ∧ Models cs ds
  | _,       _       => False

@[simp] theorem Models_nil : Models [] [] := by trivial

@[simp] theorem Models_cons (c : Nat) (cs : CShape) (d : Dim) (ds : AShape) :
    Models (c :: cs) (d :: ds) ↔ (Dim.dimModels c d ∧ Models cs ds) := by
  simp [Models]

/-- A concretization has the same rank as the abstract shape it models. -/
theorem Models_length {cs : CShape} {as : AShape} (h : Models cs as) :
    cs.length = as.length := by
  induction cs generalizing as with
  | nil => cases as with
    | nil => rfl
    | cons _ _ => simp [Models] at h
  | cons c cs ih =>
    cases as with
    | nil => simp [Models] at h
    | cons d ds =>
      simp [Models] at h
      simpa using ih h.2

end Symexec
end TensorGuard
