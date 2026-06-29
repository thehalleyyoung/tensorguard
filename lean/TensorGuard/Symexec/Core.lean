/-
TensorGuard.Symexec.Core

The core *refutation-soundness* lemmas shared by every shape/axis bug check in
the symbolic-execution engine.  The engine's defining property is **no false
positive**: a bug is reported only when *every* concrete value modeled by the
abstract operands makes the operation genuinely ill-typed, and a *certified
counterexample* (a concrete witness, cf. `force_counterexample` in
`src/symexec/concretize.py`) exists.

Most checks reduce to one of two primitive shapes of argument:

* `disagreeFires` — two dims that the operator requires to be *equal* are both
  known and unequal (matmul contraction, broadcast trailing dims, cat/stack
  must-match dims, einsum repeated index, Linear in-features, unpack arity);
* `boundFires` — a known value violates a known bound (axis-out-of-range,
  index-out-of-bounds, negative dimension, reshape numel, division by zero).

For each we prove three things: it fires only on fully-known operands
(conservativity), firing entails the precondition fails on *all* concretizations
(soundness), and firing entails a concrete witness exists (the certificate).

Pure Lean 4 core (no mathlib).
-/

import TensorGuard.Symexec.Galois

namespace TensorGuard
namespace Symexec
namespace Core

open Dim

/-! ## Equality-required dimensions -/

/-- Fires when two dims required to be equal are both known and unequal. -/
def disagreeFires (a b : Dim) : Bool :=
  match a, b with
  | .known x, .known y => !(x = y)
  | _,        _        => false

/-- **Conservativity.**  If either operand is ⊤ (unknown), the check abstains. -/
theorem disagree_conservative_left (b : Dim) : disagreeFires .unk b = false := by
  cases b <;> rfl

theorem disagree_conservative_right (a : Dim) : disagreeFires a .unk = false := by
  cases a <;> rfl

/-- **Refutation soundness.**  If the check fires, then *no* pair of concrete
sizes modeled by `a` and `b` can be equal — so the equality precondition fails
on every concretization (no false positive). -/
theorem disagree_sound {a b : Dim} (h : disagreeFires a b = true) :
    ∀ c1 c2, dimModels c1 a → dimModels c2 b → c1 ≠ c2 := by
  cases a with
  | unk => simp [disagreeFires] at h
  | known x =>
    cases b with
    | unk => simp [disagreeFires] at h
    | known y =>
      intro c1 c2 h1 h2
      simp only [dimModels] at h1 h2
      subst h1; subst h2
      simp only [disagreeFires] at h
      simpa using h

/-- **Certified counterexample.**  If the check fires, a concrete witness pair
exists that is modeled by the operands yet violates the equality requirement. -/
theorem disagree_witness {a b : Dim} (h : disagreeFires a b = true) :
    ∃ c1 c2, dimModels c1 a ∧ dimModels c2 b ∧ c1 ≠ c2 := by
  cases a with
  | unk => simp [disagreeFires] at h
  | known x =>
    cases b with
    | unk => simp [disagreeFires] at h
    | known y =>
      refine ⟨x, y, ?_, ?_, ?_⟩
      · simp [dimModels]
      · simp [dimModels]
      · simp only [disagreeFires] at h; simpa using h

/-! ## Bounded values

A single abstract `Val` with a known integer (or ⊤) covers the scalar checks
(axis, index, dimension size, divisor).  We reuse `Dim` as that carrier and
phrase bounds over `Nat`. -/

/-- Fires when a known value `v` is `≥` a known bound `n` (out of range for a
0-based axis/index of extent `n`). -/
def geBoundFires (v n : Dim) : Bool :=
  match v, n with
  | .known a, .known b => decide (a ≥ b)
  | _,        _        => false

theorem geBound_conservative_left (n : Dim) : geBoundFires .unk n = false := by
  cases n <;> rfl

theorem geBound_conservative_right (v : Dim) : geBoundFires v .unk = false := by
  cases v <;> rfl

/-- **Refutation soundness for out-of-range.**  If the check fires, every
concrete value/bound pair modeled by the operands has `value ≥ bound`, i.e. the
in-range precondition `value < bound` fails on all concretizations. -/
theorem geBound_sound {v n : Dim} (h : geBoundFires v n = true) :
    ∀ a b, dimModels a v → dimModels b n → a ≥ b := by
  cases v with
  | unk => simp [geBoundFires] at h
  | known a0 =>
    cases n with
    | unk => simp [geBoundFires] at h
    | known b0 =>
      intro a b h1 h2
      simp only [dimModels] at h1 h2
      subst h1; subst h2
      simp only [geBoundFires] at h
      exact (by simpa using h)

/-- **Certified counterexample for out-of-range.** -/
theorem geBound_witness {v n : Dim} (h : geBoundFires v n = true) :
    ∃ a b, dimModels a v ∧ dimModels b n ∧ a ≥ b := by
  cases v with
  | unk => simp [geBoundFires] at h
  | known a0 =>
    cases n with
    | unk => simp [geBoundFires] at h
    | known b0 =>
      refine ⟨a0, b0, by simp [dimModels], by simp [dimModels], ?_⟩
      simp only [geBoundFires] at h
      exact (by simpa using h)

/-! ## Zero divisor -/

/-- Fires when a divisor is the known constant `0`. -/
def zeroFires (v : Dim) : Bool :=
  match v with
  | .known a => decide (a = 0)
  | .unk     => false

theorem zero_conservative : zeroFires .unk = false := rfl

/-- **Refutation soundness for division by zero.**  If the check fires, every
concrete divisor modeled by `v` is `0`. -/
theorem zero_sound {v : Dim} (h : zeroFires v = true) :
    ∀ a, dimModels a v → a = 0 := by
  cases v with
  | unk => simp [zeroFires] at h
  | known a0 =>
    intro a ha
    simp only [dimModels] at ha
    subst ha
    simp only [zeroFires] at h
    simpa using h

theorem zero_witness {v : Dim} (h : zeroFires v = true) :
    ∃ a, dimModels a v ∧ a = 0 := by
  cases v with
  | unk => simp [zeroFires] at h
  | known a0 =>
    refine ⟨a0, by simp [dimModels], ?_⟩
    simp only [zeroFires] at h; simpa using h

end Core
end Symexec
end TensorGuard
