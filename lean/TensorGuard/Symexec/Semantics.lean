/-
TensorGuard.Symexec.Semantics

A Lean rendering of the engine's abstract small-step semantics
(`docs/symexec/semantics.md`, Step 91) together with the *uniform* soundness
framework that every bug check satisfies.

We abstract the common shape of a check into two records:

* `DimPairCheck` — a check over two dims requiring a binary precondition `Ok`
  (matmul / broadcast / reshape / cat-stack / linear / unpack / einsum / axis /
  index);
* `UnaryCheck` — a check over a single dim requiring a unary precondition
  (division-by-zero).

Each record bundles its `fires` predicate with the two proof obligations the
engine must meet — **refutation soundness** (`fires ⇒ no concretization
satisfies Ok`) and a **certified witness** (`fires ⇒ a violating concretization
exists`).  The meta-theorems here (`pair_no_false_positive`,
`pair_certificate`, …) then hold for *any* such check, and the
`TensorGuard.Symexec.Transfer.*` modules supply the concrete instances.

Pure Lean 4 core (no mathlib).
-/

import TensorGuard.Symexec.Store

namespace TensorGuard
namespace Symexec
namespace Sem

open Dim

/-- A binary dimension check with its proof obligations. -/
structure DimPairCheck where
  fires   : Dim → Dim → Bool
  Ok      : Nat → Nat → Prop
  sound   : ∀ {a b}, fires a b = true →
              ∀ x y, dimModels x a → dimModels y b → ¬ Ok x y
  witness : ∀ {a b}, fires a b = true →
              ∃ x y, dimModels x a ∧ dimModels y b ∧ ¬ Ok x y

/-- **No false positive (binary).**  If a binary check fires on `(a, b)`, then
there is no concretization of the operands satisfying the precondition. -/
theorem pair_no_false_positive (C : DimPairCheck) {a b : Dim}
    (h : C.fires a b = true) :
    ¬ ∃ x y, dimModels x a ∧ dimModels y b ∧ C.Ok x y := by
  rintro ⟨x, y, hx, hy, hok⟩
  exact C.sound h x y hx hy hok

/-- **Certificate (binary).**  A fired binary check yields a concrete failing
witness (the proof-carrying counterexample of Step 94). -/
theorem pair_certificate (C : DimPairCheck) {a b : Dim}
    (h : C.fires a b = true) :
    ∃ x y, dimModels x a ∧ dimModels y b ∧ ¬ C.Ok x y :=
  C.witness h

/-- A unary dimension check with its proof obligations. -/
structure UnaryCheck where
  fires   : Dim → Bool
  Ok      : Nat → Prop
  sound   : ∀ {a}, fires a = true → ∀ x, dimModels x a → ¬ Ok x
  witness : ∀ {a}, fires a = true → ∃ x, dimModels x a ∧ ¬ Ok x

theorem unary_no_false_positive (C : UnaryCheck) {a : Dim}
    (h : C.fires a = true) : ¬ ∃ x, dimModels x a ∧ C.Ok x := by
  rintro ⟨x, hx, hok⟩
  exact C.sound h x hx hok

theorem unary_certificate (C : UnaryCheck) {a : Dim}
    (h : C.fires a = true) : ∃ x, dimModels x a ∧ ¬ C.Ok x :=
  C.witness h

/-! ## A straight-line program and the report it induces

A program is a list of binary checks applied to abstract operands drawn from a
store.  The engine *reports* iff some check fires; the soundness theorem says a
non-empty report entails a genuinely failing concrete execution. -/

/-- One step of the modeled program: apply binary check `C` to the dims held by
two variables.  (Assignments thread values through the store; here we expose the
checking step, which is where bugs are emitted.) -/
structure Step where
  check : DimPairCheck
  lhs   : String
  rhs   : String

/-- The step fires under store `σ`. -/
def Step.fires (s : Step) (σ : Store) : Bool :=
  s.check.fires (σ s.lhs) (σ s.rhs)

/-- **Local soundness of a checking step.**  If the step fires under an abstract
store `σ`, then *every* concrete store modeled by `σ` makes the checked
operation's precondition fail on the two operands — no concrete run modeled by
`σ` satisfies it. -/
theorem step_local_sound (s : Step) (σ : Store) (c : CStore)
    (hσ : StoreModels c σ) (hf : s.fires σ = true) :
    ¬ s.check.Ok (c s.lhs) (c s.rhs) := by
  have := s.check.sound (a := σ s.lhs) (b := σ s.rhs) hf
  exact this (c s.lhs) (c s.rhs) (hσ s.lhs) (hσ s.rhs)

/-- **Local certificate of a checking step.**  A fired step exhibits a concrete
store, modeled by `σ`, on which the operation fails — extending any model on the
two operands to the witnessing sizes. -/
theorem step_certificate (s : Step) (σ : Store) (hf : s.fires σ = true) :
    ∃ x y, dimModels x (σ s.lhs) ∧ dimModels y (σ s.rhs) ∧ ¬ s.check.Ok x y :=
  s.check.witness hf

end Sem
end Symexec
end TensorGuard
