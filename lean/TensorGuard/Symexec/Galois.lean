/-
TensorGuard.Symexec.Galois

The Galois connection (α, γ) for the dimension lattice of the symbolic-execution
engine (SYMEXEC_100_STEPS Step 93).  `src/symexec/concretize.py` implements
`alpha`/`gamma`; here we prove they form a Galois connection:

    α(c) ⊑ d   ↔   c ∈ γ(d)

and the standard consequences (γ ∘ α is extensive, α is the best abstraction).
This is the formal underpinning of the abstraction-soundness notes in
`docs/symexec/semantics.md` and of the whole-program soundness theorem
(`Symexec.Soundness`).

Pure Lean 4 core (no mathlib).
-/

import TensorGuard.Symexec.Lattice

namespace TensorGuard
namespace Symexec
namespace Dim

/-- Abstraction of a single concrete size: the most precise abstract dim that
admits it.  This is `alpha` for the dimension lattice. -/
def alpha (c : Nat) : Dim := .known c

/-- **Galois connection.**  `α(c) ⊑ d ↔ c ∈ γ(d)`: the abstraction of a
concrete size is below an abstract dim exactly when that dim concretizes to the
size.  This is the defining adjunction of the (α, γ) pair. -/
theorem galois (c : Nat) (d : Dim) :
    dle (alpha c) d = true ↔ dimModels c d := by
  cases d with
  | unk => simp [alpha, dle, dimModels]
  | known m =>
    simp only [alpha, dle, dimModels]
    constructor
    · intro h; exact (Nat.eq_of_beq_eq_true (by simpa using h))
    · intro h; subst h; simp

/-- **γ ∘ α is extensive.**  Every concrete size is in the concretization of its
own abstraction (the unit of the adjunction). -/
theorem gamma_alpha_extensive (c : Nat) : dimModels c (alpha c) := by
  simp [alpha, dimModels]

/-- **α is the best abstraction.**  Any abstract dim that admits `c` is above
`α(c)`; no abstraction of `c` is strictly more precise than `α(c)`. -/
theorem alpha_best (c : Nat) (d : Dim) (h : dimModels c d) :
    dle (alpha c) d = true :=
  (galois c d).2 h

/-- **Soundness direction (the one the bug checks rely on).**  If `α(c) ⋢ d`
then `c ∉ γ(d)` — an abstract value that does *not* sit above `α(c)` genuinely
excludes the concrete size `c`.  Contrapositive of `alpha_best`. -/
theorem excludes_of_not_le (c : Nat) (d : Dim)
    (h : dle (alpha c) d = false) : ¬ dimModels c d := by
  intro hm
  rw [alpha_best c d hm] at h
  exact Bool.noConfusion h

end Dim
end Symexec
end TensorGuard
