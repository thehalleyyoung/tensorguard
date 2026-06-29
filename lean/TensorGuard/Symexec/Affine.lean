/-
TensorGuard.Symexec.Affine

Machine-checked soundness of the **symbolic-dimension (affine) layer** that
underpins TensorGuard's relational shape reasoning (`src/symexec/symdim.py`).

`SymDim` models a tensor dimension as an affine integer form `const + Σ cᵢ·vᵢ`.
Two oracles built on it drive real detectors:

  * `SymDim.definitely_eq` — used wherever the engine must decide that two
    symbolic dimensions are *equal on every concrete assignment* (symbolic shape
    equality in matmul/cat/broadcast/reshape over batch-like dims). It returns
    `True` **only** when the normalized difference `a - b` is the zero form.

  * `SymDim.definitely_divisible_by k` — used by the reshape / einops
    decomposition checks (a group of size `g` must evenly divide a dimension).
    It returns `True` **only** when `k` divides the constant and every
    coefficient.

This file proves both oracles **sound** against the obvious concrete semantics
(evaluate the affine form under an arbitrary integer assignment to its
variables), plus that the affine transfer functions (`+`, `·` by a constant) are
*exact* homomorphisms.  Together these are the soundness obligations of the
symbolic-dim abstraction — the relational counterpart to the per-detector
forced-failure refutations, extending the machine-checked frontier below the
op-level checks down into the dimension algebra itself.

Pure Lean 4 core (no mathlib).
-/

namespace TensorGuard
namespace Symexec
namespace Affine

/-- An assignment of a concrete integer extent to each dimension variable. -/
abbrev Env := String → Int

/-- An affine integer form `const + Σ (cᵢ · vᵢ)`.  Terms may repeat a variable;
the concrete semantics simply sums them, so this faithfully models both the
pre- and post-normalization representations in `symdim.py` (normalization only
combines like terms and drops zero coefficients, which preserves `eval`). -/
structure Affine where
  const : Int
  terms : List (String × Int)

/-- Evaluate the variable part of an affine form under an assignment. -/
def evalTerms (env : Env) : List (String × Int) → Int
  | []            => 0
  | (n, c) :: rest => c * env n + evalTerms env rest

/-- Concrete semantics: the integer value of the form under an assignment. -/
def eval (a : Affine) (env : Env) : Int :=
  a.const + evalTerms env a.terms

/-- Addition transfer function: concatenate the term lists, add the constants.
(`symdim.py` merges like terms afterwards; that is a `eval`-preserving
normalization, modeled by `evalTerms_append`.) -/
def add (a b : Affine) : Affine :=
  ⟨a.const + b.const, a.terms ++ b.terms⟩

/-- Negation transfer function (scales every coefficient and the constant). -/
def neg (a : Affine) : Affine :=
  ⟨-a.const, a.terms.map (fun p => (p.1, -p.2))⟩

/-- Subtraction, as the engine computes `a - b`. -/
def sub (a b : Affine) : Affine :=
  add a (neg b)

/-- Scalar (constant) multiplication transfer function — the only multiplication
that stays affine in `symdim.py` (`linear · constant`). -/
def smul (k : Int) (a : Affine) : Affine :=
  ⟨k * a.const, a.terms.map (fun p => (p.1, k * p.2))⟩

/-- The semantic "this form is identically zero" predicate: a zero constant and
every coefficient zero.  `definitely_eq` fires exactly when the normalized
difference reaches this shape (empty term list ⇒ the `∀`-clause is vacuous). -/
def IsZero (a : Affine) : Prop :=
  a.const = 0 ∧ ∀ p ∈ a.terms, p.2 = 0

/-- The oracle precondition of `definitely_divisible_by k`: `k` divides the
constant and every coefficient. -/
def AllDvd (k : Int) (a : Affine) : Prop :=
  (k ∣ a.const) ∧ ∀ p ∈ a.terms, k ∣ p.2

-- --------------------------------------------------------------------------- --
-- Homomorphism lemmas: the affine transfer functions are exact.               --
-- --------------------------------------------------------------------------- --

/-- The variable-part evaluator is additive over list concatenation. -/
theorem evalTerms_append (env : Env) :
    ∀ xs ys : List (String × Int),
      evalTerms env (xs ++ ys) = evalTerms env xs + evalTerms env ys
  | [],            ys => by simp [evalTerms]
  | (n, c) :: rest, ys => by
      simp [evalTerms, evalTerms_append env rest ys]
      omega

/-- **Addition is exact.** -/
theorem eval_add (a b : Affine) (env : Env) :
    eval (add a b) env = eval a env + eval b env := by
  simp only [eval, add, evalTerms_append env a.terms b.terms]
  omega

/-- The variable-part evaluator negates when every coefficient is negated. -/
theorem evalTerms_neg (env : Env) :
    ∀ ts : List (String × Int),
      evalTerms env (ts.map (fun p => (p.1, -p.2))) = - evalTerms env ts
  | []            => by simp [evalTerms]
  | (n, c) :: rest => by
      simp [evalTerms, evalTerms_neg env rest]
      omega

/-- **Negation is exact.** -/
theorem eval_neg (a : Affine) (env : Env) :
    eval (neg a) env = - eval a env := by
  simp only [eval, neg, evalTerms_neg env a.terms]
  omega

/-- **Subtraction is exact.** -/
theorem eval_sub (a b : Affine) (env : Env) :
    eval (sub a b) env = eval a env - eval b env := by
  simp only [sub, eval_add, eval_neg]
  omega

/-- The variable-part evaluator scales when every coefficient is scaled. -/
theorem evalTerms_smul (env : Env) (k : Int) :
    ∀ ts : List (String × Int),
      evalTerms env (ts.map (fun p => (p.1, k * p.2))) = k * evalTerms env ts
  | []            => by simp [evalTerms]
  | (n, c) :: rest => by
      simp [evalTerms, evalTerms_smul env k rest]
      rw [Int.mul_add, Int.mul_assoc]

/-- **Constant multiplication is exact.** -/
theorem eval_smul (k : Int) (a : Affine) (env : Env) :
    eval (smul k a) env = k * eval a env := by
  simp only [eval, smul, evalTerms_smul env k a.terms]
  rw [Int.mul_add]

-- --------------------------------------------------------------------------- --
-- Oracle soundness.                                                           --
-- --------------------------------------------------------------------------- --

/-- The variable part of an all-zero-coefficient form evaluates to `0`. -/
theorem evalTerms_allzero (env : Env) :
    ∀ ts : List (String × Int), (∀ p ∈ ts, p.2 = 0) → evalTerms env ts = 0
  | [],            _ => by simp [evalTerms]
  | (n, c) :: rest, h => by
      have hc : c = 0 := h (n, c) (List.mem_cons_self (n, c) rest)
      have hrest : ∀ p ∈ rest, p.2 = 0 := fun p hp =>
        h p (List.mem_cons_of_mem (n, c) hp)
      simp only [evalTerms, hc, evalTerms_allzero env rest hrest]
      simp

/-- An identically-zero form evaluates to `0` under every assignment. -/
theorem eval_isZero {a : Affine} (h : IsZero a) (env : Env) : eval a env = 0 := by
  obtain ⟨hc, ht⟩ := h
  have hterms : evalTerms env a.terms = 0 := evalTerms_allzero env a.terms ht
  simp [eval, hc, hterms]

/-- **`definitely_eq` is sound.** When the engine certifies two symbolic
dimensions equal — i.e. their difference is the zero form — they are equal under
*every* concrete assignment.  The equality oracle never claims a false equality,
so a detector relying on it never abstains-when-it-should nor fires-when-it-
shouldn't on symbolic shape equality. -/
theorem definitely_eq_sound {a b : Affine} (h : IsZero (sub a b)) (env : Env) :
    eval a env = eval b env := by
  have h0 : eval (sub a b) env = 0 := eval_isZero h env
  rw [eval_sub] at h0
  omega

/-- `k` divides each summand of a coefficient-divisible term list. -/
theorem dvd_evalTerms {k : Int} (env : Env) :
    ∀ ts : List (String × Int), (∀ p ∈ ts, k ∣ p.2) → k ∣ evalTerms env ts
  | [],            _ => by simp only [evalTerms]; exact ⟨0, (Int.mul_zero k).symm⟩
  | (n, c) :: rest, h => by
      have hc : k ∣ c := h (n, c) (List.mem_cons_self (n, c) rest)
      have hrest : ∀ p ∈ rest, k ∣ p.2 := fun p hp =>
        h p (List.mem_cons_of_mem (n, c) hp)
      have hd : k ∣ evalTerms env rest := dvd_evalTerms env rest hrest
      have hterm : k ∣ c * env n := by
        obtain ⟨q, rfl⟩ := hc
        exact ⟨q * env n, Int.mul_assoc k q (env n)⟩
      simp only [evalTerms]
      obtain ⟨s, hs⟩ := hterm
      obtain ⟨t, ht⟩ := hd
      exact ⟨s + t, by rw [hs, ht, Int.mul_add]⟩

/-- **`definitely_divisible_by` is sound.** When `k` divides the constant and
every coefficient of an affine dimension, it divides the dimension's value under
*every* concrete assignment — so the reshape/einops decomposition-divisibility
oracle is sound (it never wrongly certifies a divisibility that a concrete shape
could violate). -/
theorem definitely_divisible_sound {k : Int} {a : Affine}
    (h : AllDvd k a) (env : Env) : k ∣ eval a env := by
  obtain ⟨hc, ht⟩ := h
  have hd : k ∣ evalTerms env a.terms := dvd_evalTerms env a.terms ht
  obtain ⟨s, hs⟩ := hc
  obtain ⟨t, ht'⟩ := hd
  exact ⟨s + t, by simp only [eval]; rw [hs, ht', Int.mul_add]⟩

end Affine
end Symexec
end TensorGuard
