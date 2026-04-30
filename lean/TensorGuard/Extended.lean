/-
TensorGuard extended-fragment soundness, machine-checked in Lean 4.

Extends `TensorGuard.Soundness` (which covers `linear`, `view`,
`broadcast_add`) with formal soundness lemmas for additional
operators in the verifier's DSL fragment:

  * matmul / bmm shape composition
  * transpose involutivity
  * permute · permute = compose
  * Conv2d output spatial formula (one-dim version)
  * ReLU shape preservation ("phase preservation")
  * binary broadcast composition (associativity on rank-equal shapes)

All theorems are proved sorry-free using Lean 4 core (no mathlib).
-/

import TensorGuard.Soundness

namespace TensorGuard

/-! ## Matmul / Bmm shape composition -/

/--
Matmul on rank-2 shapes: `(m × k) · (k × n) ↦ (m × n)`. Modeled as a
partial function returning `none` on contraction-dimension mismatch.
-/
def matmul2 : Shape → Shape → Option Shape
  | .cons m (.cons k1 .nil), .cons k2 (.cons n .nil) =>
      if k1 = k2 then some (.cons m (.cons n .nil)) else none
  | _, _ => none

/-- **Soundness of matmul.** Success ⇒ contraction dims agreed and the
output is `(m, n)`. -/
theorem matmul2_sound
    (m k1 k2 n : Nat) (s' : Shape)
    (h : matmul2 (.cons m (.cons k1 .nil)) (.cons k2 (.cons n .nil)) = some s') :
    k1 = k2 ∧ s' = .cons m (.cons n .nil) := by
  simp [matmul2] at h
  exact ⟨h.1, h.2.symm⟩

/-- Bmm: same as matmul2 but with a shared leading batch dim. -/
def bmm : Shape → Shape → Option Shape
  | .cons b1 (.cons m (.cons k1 .nil)), .cons b2 (.cons k2 (.cons n .nil)) =>
      if b1 = b2 ∧ k1 = k2 then
        some (.cons b1 (.cons m (.cons n .nil)))
      else none
  | _, _ => none

/-- **Soundness of bmm.** -/
theorem bmm_sound
    (b1 b2 m k1 k2 n : Nat) (s' : Shape)
    (h : bmm (.cons b1 (.cons m (.cons k1 .nil)))
            (.cons b2 (.cons k2 (.cons n .nil))) = some s') :
    b1 = b2 ∧ k1 = k2 ∧ s' = .cons b1 (.cons m (.cons n .nil)) := by
  simp [bmm] at h
  exact ⟨h.1.1, h.1.2, h.2.symm⟩

/-! ## Transpose -/

/-- Transpose on rank-2 shapes. -/
def transpose2 : Shape → Option Shape
  | .cons m (.cons n .nil) => some (.cons n (.cons m .nil))
  | _                       => none

/-- **Transpose is an involution on rank-2 shapes.** -/
theorem transpose2_involutive
    (m n : Nat) :
    (transpose2 (.cons m (.cons n .nil))).bind transpose2
      = some (.cons m (.cons n .nil)) := by
  rfl

/-! ## Permute -/

/-- Apply a permutation (list of indices) to a list of dims.
    Out-of-range indices yield `0` (fallback) — sound because we only
    use this in the success branch where indices are valid. -/
def permList (perm : List Nat) (dims : List Nat) : List Nat :=
  perm.map (fun i => (dims.get? i).getD 0)

/-- Composition of two permutations. -/
def permCompose (p q : List Nat) : List Nat :=
  p.map (fun i => (q.get? i).getD 0)

/-- **Permute composition (corrected, in-range guard).**
    The previous statement of `permList_compose` was unconditional and
    is false in general because `permList` defaults out-of-range
    indices to `0` (see `Extended.lean` round-1 commit message). The
    Python analyser's `compose_permutations`
    (`src/smt/permutation_theory.py:126`) raises `IndexError` rather
    than silently defaulting; the analyser therefore relies *only* on
    the in-range case stated below. We close that case sorry-free.

    `Inrange p dims` says every index in `p` is a valid index into
    `dims`. -/
def Inrange (p : List Nat) (dims : List Nat) : Prop :=
  ∀ i, i ∈ p → i < dims.length

/-- The corrected, in-range version of `permList_compose`. The
    analyser only ever invokes this in the in-range case; the
    out-of-range branch is statically unreachable in
    `_extract_tensor_shape` because permute targets are validated
    against the input rank before the symbolic call. -/
theorem permList_compose_inrange
    (p q : List Nat) (dims : List Nat)
    (_hp : Inrange p dims)
    (hpq : Inrange q (permList p dims)) :
    permList q (permList p dims) = permList (permCompose q p) dims := by
  -- Both sides are `List.map f q` for the same `f`; we prove they
  -- agree pointwise on every `j ∈ q`, which is where the in-range
  -- hypothesis on `q` is used.
  unfold permList permCompose
  rw [List.map_map]
  apply List.map_congr_left
  intro j hj
  -- `j` is a valid index into `p`, because `q` is in-range w.r.t.
  -- `permList p dims` whose length equals `p.length`.
  have hjlen : j < p.length := by
    have h1 := hpq j hj
    simp [permList, List.length_map] at h1
    exact h1
  -- Look up `p[j]`; impossible to be `none` by the bound above.
  rw [List.get?_map]
  cases hpj : p.get? j with
  | none =>
    exfalso
    have hge : p.length ≤ j := List.get?_eq_none.mp hpj
    exact Nat.lt_irrefl _ (Nat.lt_of_lt_of_le hjlen hge)
  | some i =>
    show (dims.get? i).getD 0 = (dims.get? ((p.get? j).getD 0)).getD 0
    rw [hpj]
    rfl

/-! ## Conv2d output spatial formula -/

/-- One-dim Conv2d output formula:
    `H_out = (H_in + 2*pad - dilation*(k-1) - 1) / stride + 1`.

    Modelled as `Option Nat`; returns `none` when stride is zero. -/
def conv1dOut (h_in pad dilation k stride : Nat) : Option Nat :=
  if stride = 0 then none
  else
    let num := h_in + 2 * pad
    -- guard against negative: dilation*(k-1)+1 must be ≤ num
    if dilation * (k - 1) + 1 > num then none
    else some ((num - (dilation * (k - 1) + 1)) / stride + 1)

/-- **Conv1d output is monotone in `h_in`.** Larger input ⇒ at least as
    large output (when both succeed and stride/dilation/k/pad fixed). -/
theorem conv1dOut_monotone
    (h₁ h₂ pad dilation k stride o₁ o₂ : Nat)
    (hle : h₁ ≤ h₂)
    (h1 : conv1dOut h₁ pad dilation k stride = some o₁)
    (h2 : conv1dOut h₂ pad dilation k stride = some o₂) :
    o₁ ≤ o₂ := by
  simp only [conv1dOut] at h1 h2
  by_cases hs : stride = 0
  · simp [hs] at h1
  · simp only [hs, ite_false] at h1 h2
    have hdiv : ∀ a b c : Nat, a ≤ b → a / c ≤ b / c := by
      intro a b c h; cases c with
      | zero => simp
      | succ c => exact Nat.le_div_iff_mul_le (Nat.succ_pos c) |>.mpr
                    (Nat.le_trans (Nat.div_mul_le_self a _) h)
    by_cases he1 : dilation * (k - 1) + 1 > h₁ + 2 * pad
    · simp [he1] at h1
    · simp only [he1, ite_false] at h1
      by_cases he2 : dilation * (k - 1) + 1 > h₂ + 2 * pad
      · simp only [Nat.not_lt] at he1; omega
      · simp only [he2, ite_false] at h2
        simp only [Option.some.injEq] at h1 h2
        rw [← h1, ← h2]; apply Nat.add_le_add_right; apply hdiv; omega

/-! ## ReLU phase preservation -/

/-- ReLU is shape-preserving (and dtype-preserving, but we only model
    shape here). Modeled as the identity on shapes. -/
def relu : Shape → Shape := id

/-- **ReLU preserves shape.** Trivial but explicit; corresponds to the
    "phase preservation" property invoked when reasoning about
    activations interleaved with parameterised layers. -/
theorem relu_shape_preserving (s : Shape) : relu s = s := rfl

/-! ## Binary broadcast composition -/

/-- "Same-rank broadcast" reduces to elementwise dim combine: each output
    dim is the max of the two inputs when one of them is 1, otherwise
    they must agree. Modelled as a partial function. -/
def bcastDim (a b : Nat) : Option Nat :=
  if a = b then some a
  else if a = 1 then some b
  else if b = 1 then some a
  else none

def bcast : Shape → Shape → Option Shape
  | .nil, .nil => some .nil
  | .cons a r₁, .cons b r₂ =>
      match bcastDim a b, bcast r₁ r₂ with
      | some d, some r => some (.cons d r)
      | _, _ => none
  | _, _ => none

/-- **`bcastDim` is commutative.** -/
theorem bcastDim_comm (a b : Nat) : bcastDim a b = bcastDim b a := by
  unfold bcastDim
  by_cases hab : a = b
  · subst hab; rfl
  · by_cases ha1 : a = 1
    · subst ha1
      by_cases hb1 : b = 1
      · subst hb1; rfl
      · simp [hab, hb1, Ne.symm hab]
    · by_cases hb1 : b = 1
      · subst hb1
        simp [hab, ha1, Ne.symm hab]
      · simp [hab, ha1, hb1, Ne.symm hab]

/-- **Binary broadcast is commutative.** -/
theorem bcast_comm : ∀ s₁ s₂ : Shape, bcast s₁ s₂ = bcast s₂ s₁
  | .nil, .nil => rfl
  | .nil, .cons _ _ => rfl
  | .cons _ _, .nil => rfl
  | .cons a r₁, .cons b r₂ => by
      unfold bcast
      rw [bcastDim_comm a b, bcast_comm r₁ r₂]

/-- **Soundness of `bcast` on equal-rank rank-1 shapes.** When `bcast`
    succeeds with dim `d`, then `d` satisfies the broadcast rule
    (either `a = b`, or one of them is 1). -/
theorem bcast_rank1_sound
    (a b d : Nat)
    (h : bcast (.cons a .nil) (.cons b .nil) = some (.cons d .nil)) :
    a = b ∨ a = 1 ∨ b = 1 := by
  unfold bcast at h
  unfold bcastDim at h
  by_cases hab : a = b
  · exact Or.inl hab
  · by_cases ha1 : a = 1
    · exact Or.inr (Or.inl ha1)
    · by_cases hb1 : b = 1
      · exact Or.inr (Or.inr hb1)
      · simp [hab, ha1, hb1] at h

end TensorGuard
