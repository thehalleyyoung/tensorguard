/-
TensorGuard core-fragment soundness, machine-checked in Lean 4.

This file models a tiny shape DSL covering three operators (`linear`, `view`,
`broadcast_add`) and proves that the partial transition function `applyOp`
is *sound*: whenever it returns `some s'`, the structural / numerical
preconditions encoded in the operator's typing rule must have held.

Pure Lean 4 core (no mathlib).
-/

namespace TensorGuard

inductive Shape
  | nil  : Shape
  | cons : Nat → Shape → Shape
  deriving DecidableEq, Repr

namespace Shape

def dims : Shape → List Nat
  | .nil       => []
  | .cons n r  => n :: r.dims

def prod : Shape → Nat
  | .nil       => 1
  | .cons n r  => n * r.prod

def ofList : List Nat → Shape
  | []      => .nil
  | n :: ns => .cons n (ofList ns)

end Shape

inductive Op
  | linear        (in_dim out_dim : Nat) : Op
  | view          (out : List Nat)        : Op
  | broadcast_add                          : Op
  deriving Repr

/-- Total folded product of a list, used as the size invariant for `view`. -/
def listProd (xs : List Nat) : Nat :=
  xs.foldl (· * ·) 1

/--
Partial shape transition function. Returns `some s'` when the operator is
applicable to the input shape, and `none` otherwise.
-/
def applyOp : Op → Shape → Option Shape
  | .linear i o, .cons n .nil =>
      if n = i then some (.cons o .nil) else none
  | .view out, s =>
      if s.prod = listProd out then
        some (Shape.ofList out)
      else none
  | .broadcast_add, s => some s
  | _, _ => none

/-! ## Soundness lemmas -/

/--
**Soundness of `linear`.** If `applyOp (.linear i o) (.cons n .nil)` succeeds
with verdict `s'`, then the input dimension matched (`n = i`) and the output
shape is exactly `.cons o .nil`.
-/
theorem applyOp_sound_linear
    (i o n : Nat) (s' : Shape)
    (h : applyOp (.linear i o) (.cons n .nil) = some s') :
    s' = .cons o .nil ∧ n = i := by
  simp [applyOp] at h
  exact ⟨h.2.symm, h.1⟩

/--
**Soundness of `view`.** If `applyOp (.view out) s` succeeds, then the input
shape's element count equals the product of the requested view dimensions.
-/
theorem applyOp_sound_view
    (out : List Nat) (s s' : Shape)
    (h : applyOp (.view out) s = some s') :
    s.prod = listProd out := by
  simp [applyOp] at h
  exact h.1

/--
**Soundness of `view` (verdict shape).** When `view` succeeds, the verdict
shape is the canonical encoding of the requested dimension list.
-/
theorem applyOp_view_verdict
    (out : List Nat) (s s' : Shape)
    (h : applyOp (.view out) s = some s') :
    s' = Shape.ofList out := by
  simp [applyOp] at h
  exact h.2.symm

/--
**Divisibility precondition for `view`.** For every concrete instantiation
in which `view out` succeeds and the dimension list `out` has positive
product, the input element count is *exactly* divisible by the product of
all but one chosen target dim.  This is the formal hypothesis that
appears in the strengthened statement of \texttt{Theorem 1} (paper),
explicitly promoted from the appendix into the headline theorem.
-/
theorem applyOp_view_divisible
    (out : List Nat) (s s' : Shape) (k : Nat)
    (hpos : listProd out = k) (hk : 0 < k)
    (h : applyOp (.view out) s = some s') :
    s.prod = k := by
  have hsp : s.prod = listProd out := applyOp_sound_view _ _ _ h
  rw [hsp, hpos]

/--
**Strict view divisibility (no rounding).** A consequence of the previous
lemma: if `view` succeeds then there is no remainder.  Equivalent to
`s.prod % listProd out = 0` *and* the quotient equals `1`.
-/
theorem applyOp_view_no_remainder
    (out : List Nat) (s s' : Shape)
    (h : applyOp (.view out) s = some s') :
    s.prod % listProd out = 0 := by
  have : s.prod = listProd out := applyOp_sound_view _ _ _ h
  rw [this]
  exact Nat.mod_self _

/--
**Soundness of `broadcast_add`.** The (modeled) broadcast-add is the identity
on shapes, so the verdict equals the input.
-/
theorem applyOp_sound_broadcast_add
    (s s' : Shape)
    (h : applyOp .broadcast_add s = some s') :
    s' = s := by
  unfold applyOp at h
  exact (Option.some.inj h).symm

/--
**No spurious success.** `linear` only fires on rank-1 input shapes; any other
input rules out a `some` verdict.
-/
theorem applyOp_linear_rank
    (i o : Nat) (s s' : Shape)
    (h : applyOp (.linear i o) s = some s') :
    ∃ n, s = .cons n .nil := by
  cases s with
  | nil =>
      simp [applyOp] at h
  | cons n r =>
      cases r with
      | nil => exact ⟨n, rfl⟩
      | cons m r' => simp [applyOp] at h

end TensorGuard
