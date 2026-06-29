/-
TensorGuard.Symexec.Store

The abstract store σ and the soundness of control-flow merges (`State.join` /
`State.widen` in `src/symexec/state.py`).  A store maps variable names to
dimension abstractions; `StoreModels` lifts the pointwise concretization to
whole stores.  We prove the store join is an upper bound and that a concrete
store modeled by *either* branch is modeled by their join — this is the
soundness of merging branches at an `if`/loop join point.  Widening coincides
with join here (the dimension lattice has height 2), so loops terminate.

Pure Lean 4 core (no mathlib).
-/

import TensorGuard.Symexec.Lattice

namespace TensorGuard
namespace Symexec

open Dim

/-- An abstract store: a total map from names to dimension abstractions
(a name not yet bound reads as ⊤). -/
abbrev Store := String → Dim

/-- A concrete store maps names to concrete sizes. -/
abbrev CStore := String → Nat

/-- The empty/initial store: everything unknown. -/
def Store.top : Store := fun _ => .unk

/-- Pointwise join of two stores (the merge at a control-flow join). -/
def Store.join (σ τ : Store) : Store := fun x => Dim.join (σ x) (τ x)

/-- Widening; equal to join on this finite-height lattice. -/
def Store.widen (σ τ : Store) : Store := Store.join σ τ

/-- A concrete store is modeled by an abstract store iff every variable's
concrete size is modeled by the abstract dim. -/
def StoreModels (c : CStore) (σ : Store) : Prop :=
  ∀ x, dimModels (c x) (σ x)

/-- The join is an upper bound of its left argument at every variable. -/
theorem store_join_le_left (σ τ : Store) (x : String) :
    dle (σ x) (Store.join σ τ x) = true :=
  dle_join_left (σ x) (τ x)

theorem store_join_le_right (σ τ : Store) (x : String) :
    dle (τ x) (Store.join σ τ x) = true :=
  dle_join_right (σ x) (τ x)

/-- **Merge soundness (left branch).**  A concrete store modeled by the left
branch is modeled by the merged store. -/
theorem storeModels_join_left {c : CStore} {σ τ : Store}
    (h : StoreModels c σ) : StoreModels c (Store.join σ τ) :=
  fun x => dimModels_join_left (h x)

/-- **Merge soundness (right branch).** -/
theorem storeModels_join_right {c : CStore} {σ τ : Store}
    (h : StoreModels c τ) : StoreModels c (Store.join σ τ) :=
  fun x => dimModels_join_right (h x)

/-- **Widening soundness.**  Widening preserves concretization from either side
(it equals the join), so it is a sound — and, being ⊤-saturating in one step,
terminating — replacement for join in loops. -/
theorem storeModels_widen_left {c : CStore} {σ τ : Store}
    (h : StoreModels c σ) : StoreModels c (Store.widen σ τ) :=
  storeModels_join_left h

theorem storeModels_widen_right {c : CStore} {σ τ : Store}
    (h : StoreModels c τ) : StoreModels c (Store.widen σ τ) :=
  storeModels_join_right h

/-- The top store models every concrete store (the sound starting point). -/
theorem storeModels_top (c : CStore) : StoreModels c Store.top :=
  fun _ => by simp [Store.top, dimModels]

end Symexec
end TensorGuard
