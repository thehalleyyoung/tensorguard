/-
TensorGuard `nn.Embedding` shape-and-index rule, machine-checked in Lean 4
(Step 149).

`nn.Embedding(num_embeddings, embedding_dim)` is a learned lookup table.  Applied
to an integer index tensor of shape `S` it returns a tensor of shape
`S ++ [embedding_dim]` (rank **+1**, the new trailing dim being `embedding_dim`)
— exactly what `src/model_checker.py::_propagate_embedding` computes.  It is a
runtime error unless **every index is in range** `0 ≤ idx < num_embeddings`.

**Scope (honest):** indices are modelled as `Nat`, so the lower bound
`0 ≤ idx` holds by construction; the proved guard is the **upper bound**
`idx < num_embeddings` (the bound that distinguishes a valid table lookup from an
out-of-range one).  The companion test additionally exercises a real **negative**
index against torch to confirm the lower bound is enforced by the engine.

Proved laws:

  * **rank +1** (`emb_rank`): the output rank is the input rank plus one;
  * **trailing dim** (`emb_trailing`): the appended dim is exactly
    `embedding_dim`, and the leading dims are the unchanged input
    (`emb_prefix`);
  * **numel scaling** (`emb_numel`): the output numel is the input numel times
    `embedding_dim`;
  * **index-range refutation** (`idxValid_iff`, `allValid_iff`): the guard flags
    iff some index is out of range — the soundness direction behind the
    out-of-bounds embedding alarm;
  * **range monotonicity** (`allValid_mono`): growing `num_embeddings` only ever
    keeps a valid index set valid (enlarging the table never invalidates a
    lookup).

The companion test `tests/test_embedding_rule.py` replays the rule on a **real**
`nn.Embedding`: the output shape equals `input.shape + (embedding_dim,)` and an
out-of-range index makes torch **raise** — exactly when the Lean guard flags it.

Pure Lean 4 core (no mathlib).
-/

namespace TensorGuard
namespace EmbeddingRule

/-- Product (numel) of a list of concrete dim sizes. -/
def prod : List Nat → Nat
  | [] => 1
  | d :: rest => d * prod rest

theorem prod_append (xs ys : List Nat) : prod (xs ++ ys) = prod xs * prod ys := by
  induction xs with
  | nil => simp [prod]
  | cons a t ih => simp [prod, ih, Nat.mul_assoc]

/-- Output shape of an embedding lookup: input dims with `embedding_dim`
    appended. -/
def embShape (input : List Nat) (embDim : Nat) : List Nat :=
  input ++ [embDim]

/- ===================================================================== -/
/- 1. Shape laws                                                         -/
/- ===================================================================== -/

/-- **Rank +1**: the lookup raises the rank by exactly one. -/
theorem emb_rank (input : List Nat) (embDim : Nat) :
    (embShape input embDim).length = input.length + 1 := by
  simp [embShape]

/-- **Trailing dim**: dropping the (unchanged) input prefix leaves exactly the
    appended `embedding_dim`. -/
theorem emb_trailing (input : List Nat) (embDim : Nat) :
    (embShape input embDim).drop input.length = [embDim] := by
  simp [embShape, List.drop_left]

/-- The leading dims are exactly the input shape, untouched. -/
theorem emb_prefix (input : List Nat) (embDim : Nat) :
    (embShape input embDim).take input.length = input := by
  simp [embShape, List.take_left]

/-- **Numel scaling**: the output element count is the input count times
    `embedding_dim`. -/
theorem emb_numel (input : List Nat) (embDim : Nat) :
    prod (embShape input embDim) = prod input * embDim := by
  simp [embShape, prod_append, prod]

/- ===================================================================== -/
/- 2. Index-range rule                                                   -/
/- ===================================================================== -/

/-- A single index is in range iff it is `< num_embeddings`. -/
def idxValid (idx num : Nat) : Bool := decide (idx < num)

/-- Every index in a batch is in range. -/
def allValid (idxs : List Nat) (num : Nat) : Bool :=
  idxs.all (fun i => idxValid i num)

/-- A lone index passes the guard iff it is genuinely in range. -/
theorem idxValid_iff (idx num : Nat) : idxValid idx num = true ↔ idx < num := by
  unfold idxValid; simp

/-- **Refutation soundness**: the batch guard passes iff *every* index is in
    range (so an out-of-range index is always flagged). -/
theorem allValid_iff (idxs : List Nat) (num : Nat) :
    allValid idxs num = true ↔ ∀ i ∈ idxs, i < num := by
  unfold allValid
  rw [List.all_eq_true]
  constructor
  · intro h i hi; exact (idxValid_iff i num).mp (h i hi)
  · intro h i hi; exact (idxValid_iff i num).mpr (h i hi)

/-- A concrete out-of-range index is flagged. -/
theorem outOfRange_flagged (idx num : Nat) (h : num ≤ idx) :
    idxValid idx num = false := by
  unfold idxValid; simp [Nat.not_lt.mpr h]

/-- **Range monotonicity**: enlarging the table (`num ≤ num'`) keeps a valid
    index set valid — growing `num_embeddings` never invalidates a lookup. -/
theorem allValid_mono (idxs : List Nat) (num num' : Nat)
    (hle : num ≤ num') (h : allValid idxs num = true) :
    allValid idxs num' = true := by
  rw [allValid_iff] at h ⊢
  intro i hi
  exact Nat.lt_of_lt_of_le (h i hi) hle

end EmbeddingRule
end TensorGuard
