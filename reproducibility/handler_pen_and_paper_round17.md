# Pen-and-paper soundness promotions (round 17)

This document records pen-and-paper soundness arguments for a set of
handlers that were previously categorised as `tested_only`.  For each
handler we (a) state the implemented forward/backward shape rule,
(b) state the standard PyTorch contract for the operator, and
(c) give a short proof sketch that (a) matches (b) on every well-typed
input.  The result is that these handlers can be moved from the
`tested_only` bucket to the `pen_and_paper` bucket of the audited
soundness footprint, on the same footing as `elementwise_binary`,
`reduce`, and `einsum` (which were already in that bucket).

The dispatch tables that implement the rules are referenced by line
range only; the rules themselves are reproduced inline so that the
proofs are self-contained.

## Notation

Write a tensor specification as `Tensor(s)` where `s = (d_1, ..., d_n)`
is the shape and each `d_i` is either a positive integer or a free
symbolic positive integer.  We write `[s_1, ..., s_n]` for an input
list.  The forward rule is a partial function `Forward : Op × [Shape]
→ Shape`; the backward rule is a partial function
`Backward : Op × [Shape] × Shape → [Optional[Shape]]`.

Elementwise unary preservation is the lemma:

> **Lemma EU.** Let `f : R → R` be a measurable scalar function.  The
> tensor lift `F(x) = f.(x)` is shape-preserving:
> `Forward(F, [s]) = s` and
> `Backward(F, [s], g) = [g]`.

This is immediate from the definition of broadcasting: `f` is applied
pointwise, and PyTorch's autograd `Function` for `f` returns a
gradient of the same shape as the input by construction (see
`torch.autograd.Function` invariants documented in the PyTorch
Internals notes).

## Handlers promoted to `pen_and_paper`

### 1. `relu`, `gelu`, `silu`, `tanh`, `sigmoid`

* **Forward rule (implemented):** identity on shape:
  `Forward(act, [s]) = s`.
* **Backward rule (implemented):** identity on shape:
  `Backward(act, [s], g) = [g]`, with `g.shape = s`.
* **PyTorch contract:** `torch.nn.functional.relu` (and the analogous
  `gelu`, `silu`, `tanh`, `sigmoid`) is defined as the elementwise
  application of a fixed scalar function `R → R`.
* **Soundness proof:** Direct application of Lemma EU to
  `f(x) = max(0, x)`, `f = gelu`, `f = silu`, `f = tanh`,
  `f = sigmoid` respectively.  None of these functions changes the
  number of dimensions or the size of any dimension; the gradient is
  `f'(x) ⊙ g`, which has the same shape as `x` because Hadamard
  product preserves shape.  ∎

### 2. `softmax`, `log_softmax`

* **Forward rule (implemented):** `Forward(softmax, [s], dim=k) = s`.
* **Backward rule (implemented):** `Backward(softmax, [s], g) = [g]`
  with `g.shape = s`.
* **PyTorch contract:** `softmax(x, dim=k)` divides each entry by the
  sum of `exp` over axis `k`; the output therefore has the same shape
  as `x`.  The Jacobian `J_ij = δ_ij p_i − p_i p_j` (within an axis-`k`
  slice) is square in axis `k` and the identity outside it, so the
  vector–Jacobian product preserves shape.
* **Soundness proof:** The output is `exp(x) / Σ_k exp(x)` where the
  denominator is broadcast over axis `k`; this is shape-preserving
  because both the numerator and the broadcast denominator have shape
  `s`.  The backward step is a sum-over-axis-`k` of the Jacobian
  applied to `g`, which restores axis `k` to length `s_k`, giving a
  result of shape `s`.  ∎

### 3. `dropout`

* **Forward rule (implemented):** `Forward(dropout, [s], p) = s`.
* **Backward rule (implemented):** `Backward(dropout, [s], g) = [g]`.
* **PyTorch contract:** `dropout` multiplies each element by an
  independent Bernoulli mask of shape `s` and rescales by `1/(1−p)`;
  the output has shape `s`.
* **Soundness proof:** Multiplication of two shape-`s` tensors is
  shape-preserving (Lemma EU applied to the bilinear form `(x, m) ↦
  x · m / (1−p)` viewed as a unary function of `x` for fixed `m`).
  The backward rule multiplies the upstream gradient by the same
  saved mask, hence preserves shape.  ∎

### 4. `detach`

* **Forward rule (implemented):** `Forward(detach, [s]) = s`.
* **Backward rule (implemented):** `Backward(detach, [s], g) = [None]`
  (no gradient flow).
* **PyTorch contract:** `detach()` returns a tensor sharing storage
  with its input but with `requires_grad=False`; the shape is
  unchanged, and no gradient flows through it.
* **Soundness proof:** `detach` is the identity on storage and shape;
  the backward edge is severed by construction.  Severance is sound:
  if no gradient is produced for the input, there is no shape
  obligation to discharge.  ∎

### 5. `squeeze`, `unsqueeze`, `flatten`

* **Forward rule (implemented):**
  `Forward(squeeze, [s], dim=k) = s` with `s_k` removed if `s_k = 1`,
  else `s`.  `Forward(unsqueeze, [s], dim=k) = s` with a `1` inserted
  at position `k`.  `Forward(flatten, [s], start=a, end=b) =
  (s_1, …, s_{a−1}, ∏_{i=a}^{b} s_i, s_{b+1}, …, s_n)`.
* **Backward rule (implemented):** the inverse permutation/reshape on
  the gradient: `Backward(squeeze, [s], g) = [unsqueeze(g, dim=k)]`,
  symmetrically for `unsqueeze` and `flatten`.
* **PyTorch contract:** All three are pure index-reshape operations
  with deterministic, total shape functions.  They never change the
  underlying storage layout (modulo `contiguous()` for non-contiguous
  inputs to `flatten`), and the backward is the corresponding inverse
  reshape.
* **Soundness proof:** Each rule is the deterministic shape function
  given by the PyTorch documentation; we read off the rule from the
  documentation and inspect the implementation in
  `src/tensor_shapes.py` (lines 1269–1310) to confirm structural
  equality.  For any well-typed input the implemented rule and the
  documented rule produce the same shape on every dimension.  The
  backward composes the inverse on the gradient, which has the same
  shape as the forward input by structural induction on the reshape
  expression.  ∎

### 6. `pad`

* **Forward rule (implemented):** for `pad(x, pad=(p_l1, p_r1, …, p_lk,
  p_rk))` over the last `k` dims of an `n`-d tensor `s`, the output
  shape is
  `(s_1, …, s_{n−k}, s_{n−k+1} + p_l1 + p_r1, …, s_n + p_lk + p_rk)`.
* **Backward rule:** crop the gradient back to `s` by removing the
  padded margins; the result has shape `s`.
* **PyTorch contract:** `torch.nn.functional.pad` documents exactly
  this shape function for `mode in {"constant", "reflect",
  "replicate", "circular"}`.
* **Soundness proof:** The output shape is a fixed linear function of
  the input shape and the constant `pad` tuple; both the forward and
  the cropping backward are structurally equal to the documented
  formula by inspection of `src/stdlib/modern_ops.py` MODERN_TORCH_SHAPE_OPS
  entry for `pad`.  ∎

### 7. `where`, `masked_fill`

* **Forward rule (implemented):** `Forward(where, [s_c, s_t, s_f]) =
  broadcast(s_c, s_t, s_f)`; `Forward(masked_fill, [s, s_m, ()]) =
  broadcast(s, s_m)`.
* **Backward rule:** the input gradients are produced by
  `_broadcast_reduce` (sum-to), which is the same routine used by
  `add`, `sub`, `mul`, `div` (the `elementwise_binary` family already
  in the pen-and-paper bucket).
* **PyTorch contract:** `where(c, x, y)` is elementwise selection with
  full NumPy-style broadcasting on `(c, x, y)`; `masked_fill(x, m, v)`
  is elementwise `x[m] = v` with broadcasting on `(x, m)`.
* **Soundness proof:** Reduce both ops to the elementwise-broadcast
  family by treating `where` as a 3-ary elementwise op and
  `masked_fill` as a 2-ary elementwise op; the broadcast rule is the
  one already audited under `elementwise_binary` (Lemma B from
  appendix), and `_broadcast_reduce` is the corresponding
  shape-reduction adjoint.  ∎

## Effect on the audited footprint

After promoting these 11 handlers, the partition of the 79 active
handlers becomes:

| bucket          | before | after |
|---|---:|---:|
| Lean-verified   | 28     | 28    |
| pen-and-paper   | 3      | 14    |
| tested-only     | 48     | 37    |

We re-ran the per-block handler-scope script on the 488-block real
corpus and the 185 in-soundness verdicts (V+CV under the headline
regime) with the new partition; the resulting numbers are stored in
`reproducibility/handler_scope_per_block_round17.json` and the
markdown re-render in `reproducibility/handler_scope_per_block.md`.
The promotion roughly doubles the audited-only footprint inside the
soundness theorem (the round-20 reconciliation supersedes the
earlier $36/185$ figure with the canonical $62/185$ partition;
see `reproducibility/canonical_partition_round20.md`).

## Citations

* PyTorch operator semantics: PyTorch 2.9.1 documentation for the
  named operators (https://docs.pytorch.org/docs/2.9/).
* Implemented forward rules: `src/tensor_shapes.py`,
  `src/stdlib/modern_ops.py`, and `src/smt/encoder.py`
  (FUNCTIONAL_SHAPE_RULES).
* Implemented backward rules: `src/v5/backward_shape.py` SHAPE_RULES,
  in particular `_unary_same_shape` (lines 145–155) for the activation
  family and `_broadcast_reduce` (lines 120–127) for the broadcast
  adjoint.
* Pre-existing pen-and-paper bucket: `src/typing_rules.py` Soundness
  Conjecture comment (around line 16), reproduced as Theorem A.7 of
  the appendix in the paper.

This artifact is referenced by the eval-section sentence on the
audited footprint and by the limitations paragraph on the
pen-and-paper / Lean partition.
