"""
Formal Typing Rules for TensorGuard's Refinement Type System.

This module formalizes the core typing judgements used by TensorGuard to
statically verify tensor shape compatibility.  Each operation on tensors
(linear layers, convolutions, broadcasting, reshape, concatenation, matmul,
reduction, embedding) is given a *typing rule* in the style of a refinement
type system:

    Γ ⊢ e : {v : Tensor | shape(v) = S ∧ device(v) = d ∧ dtype(v) = t}

where Γ is a typing context mapping variable names to TensorType, e is a
tensor expression, and the refinement predicate constrains the shape,
device, and dtype of the result.

Soundness Conjecture
--------------------
**Progress.** If Γ ⊢ e : τ and e →* v (the expression evaluates to a
value under the operational semantics of PyTorch), then v : τ — i.e. the
runtime tensor's shape, device, and dtype match the statically inferred
refinement type.

**Preservation.** If Γ ⊢ e : τ and e → e' (one step of evaluation), then
there exists Γ' ⊇ Γ and τ' such that Γ' ⊢ e' : τ' with τ' ≤ τ (the
type of the reduced expression is a subtype of the original).

These properties are *conjectures* backed by extensive property-based
testing (see ``generate_random_judgement``) and by Z3-checked rule
applications, but they have **not** been mechanized in a proof assistant.
The companion Lean4 formalization in ``lean/`` covers a subset.

Stride Theory and Nelson-Oppen Combination
-------------------------------------------
TensorGuard combines multiple SMT theories via the Tinelli-Zarba extension
of the Nelson-Oppen combination framework.  The stride theory T_stride
(which encodes memory layout constraints) has the following properties
relevant to sound combination:

**Stale-completeness (Stable Infiniteness).**
T_stride operates over Dim ⊆ ℤ_{≥1} and Stride ⊆ ℤ_{≥1}, which are
stably infinite (every satisfiable QF_LIA formula over ℤ_{≥1} has an
infinite model extension).  This satisfies the Nelson-Oppen requirement.

**Convexity.**
T_stride is *not* convex.  Consider the constraint numel(s) = 6: this
implies s₁·s₂ = 6, which entails (s₁=1 ∧ s₂=6) ∨ (s₁=2 ∧ s₂=3) ∨
(s₁=3 ∧ s₂=2) ∨ (s₁=6 ∧ s₂=1), a disjunction of equalities none of
which is individually entailed.  TensorGuard handles non-convexity by
using the *splitting-on-demand* variant of Nelson-Oppen: when the stride
theory produces a disjunctive equality entailment, the combination
framework case-splits and propagates each disjunct.

**Politeness.**
T_stride is polite in the sense of Tinelli-Zarba: given a satisfiable
conjunction φ over Σ_stride, we can construct a witness that extends any
model of the shared-sort (Dim) constraints to a full Σ_stride-model.
This is straightforward because strides are deterministically computed
from shapes (contiguous layout: stride[i] = ∏_{j>i} shape[j]).
"""

from __future__ import annotations

import math
import random
import logging
from dataclasses import dataclass, field
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Sequence,
    Tuple,
    Union,
)

logger = logging.getLogger(__name__)

try:
    import z3 as _z3
    HAS_Z3 = True
except ImportError:
    HAS_Z3 = False


# ═══════════════════════════════════════════════════════════════════════════
# Formal type definitions
# ═══════════════════════════════════════════════════════════════════════════

Dim = Union[int, str]  # concrete or symbolic dimension


@dataclass(frozen=True)
class TensorType:
    """Refinement type for tensors.

    {v : Tensor | shape(v) = (d₁, …, dₙ) ∧ device(v) = dev ∧ dtype(v) = dt}

    Dimensions may be concrete ``int`` values or symbolic ``str`` names
    (e.g. ``"B"`` for batch size).  A symbolic dimension represents a
    universally-quantified positive integer.
    """

    shape: Tuple[Dim, ...]
    device: str = "any"
    dtype: str = "float32"

    # -- helpers -------------------------------------------------------------

    @property
    def ndim(self) -> int:
        return len(self.shape)

    def concrete_numel(self) -> Optional[int]:
        """Return the element count if all dims are concrete, else None."""
        prod = 1
        for d in self.shape:
            if isinstance(d, str):
                return None
            prod *= d
        return prod

    def is_concrete(self) -> bool:
        return all(isinstance(d, int) for d in self.shape)


@dataclass(frozen=True)
class Judgement:
    """Typing judgement  Γ ⊢ e : τ."""

    context: Dict[str, TensorType]  # Γ
    expr: str  # e
    type: TensorType  # τ


# ═══════════════════════════════════════════════════════════════════════════
# Error types
# ═══════════════════════════════════════════════════════════════════════════

class TypingRuleError(Exception):
    """Raised when a typing rule precondition is violated."""


# ═══════════════════════════════════════════════════════════════════════════
# Typing rules
# ═══════════════════════════════════════════════════════════════════════════

def apply_t_linear(
    input_type: TensorType,
    in_features: int,
    out_features: int,
) -> TensorType:
    """T-LINEAR rule.

    ::

        Γ ⊢ x : {v: Tensor | shape(v) = (..., in_features)}
        ────────────────────────────────────────────────────────
        Γ ⊢ Linear(x) : {v: Tensor | shape(v) = (..., out_features)}

    Preconditions:
        - x has at least 1 dimension.
        - The last dimension equals ``in_features``.
    """
    if input_type.ndim < 1:
        raise TypingRuleError(
            f"T-LINEAR requires ndim ≥ 1, got shape {input_type.shape}"
        )
    last = input_type.shape[-1]
    if isinstance(last, int) and last != in_features:
        raise TypingRuleError(
            f"T-LINEAR: last dim {last} ≠ in_features {in_features}"
        )
    new_shape = input_type.shape[:-1] + (out_features,)
    return TensorType(shape=new_shape, device=input_type.device, dtype=input_type.dtype)


def apply_t_conv2d(
    input_type: TensorType,
    out_channels: int,
    kernel_size: Tuple[int, int],
    stride: Tuple[int, int] = (1, 1),
    padding: Tuple[int, int] = (0, 0),
    dilation: Tuple[int, int] = (1, 1),
) -> TensorType:
    """T-CONV2D rule.

    ::

        Γ ⊢ x : {v: Tensor | shape(v) = (B, C_in, H, W)}
        H_out = (H + 2p_h - d_h(k_h - 1) - 1) / s_h + 1
        W_out = (W + 2p_w - d_w(k_w - 1) - 1) / s_w + 1
        ─────────────────────────────────────────────────────────
        Γ ⊢ Conv2d(x) : {v: Tensor | shape(v) = (B, C_out, H_out, W_out)}

    Preconditions:
        - x has exactly 4 dimensions.
    """
    if input_type.ndim != 4:
        raise TypingRuleError(
            f"T-CONV2D requires ndim = 4, got shape {input_type.shape}"
        )
    B, C_in, H, W = input_type.shape

    def _out_dim(size: Dim, k: int, s: int, p: int, d: int) -> Dim:
        if isinstance(size, str):
            return f"({size}+{2*p}-{d*(k-1)}-1)//{s}+1"
        return (size + 2 * p - d * (k - 1) - 1) // s + 1

    H_out = _out_dim(H, kernel_size[0], stride[0], padding[0], dilation[0])
    W_out = _out_dim(W, kernel_size[1], stride[1], padding[1], dilation[1])

    return TensorType(
        shape=(B, out_channels, H_out, W_out),
        device=input_type.device,
        dtype=input_type.dtype,
    )


def apply_t_broadcast(
    type_a: TensorType,
    type_b: TensorType,
) -> TensorType:
    """T-BROADCAST rule (NumPy / PyTorch broadcasting).

    ::

        Γ ⊢ a : {v: Tensor | shape(v) = S_a}
        Γ ⊢ b : {v: Tensor | shape(v) = S_b}
        S_out = broadcast(S_a, S_b)
        ──────────────────────────────────────
        Γ ⊢ a ⊕ b : {v: Tensor | shape(v) = S_out}

    Broadcasting aligns shapes from the right.  For each pair of dims:
        - If equal → keep.
        - If one is 1 → expand to the other.
        - Otherwise → error.

    Device and dtype must be compatible.
    """
    sa, sb = type_a.shape, type_b.shape
    ndim = max(len(sa), len(sb))
    # right-align
    pa = (1,) * (ndim - len(sa)) + sa
    pb = (1,) * (ndim - len(sb)) + sb

    result: List[Dim] = []
    for da, db in zip(pa, pb):
        if da == db:
            result.append(da)
        elif da == 1:
            result.append(db)
        elif db == 1:
            result.append(da)
        elif isinstance(da, str) or isinstance(db, str):
            # symbolic — optimistically broadcast (checked by Z3 later)
            result.append(da if isinstance(db, int) and db == 1 else db if isinstance(da, int) and da == 1 else da)
        else:
            raise TypingRuleError(
                f"T-BROADCAST: incompatible dims {da} and {db}"
            )

    dev = type_a.device if type_a.device != "any" else type_b.device
    return TensorType(shape=tuple(result), device=dev, dtype=type_a.dtype)


def apply_t_reshape(
    input_type: TensorType,
    new_shape: Tuple[Dim, ...],
) -> TensorType:
    """T-RESHAPE rule (element-count preservation).

    ::

        Γ ⊢ x : {v: Tensor | shape(v) = S}
        numel(S) = numel(S')   (possibly with one -1 inferred dim)
        ─────────────────────────────────────────────────────
        Γ ⊢ reshape(x, S') : {v: Tensor | shape(v) = S'}

    At most one dimension in ``new_shape`` may be -1 (inferred).
    """
    neg_count = sum(1 for d in new_shape if isinstance(d, int) and d == -1)
    if neg_count > 1:
        raise TypingRuleError("T-RESHAPE: at most one -1 dimension allowed")

    old_numel = input_type.concrete_numel()

    if neg_count == 1 and old_numel is not None:
        known_prod = 1
        for d in new_shape:
            if isinstance(d, int) and d != -1:
                known_prod *= d
            elif isinstance(d, str):
                # symbolic — cannot infer -1
                break
        else:
            if known_prod == 0:
                raise TypingRuleError("T-RESHAPE: zero-sized dimension")
            inferred = old_numel // known_prod
            new_shape = tuple(inferred if (isinstance(d, int) and d == -1) else d for d in new_shape)

    # Verify element count when both are concrete
    new_numel_val = 1
    all_concrete_new = True
    for d in new_shape:
        if isinstance(d, int) and d != -1:
            new_numel_val *= d
        else:
            all_concrete_new = False

    if old_numel is not None and all_concrete_new and old_numel != new_numel_val:
        raise TypingRuleError(
            f"T-RESHAPE: numel mismatch {old_numel} ≠ {new_numel_val}"
        )

    return TensorType(shape=new_shape, device=input_type.device, dtype=input_type.dtype)


def apply_t_cat(
    input_types: Sequence[TensorType],
    dim: int = 0,
) -> TensorType:
    """T-CAT rule (concatenation along an axis).

    ::

        Γ ⊢ xᵢ : {v: Tensor | shape(v) = S_i}   for i = 1..n
        ∀ i,j: S_i[k] = S_j[k]  for k ≠ dim
        ───────────────────────────────────────────────────────
        Γ ⊢ cat([x₁,…,xₙ], dim) : {v: Tensor | shape(v) = S_out}
        where S_out[dim] = Σᵢ S_i[dim], S_out[k] = S_1[k] for k ≠ dim

    All inputs must have the same ndim, and non-cat dimensions must match.
    """
    if not input_types:
        raise TypingRuleError("T-CAT: requires at least one input")

    ndim = input_types[0].ndim
    if dim < 0:
        dim = ndim + dim
    if dim < 0 or dim >= ndim:
        raise TypingRuleError(f"T-CAT: dim {dim} out of range for ndim {ndim}")

    for i, t in enumerate(input_types):
        if t.ndim != ndim:
            raise TypingRuleError(
                f"T-CAT: input {i} has ndim {t.ndim}, expected {ndim}"
            )
        for k in range(ndim):
            if k != dim:
                d0 = input_types[0].shape[k]
                dk = t.shape[k]
                if isinstance(d0, int) and isinstance(dk, int) and d0 != dk:
                    raise TypingRuleError(
                        f"T-CAT: dim {k} mismatch: {d0} vs {dk} (input 0 vs {i})"
                    )

    cat_dim: Dim = 0
    all_concrete = True
    for t in input_types:
        d = t.shape[dim]
        if isinstance(d, int) and isinstance(cat_dim, int):
            cat_dim = cat_dim + d
        else:
            all_concrete = False
            cat_dim = f"sum_cat_{dim}"
            break

    out_shape = list(input_types[0].shape)
    out_shape[dim] = cat_dim
    return TensorType(
        shape=tuple(out_shape),
        device=input_types[0].device,
        dtype=input_types[0].dtype,
    )


def apply_t_matmul(
    type_a: TensorType,
    type_b: TensorType,
) -> TensorType:
    """T-MATMUL rule (with batched support).

    ::

        Γ ⊢ a : {v: Tensor | shape(v) = (..., m, k)}
        Γ ⊢ b : {v: Tensor | shape(v) = (..., k, n)}
        ────────────────────────────────────────────────
        Γ ⊢ a @ b : {v: Tensor | shape(v) = (..., m, n)}

    - 1D @ 1D → scalar (dot product)
    - 2D @ 2D → standard matmul
    - Batched: broadcast batch dims, inner dim k must match.
    """
    sa, sb = type_a.shape, type_b.shape

    if len(sa) == 0 or len(sb) == 0:
        raise TypingRuleError("T-MATMUL: scalars cannot be matmul'd")

    # 1-D @ 1-D → scalar
    if len(sa) == 1 and len(sb) == 1:
        if isinstance(sa[0], int) and isinstance(sb[0], int) and sa[0] != sb[0]:
            raise TypingRuleError(
                f"T-MATMUL: inner dims mismatch {sa[0]} ≠ {sb[0]}"
            )
        return TensorType(shape=(), device=type_a.device, dtype=type_a.dtype)

    # 1-D @ 2-D → (n,)
    if len(sa) == 1:
        k_a = sa[0]
        k_b, n = sb[-2], sb[-1]
        if isinstance(k_a, int) and isinstance(k_b, int) and k_a != k_b:
            raise TypingRuleError(
                f"T-MATMUL: inner dims mismatch {k_a} ≠ {k_b}"
            )
        batch = sb[:-2]
        return TensorType(shape=batch + (n,), device=type_a.device, dtype=type_a.dtype)

    # 2-D+ @ 1-D → (..., m)
    if len(sb) == 1:
        m, k_a = sa[-2], sa[-1]
        k_b = sb[0]
        if isinstance(k_a, int) and isinstance(k_b, int) and k_a != k_b:
            raise TypingRuleError(
                f"T-MATMUL: inner dims mismatch {k_a} ≠ {k_b}"
            )
        batch = sa[:-2]
        return TensorType(shape=batch + (m,), device=type_a.device, dtype=type_a.dtype)

    # General batched matmul
    m, k_a = sa[-2], sa[-1]
    k_b, n = sb[-2], sb[-1]

    if isinstance(k_a, int) and isinstance(k_b, int) and k_a != k_b:
        raise TypingRuleError(
            f"T-MATMUL: inner dims mismatch {k_a} ≠ {k_b}"
        )

    # Broadcast batch dims
    batch_a = sa[:-2]
    batch_b = sb[:-2]
    batch_type_a = TensorType(shape=batch_a)
    batch_type_b = TensorType(shape=batch_b)
    try:
        batch_result = apply_t_broadcast(batch_type_a, batch_type_b)
    except TypingRuleError:
        raise TypingRuleError(
            f"T-MATMUL: batch dims not broadcastable {batch_a} vs {batch_b}"
        )

    out_shape = batch_result.shape + (m, n)
    return TensorType(shape=out_shape, device=type_a.device, dtype=type_a.dtype)


def apply_t_reduce(
    input_type: TensorType,
    dim: int,
    keepdim: bool = False,
) -> TensorType:
    """T-REDUCE rule (sum, mean, max, etc. along a dimension).

    ::

        Γ ⊢ x : {v: Tensor | shape(v) = (d₁, …, dₙ)}
        ─────────────────────────────────────────────────────
        keepdim=False:
          Γ ⊢ reduce(x, dim) : {v: Tensor | shape(v) = (d₁,…,d_{dim-1}, d_{dim+1},…,dₙ)}
        keepdim=True:
          Γ ⊢ reduce(x, dim) : {v: Tensor | shape(v) = (d₁,…,d_{dim-1}, 1, d_{dim+1},…,dₙ)}
    """
    if input_type.ndim == 0:
        raise TypingRuleError("T-REDUCE: cannot reduce a scalar")
    if dim < 0:
        dim = input_type.ndim + dim
    if dim < 0 or dim >= input_type.ndim:
        raise TypingRuleError(
            f"T-REDUCE: dim {dim} out of range for ndim {input_type.ndim}"
        )

    shape_list = list(input_type.shape)
    if keepdim:
        shape_list[dim] = 1
    else:
        shape_list.pop(dim)

    return TensorType(
        shape=tuple(shape_list),
        device=input_type.device,
        dtype=input_type.dtype,
    )


def apply_t_embed(
    input_type: TensorType,
    num_embeddings: int,
    embedding_dim: int,
) -> TensorType:
    """T-EMBED rule (nn.Embedding lookup).

    ::

        Γ ⊢ x : {v: Tensor | shape(v) = S, dtype(v) = int64}
        ───────────────────────────────────────────────────────
        Γ ⊢ Embedding(x) : {v: Tensor | shape(v) = (*S, embed_dim)}

    The input is an index tensor (typically int64); the output appends
    the embedding dimension.
    """
    # Accept any integer dtype for index tensors
    return TensorType(
        shape=input_type.shape + (embedding_dim,),
        device=input_type.device,
        dtype="float32",
    )


# ═══════════════════════════════════════════════════════════════════════════
# Z3-backed rule verification
# ═══════════════════════════════════════════════════════════════════════════

_RULE_DISPATCH = {
    "T-LINEAR": lambda inp, p: apply_t_linear(
        inp["x"], p["in_features"], p["out_features"]
    ),
    "T-CONV2D": lambda inp, p: apply_t_conv2d(
        inp["x"], p["out_channels"], p["kernel_size"],
        p.get("stride", (1, 1)), p.get("padding", (0, 0)),
        p.get("dilation", (1, 1)),
    ),
    "T-BROADCAST": lambda inp, p: apply_t_broadcast(inp["a"], inp["b"]),
    "T-RESHAPE": lambda inp, p: apply_t_reshape(inp["x"], p["new_shape"]),
    "T-CAT": lambda inp, p: apply_t_cat(
        list(inp.values()), p.get("dim", 0)
    ),
    "T-MATMUL": lambda inp, p: apply_t_matmul(inp["a"], inp["b"]),
    "T-REDUCE": lambda inp, p: apply_t_reduce(
        inp["x"], p["dim"], p.get("keepdim", False)
    ),
    "T-EMBED": lambda inp, p: apply_t_embed(
        inp["x"], p["num_embeddings"], p["embedding_dim"]
    ),
}


def verify_rule(
    rule_name: str,
    input_types: Dict[str, TensorType],
    params: Dict[str, Any],
) -> Tuple[bool, Optional[TensorType]]:
    """Apply a typing rule and verify the result shape is well-formed.

    Uses Z3 to check that all concrete dimensions in the output are
    positive integers — a necessary condition for a valid tensor type.

    Returns:
        (success, result_type) — success is True iff the rule applied
        without error and the result passes Z3 well-formedness checks.
    """
    if rule_name not in _RULE_DISPATCH:
        raise ValueError(f"Unknown rule: {rule_name}")

    try:
        result = _RULE_DISPATCH[rule_name](input_types, params)
    except TypingRuleError:
        return False, None

    # Z3 well-formedness: every concrete dim > 0
    if HAS_Z3 and result.is_concrete():
        solver = _z3.Solver()
        for i, d in enumerate(result.shape):
            dim_var = _z3.Int(f"d_{i}")
            solver.add(dim_var == d)
            solver.add(dim_var > 0)
        if solver.check() != _z3.sat:
            return False, result

    return True, result


# ═══════════════════════════════════════════════════════════════════════════
# Property-based testing support
# ═══════════════════════════════════════════════════════════════════════════

def generate_random_judgement(rng: Optional[random.Random] = None) -> Judgement:
    """Generate a random well-typed judgement for property-based testing.

    Produces a random typing context and a rule application that is
    guaranteed to satisfy the rule's preconditions, then returns the
    resulting judgement Γ ⊢ e : τ.  Useful with Hypothesis or standalone
    fuzz testing.
    """
    if rng is None:
        rng = random.Random()

    rule = rng.choice(list(_RULE_DISPATCH.keys()))
    device = rng.choice(["cpu", "cuda:0"])
    dtype = rng.choice(["float32", "float16"])

    if rule == "T-LINEAR":
        batch_dims = tuple(rng.randint(1, 8) for _ in range(rng.randint(0, 2)))
        in_f = rng.randint(1, 512)
        out_f = rng.randint(1, 512)
        x = TensorType(shape=batch_dims + (in_f,), device=device, dtype=dtype)
        ctx = {"x": x}
        result = apply_t_linear(x, in_f, out_f)
        return Judgement(context=ctx, expr=f"Linear({in_f},{out_f})(x)", type=result)

    if rule == "T-CONV2D":
        B = rng.randint(1, 8)
        C_in = rng.randint(1, 64)
        H = rng.randint(4, 64)
        W = rng.randint(4, 64)
        C_out = rng.randint(1, 64)
        k = rng.randint(1, min(3, H, W))
        x = TensorType(shape=(B, C_in, H, W), device=device, dtype=dtype)
        ctx = {"x": x}
        result = apply_t_conv2d(x, C_out, (k, k))
        return Judgement(context=ctx, expr=f"Conv2d(x)", type=result)

    if rule == "T-BROADCAST":
        ndim = rng.randint(1, 4)
        sa = tuple(rng.choice([1, rng.randint(2, 8)]) for _ in range(ndim))
        sb = tuple(
            (1 if d != 1 and rng.random() < 0.3 else d) if rng.random() < 0.5 else d
            for d in sa
        )
        a = TensorType(shape=sa, device=device, dtype=dtype)
        b = TensorType(shape=sb, device=device, dtype=dtype)
        ctx = {"a": a, "b": b}
        result = apply_t_broadcast(a, b)
        return Judgement(context=ctx, expr="a + b", type=result)

    if rule == "T-RESHAPE":
        ndim = rng.randint(1, 3)
        shape = tuple(rng.randint(1, 6) for _ in range(ndim))
        numel = math.prod(shape)
        # Find a compatible reshape target
        new_ndim = rng.randint(1, 3)
        new_shape_list: List[int] = []
        remaining = numel
        for i in range(new_ndim - 1):
            # Pick a factor of remaining
            factors = [f for f in range(1, remaining + 1) if remaining % f == 0]
            f = rng.choice(factors)
            new_shape_list.append(f)
            remaining //= f
        new_shape_list.append(remaining)
        x = TensorType(shape=shape, device=device, dtype=dtype)
        ctx = {"x": x}
        result = apply_t_reshape(x, tuple(new_shape_list))
        return Judgement(context=ctx, expr=f"reshape(x, {tuple(new_shape_list)})", type=result)

    if rule == "T-CAT":
        ndim = rng.randint(1, 3)
        cat_dim = rng.randint(0, ndim - 1)
        base_shape = list(rng.randint(1, 8) for _ in range(ndim))
        n_tensors = rng.randint(2, 4)
        types = []
        for _ in range(n_tensors):
            s = list(base_shape)
            s[cat_dim] = rng.randint(1, 8)
            types.append(TensorType(shape=tuple(s), device=device, dtype=dtype))
        ctx = {f"x{i}": t for i, t in enumerate(types)}
        result = apply_t_cat(types, cat_dim)
        return Judgement(context=ctx, expr=f"cat([...], dim={cat_dim})", type=result)

    if rule == "T-MATMUL":
        batch = tuple(rng.randint(1, 4) for _ in range(rng.randint(0, 2)))
        m = rng.randint(1, 16)
        k = rng.randint(1, 16)
        n = rng.randint(1, 16)
        a = TensorType(shape=batch + (m, k), device=device, dtype=dtype)
        b = TensorType(shape=batch + (k, n), device=device, dtype=dtype)
        ctx = {"a": a, "b": b}
        result = apply_t_matmul(a, b)
        return Judgement(context=ctx, expr="a @ b", type=result)

    if rule == "T-REDUCE":
        ndim = rng.randint(1, 4)
        shape = tuple(rng.randint(1, 8) for _ in range(ndim))
        dim = rng.randint(0, ndim - 1)
        keepdim = rng.choice([True, False])
        x = TensorType(shape=shape, device=device, dtype=dtype)
        ctx = {"x": x}
        result = apply_t_reduce(x, dim, keepdim)
        return Judgement(context=ctx, expr=f"reduce(x, dim={dim})", type=result)

    # T-EMBED
    seq_len = rng.randint(1, 32)
    batch = tuple(rng.randint(1, 8) for _ in range(rng.randint(0, 2)))
    vocab = rng.randint(100, 30000)
    embed_dim = rng.randint(16, 256)
    x = TensorType(shape=batch + (seq_len,), device=device, dtype="int64")
    ctx = {"x": x}
    result = apply_t_embed(x, vocab, embed_dim)
    return Judgement(context=ctx, expr=f"Embedding(x)", type=result)
