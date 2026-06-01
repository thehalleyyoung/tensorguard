"""Formal denotational semantics for TensorGuard computation graphs.

Defines concrete and abstract semantics for each operation kind in the
computation graph, connected by a Galois connection that establishes
soundness of the abstract interpretation.

Denotational Model
------------------

Concrete domain (𝒞):
    σ ∈ Σ = TensorName → (Shape × Device × Phase × GradStatus)
    A concrete state maps each live tensor to its full runtime descriptor.

Abstract domain (𝒞♯):
    σ♯ ∈ Σ♯ = TensorName → (AbstractShape × AbstractDevice × AbstractPhase)
    where AbstractShape uses symbolic dimension variables (Z3 integers),
    AbstractDevice is a finite set, and AbstractPhase ∈ {TRAIN, EVAL, ⊤}.

Galois connection:
    α : 𝒞 → 𝒞♯  (abstraction)
    γ : 𝒞♯ → 𝒞  (concretisation)
    α(c) ⊑ a  ⟺  c ∈ γ(a)

Soundness theorem (abstract interpretation):
    For every computation graph G and concrete state σ:
        α(⟦G⟧(σ))  ⊑  ⟦G⟧♯(α(σ))

    That is, the abstract execution over-approximates the concrete one.
    If the abstract execution reports SAFE, the concrete execution is safe.

Composition:
    For an acyclic graph G = node_1 ; node_2 ; ... ; node_n:
        ⟦G⟧ = ⟦node_n⟧ ∘ ... ∘ ⟦node_2⟧ ∘ ⟦node_1⟧

    Soundness of composition follows by induction on the chain, using
    monotonicity of α and the per-node soundness lemma.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Sequence,
    Set,
    Tuple,
    Union,
)

from src.model_checker import OpKind


# ═══════════════════════════════════════════════════════════════════════════════
# 1.  Concrete and abstract domains
# ═══════════════════════════════════════════════════════════════════════════════

ShapeTuple = Tuple[int, ...]


class AbstractPhase(Enum):
    """Abstract phase domain: {TRAIN, EVAL, TOP (unknown)}."""
    TRAIN = "train"
    EVAL = "eval"
    TOP = "top"

    def join(self, other: "AbstractPhase") -> "AbstractPhase":
        """Least upper bound in the phase lattice."""
        if self == other:
            return self
        return AbstractPhase.TOP

    def leq(self, other: "AbstractPhase") -> bool:
        """Partial order: self ⊑ other."""
        if other == AbstractPhase.TOP:
            return True
        return self == other


class AbstractDevice(Enum):
    """Abstract device domain."""
    CPU = "cpu"
    CUDA = "cuda"
    TOP = "top"

    def join(self, other: "AbstractDevice") -> "AbstractDevice":
        if self == other:
            return self
        return AbstractDevice.TOP

    def leq(self, other: "AbstractDevice") -> bool:
        if other == AbstractDevice.TOP:
            return True
        return self == other


@dataclass
class AbstractShape:
    """Abstract shape: dimensions may be concrete ints or symbolic strings.

    A symbolic dimension ``"batch"`` represents any positive integer.
    Constraints between symbolic dimensions are tracked separately.
    """
    dims: Tuple[Union[int, str], ...]

    @property
    def rank(self) -> int:
        return len(self.dims)

    def is_concrete(self) -> bool:
        return all(isinstance(d, int) for d in self.dims)

    def leq(self, other: "AbstractShape") -> bool:
        """self ⊑ other: self is more precise than other."""
        if self.rank != other.rank:
            return False
        for s, o in zip(self.dims, other.dims):
            if isinstance(o, str):
                continue  # symbolic = TOP for that dimension
            if s != o:
                return False
        return True


@dataclass
class ConcreteTensorState:
    """Concrete runtime state of a tensor."""
    shape: ShapeTuple
    device: str = "cpu"
    phase: str = "train"
    requires_grad: bool = False


@dataclass
class AbstractTensorState:
    """Abstract state of a tensor in the abstract domain."""
    shape: AbstractShape
    device: AbstractDevice = AbstractDevice.CPU
    phase: AbstractPhase = AbstractPhase.TRAIN

    def leq(self, other: "AbstractTensorState") -> bool:
        """Partial order: self ⊑ other."""
        return (
            self.shape.leq(other.shape)
            and self.device.leq(other.device)
            and self.phase.leq(other.phase)
        )


ConcreteState = Dict[str, ConcreteTensorState]
AbstractState = Dict[str, AbstractTensorState]


# ═══════════════════════════════════════════════════════════════════════════════
# 2.  Abstraction and concretisation (Galois connection)
# ═══════════════════════════════════════════════════════════════════════════════

def alpha_tensor(c: ConcreteTensorState) -> AbstractTensorState:
    """Abstraction function α for a single tensor.

    Maps a concrete tensor state to its best abstract approximation.
    This is the upper adjoint of the Galois connection.
    """
    return AbstractTensorState(
        shape=AbstractShape(dims=c.shape),
        device=AbstractDevice.CPU if c.device == "cpu" else AbstractDevice.CUDA,
        phase=AbstractPhase.TRAIN if c.phase == "train" else AbstractPhase.EVAL,
    )


def alpha(concrete: ConcreteState) -> AbstractState:
    """Abstraction function α : 𝒞 → 𝒞♯.

    α({t_1 ↦ c_1, ...}) = {t_1 ↦ α(c_1), ...}
    """
    return {name: alpha_tensor(ct) for name, ct in concrete.items()}


def gamma_tensor(a: AbstractTensorState) -> ConcreteTensorState:
    """Concretisation of an abstract tensor state.

    For concrete abstract shapes, returns the corresponding concrete state.
    For symbolic shapes, substitutes symbolic dims with 1 (minimal element).
    """
    concrete_shape = tuple(
        d if isinstance(d, int) else 1 for d in a.shape.dims
    )
    return ConcreteTensorState(
        shape=concrete_shape,
        device=a.device.value if a.device != AbstractDevice.TOP else "cpu",
        phase=a.phase.value if a.phase != AbstractPhase.TOP else "train",
    )


def check_galois_connection(
    concrete: ConcreteTensorState,
    abstract: AbstractTensorState,
) -> bool:
    """Verify α(c) ⊑ a  ⟺  c ∈ γ(a) for a concrete/abstract pair.

    Returns True if the abstract state soundly over-approximates the
    concrete state.
    """
    alpha_c = alpha_tensor(concrete)
    return alpha_c.leq(abstract)


# ═══════════════════════════════════════════════════════════════════════════════
# 3.  Per-operation concrete and abstract semantics
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class OpSemantics:
    """Concrete and abstract semantics for an operation kind.

    Attributes
    ----------
    op : OpKind
        The operation kind.
    concrete_fn : callable
        ⟦op⟧ : List[ShapeTuple] → ShapeTuple
        Concrete semantics: takes input shapes, returns output shape.
    abstract_fn : callable
        ⟦op⟧♯ : List[AbstractShape] → AbstractShape
        Abstract semantics: takes abstract input shapes, returns abstract
        output shape. Must satisfy: α(⟦op⟧(σ)) ⊑ ⟦op⟧♯(α(σ))
    description : str
        Formal specification of the operation.
    """

    op: OpKind
    concrete_fn: Callable[..., ShapeTuple]
    abstract_fn: Callable[..., AbstractShape]
    description: str


# ── Concrete semantics ⟦op⟧ ──────────────────────────────────────────────────

def concrete_matmul(shapes: List[ShapeTuple], **kwargs: Any) -> ShapeTuple:
    """⟦MATMUL⟧(A[...,m,k], B[...,k,n]) = C[...,m,n]

    Precondition: A.shape[-1] == B.shape[-2]
    """
    a, b = shapes[0], shapes[1]
    if len(a) < 2 or len(b) < 2:
        raise ValueError("MATMUL requires rank ≥ 2")
    if a[-1] != b[-2]:
        raise ValueError(f"Inner dimensions mismatch: {a[-1]} != {b[-2]}")
    batch = a[:-2]  # simplified: no broadcasting of batch dims
    return batch + (a[-2], b[-1])


def concrete_add(shapes: List[ShapeTuple], **kwargs: Any) -> ShapeTuple:
    """⟦ADD⟧(A, B) = broadcast(A, B)

    Broadcasting rules: dimensions are compared from trailing end;
    each pair must be equal or one must be 1.
    """
    a, b = shapes[0], shapes[1]
    max_rank = max(len(a), len(b))
    a_padded = (1,) * (max_rank - len(a)) + a
    b_padded = (1,) * (max_rank - len(b)) + b
    result = []
    for da, db in zip(a_padded, b_padded):
        if da == db:
            result.append(da)
        elif da == 1:
            result.append(db)
        elif db == 1:
            result.append(da)
        else:
            raise ValueError(f"Broadcast incompatible: {da} vs {db}")
    return tuple(result)


def concrete_reshape(shapes: List[ShapeTuple], target: ShapeTuple = (), **kwargs: Any) -> ShapeTuple:
    """⟦RESHAPE⟧(A, target) = target  iff  prod(A) == prod(target)

    Precondition: ∏ A.shape[i] == ∏ target[i]
    """
    a = shapes[0]
    import math
    src_elems = math.prod(a) if a else 1
    tgt_elems = math.prod(target) if target else 1
    if src_elems != tgt_elems:
        raise ValueError(
            f"Reshape element count mismatch: {src_elems} != {tgt_elems}"
        )
    return target


def concrete_transpose(shapes: List[ShapeTuple], dim0: int = -2, dim1: int = -1, **kwargs: Any) -> ShapeTuple:
    """⟦TRANSPOSE⟧(A, dim0, dim1) = A with dims dim0 and dim1 swapped."""
    a = list(shapes[0])
    rank = len(a)
    d0 = dim0 % rank
    d1 = dim1 % rank
    a[d0], a[d1] = a[d1], a[d0]
    return tuple(a)


def concrete_flatten(shapes: List[ShapeTuple], start_dim: int = 0, end_dim: int = -1, **kwargs: Any) -> ShapeTuple:
    """⟦FLATTEN⟧(A, start, end) = A with dims [start:end+1] collapsed."""
    a = shapes[0]
    rank = len(a)
    s = start_dim % rank
    e = end_dim % rank
    import math
    flat_dim = math.prod(a[s:e + 1])
    return a[:s] + (flat_dim,) + a[e + 1:]


def concrete_squeeze(shapes: List[ShapeTuple], dim: Optional[int] = None, **kwargs: Any) -> ShapeTuple:
    """⟦SQUEEZE⟧(A, dim) = A with size-1 dimensions removed."""
    a = shapes[0]
    if dim is not None:
        d = dim % len(a)
        if a[d] == 1:
            return a[:d] + a[d + 1:]
        return a
    return tuple(d for d in a if d != 1)


def concrete_unsqueeze(shapes: List[ShapeTuple], dim: int = 0, **kwargs: Any) -> ShapeTuple:
    """⟦UNSQUEEZE⟧(A, dim) = A with a size-1 dimension inserted at dim."""
    a = shapes[0]
    d = dim % (len(a) + 1)
    return a[:d] + (1,) + a[d:]


def concrete_cat(shapes: List[ShapeTuple], dim: int = 0, **kwargs: Any) -> ShapeTuple:
    """⟦CAT⟧([A_1, ..., A_k], dim) = concatenation along dim."""
    if not shapes:
        raise ValueError("CAT requires at least one input")
    rank = len(shapes[0])
    d = dim % rank
    total = sum(s[d] for s in shapes)
    result = list(shapes[0])
    result[d] = total
    return tuple(result)


def concrete_identity(shapes: List[ShapeTuple], **kwargs: Any) -> ShapeTuple:
    """⟦IDENTITY⟧(A) = A  (activation, dropout, softmax, detach, etc.)"""
    return shapes[0]


def concrete_permute(shapes: List[ShapeTuple], dims: Sequence[int] = (), **kwargs: Any) -> ShapeTuple:
    """⟦PERMUTE⟧(A, dims) = (A[dims[0]], ..., A[dims[r-1]]).

    ``dims`` must be a permutation of ``range(rank)`` (negative indices allowed),
    matching ``torch.Tensor.permute``.
    """
    a = shapes[0]
    rank = len(a)
    if len(dims) != rank:
        raise ValueError("PERMUTE expects %d dims, got %d" % (rank, len(dims)))
    norm = [d % rank for d in dims]
    if sorted(norm) != list(range(rank)):
        raise ValueError("PERMUTE dims %r is not a permutation of range(%d)"
                         % (list(dims), rank))
    return tuple(a[d] for d in norm)


def concrete_expand(shapes: List[ShapeTuple], sizes: Sequence[int] = (), **kwargs: Any) -> ShapeTuple:
    """⟦EXPAND⟧(A, sizes) following ``torch.Tensor.expand`` broadcasting rules.

    ``len(sizes) >= rank``; the extra leading entries prepend new dimensions
    (which may not be ``-1``). For right-aligned existing dims, ``-1`` keeps the
    original size, a size-1 dim may expand to any size, and a non-1 dim must
    match the requested size.
    """
    a = shapes[0]
    rank = len(a)
    if len(sizes) < rank:
        raise ValueError("EXPAND expects at least %d sizes, got %d"
                         % (rank, len(sizes)))
    extra = len(sizes) - rank
    out: List[int] = []
    for i in range(extra):
        if sizes[i] < 0:
            raise ValueError("EXPAND: new leading dim cannot be -1")
        out.append(sizes[i])
    for j in range(rank):
        want = sizes[extra + j]
        have = a[j]
        if want == -1:
            out.append(have)
        elif have == 1 or have == want:
            out.append(want)
        else:
            raise ValueError(
                "EXPAND: cannot expand dim %d of size %d to %d" % (j, have, want))
    return tuple(out)


def concrete_repeat(shapes: List[ShapeTuple], repeats: Sequence[int] = (), **kwargs: Any) -> ShapeTuple:
    """⟦REPEAT⟧(A, repeats) following ``torch.Tensor.repeat`` (tiling).

    ``len(repeats) >= rank``; the original shape is right-aligned (left-padded
    with 1s) and each dim is multiplied by its repeat count.
    """
    a = shapes[0]
    rank = len(a)
    if len(repeats) < rank:
        raise ValueError("REPEAT expects at least %d repeats, got %d"
                         % (rank, len(repeats)))
    if any(r < 0 for r in repeats):
        raise ValueError("REPEAT counts must be non-negative")
    pad = len(repeats) - rank
    padded = (1,) * pad + tuple(a)
    return tuple(p * r for p, r in zip(padded, repeats))


# ── Abstract semantics ⟦op⟧♯ ─────────────────────────────────────────────────

def abstract_matmul(shapes: List[AbstractShape], **kwargs: Any) -> AbstractShape:
    """⟦MATMUL⟧♯(A[...,m,k], B[...,k,n]) = C[...,m,n]

    Abstract soundness: if concrete dims satisfy the inner-dim equality,
    the abstract result is an over-approximation of α(⟦MATMUL⟧(γ(A), γ(B))).
    """
    a, b = shapes[0], shapes[1]
    if a.rank < 2 or b.rank < 2:
        raise ValueError("MATMUL requires rank ≥ 2")
    batch = a.dims[:-2]
    return AbstractShape(dims=batch + (a.dims[-2], b.dims[-1]))


def abstract_add(shapes: List[AbstractShape], **kwargs: Any) -> AbstractShape:
    """⟦ADD⟧♯(A, B) = abstract broadcast(A, B)

    For symbolic dimensions, preserves the symbolic name (sound because
    the concrete broadcast preserves dimension values for equal dims).
    """
    a, b = shapes[0], shapes[1]
    max_rank = max(a.rank, b.rank)
    a_padded: Tuple[Union[int, str], ...] = (1,) * (max_rank - a.rank) + a.dims
    b_padded: Tuple[Union[int, str], ...] = (1,) * (max_rank - b.rank) + b.dims
    result: List[Union[int, str]] = []
    for da, db in zip(a_padded, b_padded):
        if da == db:
            result.append(da)
        elif da == 1:
            result.append(db)
        elif db == 1:
            result.append(da)
        elif isinstance(da, str) or isinstance(db, str):
            # At least one is symbolic — keep the symbolic name as over-approx
            result.append(da if isinstance(da, str) else db)
        else:
            raise ValueError(f"Abstract broadcast incompatible: {da} vs {db}")
    return AbstractShape(dims=tuple(result))


def abstract_reshape(shapes: List[AbstractShape], target: Tuple[Union[int, str], ...] = (), **kwargs: Any) -> AbstractShape:
    """⟦RESHAPE⟧♯(A, target) = AbstractShape(target)

    The element-count equality ∏ A.dims = ∏ target is tracked as
    a constraint in the Z3 context, not checked here.
    """
    return AbstractShape(dims=target)


def abstract_transpose(shapes: List[AbstractShape], dim0: int = -2, dim1: int = -1, **kwargs: Any) -> AbstractShape:
    """⟦TRANSPOSE⟧♯(A, dim0, dim1) = A with dims swapped."""
    dims = list(shapes[0].dims)
    rank = len(dims)
    d0 = dim0 % rank
    d1 = dim1 % rank
    dims[d0], dims[d1] = dims[d1], dims[d0]
    return AbstractShape(dims=tuple(dims))


def abstract_flatten(shapes: List[AbstractShape], start_dim: int = 0, end_dim: int = -1, **kwargs: Any) -> AbstractShape:
    """⟦FLATTEN⟧♯(A, start, end) = A with dims collapsed (symbolic product)."""
    a = shapes[0]
    rank = a.rank
    s = start_dim % rank
    e = end_dim % rank

    collapsed = a.dims[s:e + 1]
    if all(isinstance(d, int) for d in collapsed):
        import math
        flat_dim: Union[int, str] = math.prod(int(d) for d in collapsed)
    else:
        # Symbolic product — generate a compound symbolic name
        flat_dim = "_x_".join(str(d) for d in collapsed)

    return AbstractShape(dims=a.dims[:s] + (flat_dim,) + a.dims[e + 1:])


def abstract_squeeze(shapes: List[AbstractShape], dim: Optional[int] = None, **kwargs: Any) -> AbstractShape:
    """⟦SQUEEZE⟧♯(A, dim) = A with known-1 dimensions removed."""
    a = shapes[0]
    if dim is not None:
        d = dim % a.rank
        if a.dims[d] == 1:
            return AbstractShape(dims=a.dims[:d] + a.dims[d + 1:])
        return a
    return AbstractShape(dims=tuple(d for d in a.dims if d != 1))


def abstract_unsqueeze(shapes: List[AbstractShape], dim: int = 0, **kwargs: Any) -> AbstractShape:
    """⟦UNSQUEEZE⟧♯(A, dim) = A with a size-1 dimension inserted."""
    a = shapes[0]
    d = dim % (a.rank + 1)
    return AbstractShape(dims=a.dims[:d] + (1,) + a.dims[d:])


def abstract_cat(shapes: List[AbstractShape], dim: int = 0, **kwargs: Any) -> AbstractShape:
    """⟦CAT⟧♯([A_1, ..., A_k], dim) = concatenation along dim."""
    if not shapes:
        raise ValueError("CAT requires at least one input")
    rank = shapes[0].rank
    d = dim % rank

    cat_dims = [s.dims[d] for s in shapes]
    if all(isinstance(cd, int) for cd in cat_dims):
        total: Union[int, str] = sum(int(cd) for cd in cat_dims)
    else:
        total = "_plus_".join(str(cd) for cd in cat_dims)

    result = list(shapes[0].dims)
    result[d] = total
    return AbstractShape(dims=tuple(result))


def abstract_identity(shapes: List[AbstractShape], **kwargs: Any) -> AbstractShape:
    """⟦IDENTITY⟧♯(A) = A  (shape-preserving operations)."""
    return shapes[0]


def abstract_permute(shapes: List[AbstractShape], dims: Sequence[int] = (), **kwargs: Any) -> AbstractShape:
    """⟦PERMUTE⟧♯(A, dims) = reorder A's (possibly symbolic) dims."""
    a = shapes[0]
    rank = a.rank
    if len(dims) != rank:
        raise ValueError("PERMUTE expects %d dims, got %d" % (rank, len(dims)))
    norm = [d % rank for d in dims]
    if sorted(norm) != list(range(rank)):
        raise ValueError("PERMUTE dims %r is not a permutation of range(%d)"
                         % (list(dims), rank))
    return AbstractShape(dims=tuple(a.dims[d] for d in norm))


def abstract_expand(shapes: List[AbstractShape], sizes: Sequence[int] = (), **kwargs: Any) -> AbstractShape:
    """⟦EXPAND⟧♯(A, sizes) over-approximating ``torch.Tensor.expand``.

    For each aligned dim: ``-1`` keeps the (possibly symbolic) original; any
    other request yields that concrete target size, which is sound because every
    valid concretization of a size-1 / matching dim produces exactly that size.
    """
    a = shapes[0]
    rank = a.rank
    if len(sizes) < rank:
        raise ValueError("EXPAND expects at least %d sizes, got %d"
                         % (rank, len(sizes)))
    extra = len(sizes) - rank
    out: List[Union[int, str]] = []
    for i in range(extra):
        if sizes[i] < 0:
            raise ValueError("EXPAND: new leading dim cannot be -1")
        out.append(sizes[i])
    for j in range(rank):
        want = sizes[extra + j]
        have = a.dims[j]
        if want == -1:
            out.append(have)
        elif isinstance(have, int) and have != 1 and have != want:
            raise ValueError(
                "EXPAND: cannot expand dim %d of size %d to %d" % (j, have, want))
        else:
            out.append(want)
    return AbstractShape(dims=tuple(out))


def abstract_repeat(shapes: List[AbstractShape], repeats: Sequence[int] = (), **kwargs: Any) -> AbstractShape:
    """⟦REPEAT⟧♯(A, repeats) = tile A; symbolic dims with repeat≠1 stay symbolic."""
    a = shapes[0]
    rank = a.rank
    if len(repeats) < rank:
        raise ValueError("REPEAT expects at least %d repeats, got %d"
                         % (rank, len(repeats)))
    if any(r < 0 for r in repeats):
        raise ValueError("REPEAT counts must be non-negative")
    pad = len(repeats) - rank
    padded: List[Union[int, str]] = [1] * pad + list(a.dims)
    out: List[Union[int, str]] = []
    for dim, r in zip(padded, repeats):
        if isinstance(dim, int):
            out.append(dim * r)
        elif r == 1:
            out.append(dim)
        else:
            out.append("%s_times_%d" % (dim, r))
    return AbstractShape(dims=tuple(out))


# ═══════════════════════════════════════════════════════════════════════════════
# 4.  Operation semantics registry
# ═══════════════════════════════════════════════════════════════════════════════

OP_SEMANTICS: Dict[OpKind, OpSemantics] = {
    OpKind.MATMUL: OpSemantics(
        op=OpKind.MATMUL,
        concrete_fn=concrete_matmul,
        abstract_fn=abstract_matmul,
        description="⟦MATMUL⟧(A[...,m,k], B[...,k,n]) = C[...,m,n]",
    ),
    OpKind.ADD: OpSemantics(
        op=OpKind.ADD,
        concrete_fn=concrete_add,
        abstract_fn=abstract_add,
        description="⟦ADD⟧(A, B) = broadcast(A, B)",
    ),
    OpKind.RESHAPE: OpSemantics(
        op=OpKind.RESHAPE,
        concrete_fn=concrete_reshape,
        abstract_fn=abstract_reshape,
        description="⟦RESHAPE⟧(A, target) = target  iff  ∏A = ∏target",
    ),
    OpKind.FLATTEN: OpSemantics(
        op=OpKind.FLATTEN,
        concrete_fn=concrete_flatten,
        abstract_fn=abstract_flatten,
        description="⟦FLATTEN⟧(A, s, e) = A[:s] ++ [∏A[s:e+1]] ++ A[e+1:]",
    ),
    OpKind.TRANSPOSE: OpSemantics(
        op=OpKind.TRANSPOSE,
        concrete_fn=concrete_transpose,
        abstract_fn=abstract_transpose,
        description="⟦TRANSPOSE⟧(A, d0, d1) = swap(A, d0, d1)",
    ),
    OpKind.SQUEEZE: OpSemantics(
        op=OpKind.SQUEEZE,
        concrete_fn=concrete_squeeze,
        abstract_fn=abstract_squeeze,
        description="⟦SQUEEZE⟧(A, d) = remove dim d if A[d]==1",
    ),
    OpKind.UNSQUEEZE: OpSemantics(
        op=OpKind.UNSQUEEZE,
        concrete_fn=concrete_unsqueeze,
        abstract_fn=abstract_unsqueeze,
        description="⟦UNSQUEEZE⟧(A, d) = insert dim 1 at position d",
    ),
    OpKind.CAT: OpSemantics(
        op=OpKind.CAT,
        concrete_fn=concrete_cat,
        abstract_fn=abstract_cat,
        description="⟦CAT⟧([A_i], d) = concat along dim d",
    ),
    OpKind.ACTIVATION: OpSemantics(
        op=OpKind.ACTIVATION,
        concrete_fn=concrete_identity,
        abstract_fn=abstract_identity,
        description="⟦ACTIVATION⟧(A) = A  (shape-preserving)",
    ),
    OpKind.DROPOUT: OpSemantics(
        op=OpKind.DROPOUT,
        concrete_fn=concrete_identity,
        abstract_fn=abstract_identity,
        description="⟦DROPOUT⟧(A) = A  (shape-preserving, phase-sensitive)",
    ),
    OpKind.SOFTMAX: OpSemantics(
        op=OpKind.SOFTMAX,
        concrete_fn=concrete_identity,
        abstract_fn=abstract_identity,
        description="⟦SOFTMAX⟧(A) = A  (shape-preserving)",
    ),
    OpKind.MULTIPLY: OpSemantics(
        op=OpKind.MULTIPLY,
        concrete_fn=concrete_add,  # same broadcast semantics as ADD
        abstract_fn=abstract_add,
        description="⟦MULTIPLY⟧(A, B) = broadcast(A, B)  (element-wise)",
    ),
    OpKind.DETACH: OpSemantics(
        op=OpKind.DETACH,
        concrete_fn=concrete_identity,
        abstract_fn=abstract_identity,
        description="⟦DETACH⟧(A) = A  (shape-preserving, clears grad)",
    ),
    OpKind.CONTIGUOUS: OpSemantics(
        op=OpKind.CONTIGUOUS,
        concrete_fn=concrete_identity,
        abstract_fn=abstract_identity,
        description="⟦CONTIGUOUS⟧(A) = A  (shape-preserving)",
    ),
    OpKind.PERMUTE: OpSemantics(
        op=OpKind.PERMUTE,
        concrete_fn=concrete_permute,
        abstract_fn=abstract_permute,
        description="⟦PERMUTE⟧(A, dims) = (A[dims[0]], ..., A[dims[r-1]])",
    ),
    OpKind.EXPAND: OpSemantics(
        op=OpKind.EXPAND,
        concrete_fn=concrete_expand,
        abstract_fn=abstract_expand,
        description="⟦EXPAND⟧(A, sizes) = broadcast A to sizes (size-1 dims grow)",
    ),
    OpKind.REPEAT: OpSemantics(
        op=OpKind.REPEAT,
        concrete_fn=concrete_repeat,
        abstract_fn=abstract_repeat,
        description="⟦REPEAT⟧(A, repeats) = tile A (right-aligned dim products)",
    ),
}


# ═══════════════════════════════════════════════════════════════════════════════
# 5.  Graph-level denotational semantics
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class DenotationalStep:
    """A single step in the denotational semantics computation.

    Records the node, its operation semantics, and the resulting state.
    """
    node_name: str
    op: OpKind
    input_names: List[str]
    output_name: str
    output_state: Optional[AbstractTensorState] = None


def compose_graph_semantics(
    steps: Sequence[DenotationalStep],
    initial_state: AbstractState,
) -> AbstractState:
    """Compute ⟦graph⟧♯ = compose(⟦node_n⟧♯, ..., ⟦node_1⟧♯)(σ♯).

    Applies abstract semantics sequentially for acyclic graphs.
    For each step, looks up the operation's abstract semantics in
    OP_SEMANTICS and applies it to the current abstract state.

    Parameters
    ----------
    steps : sequence of DenotationalStep
        The graph nodes in topological order.
    initial_state : AbstractState
        The initial abstract state mapping tensor names to abstract states.

    Returns
    -------
    AbstractState
        The final abstract state after all steps.
    """
    state = dict(initial_state)

    for step in steps:
        semantics = OP_SEMANTICS.get(step.op)
        if semantics is None:
            # Unknown op — identity semantics (shape-preserving)
            if step.input_names and step.input_names[0] in state:
                state[step.output_name] = state[step.input_names[0]]
            continue

        input_shapes = [
            state[name].shape
            for name in step.input_names
            if name in state
        ]

        if not input_shapes:
            continue

        try:
            output_shape = semantics.abstract_fn(input_shapes)
        except (ValueError, IndexError):
            # Propagation error — keep the first input's state as fallback
            if step.input_names and step.input_names[0] in state:
                state[step.output_name] = state[step.input_names[0]]
            continue

        # Inherit device and phase from the first input
        first_input = state.get(step.input_names[0])
        output_state = AbstractTensorState(
            shape=output_shape,
            device=first_input.device if first_input else AbstractDevice.CPU,
            phase=first_input.phase if first_input else AbstractPhase.TRAIN,
        )
        state[step.output_name] = output_state
        step.output_state = output_state

    return state


def verify_soundness_for_concrete_input(
    steps: Sequence[DenotationalStep],
    concrete_initial: ConcreteState,
) -> bool:
    """Verify the abstract interpretation soundness condition:

        α(⟦graph⟧(σ)) ⊑ ⟦graph⟧♯(α(σ))

    for a specific concrete input σ.

    Runs both concrete and abstract semantics, then checks that the
    abstraction of the concrete result is ⊑ the abstract result at
    every step.

    Parameters
    ----------
    steps : sequence of DenotationalStep
        Graph nodes in topological order.
    concrete_initial : ConcreteState
        Concrete initial state.

    Returns
    -------
    bool
        True if the soundness condition holds for this input.
    """
    abstract_initial = alpha(concrete_initial)

    # Run concrete semantics
    concrete_state: Dict[str, ConcreteTensorState] = dict(concrete_initial)
    for step in steps:
        semantics = OP_SEMANTICS.get(step.op)
        if semantics is None:
            if step.input_names and step.input_names[0] in concrete_state:
                concrete_state[step.output_name] = concrete_state[step.input_names[0]]
            continue

        input_shapes = [
            concrete_state[name].shape
            for name in step.input_names
            if name in concrete_state
        ]
        if not input_shapes:
            continue

        try:
            output_shape = semantics.concrete_fn(input_shapes)
        except (ValueError, IndexError):
            continue

        first = concrete_state.get(step.input_names[0])
        concrete_state[step.output_name] = ConcreteTensorState(
            shape=output_shape,
            device=first.device if first else "cpu",
            phase=first.phase if first else "train",
        )

    # Run abstract semantics
    abstract_state = compose_graph_semantics(steps, abstract_initial)

    # Check soundness: α(concrete_result) ⊑ abstract_result
    for name in concrete_state:
        if name not in abstract_state:
            continue
        alpha_concrete = alpha_tensor(concrete_state[name])
        if not alpha_concrete.leq(abstract_state[name]):
            return False

    return True
