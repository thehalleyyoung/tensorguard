"""v5 / Track-C — QKV unpacking & multi-assign destructuring.

A small, focused library that knows how to compute the shapes of the
tuple of tensors produced by the four idiomatic patterns used to split a
fused projection into Q, K and V::

    q, k, v = qkv.split(d, dim=-1)
    q, k, v = qkv.chunk(3, dim=-1)
    q, k, v = qkv.view(B, T, 3, H, D).unbind(2)
    q, k, v = einops.rearrange(qkv, 'b t (three h d) -> three b h t d',
                               three=3, h=H)

These are the four patterns we observed in BERT, GPT-2, ViT, T5 and
Llama transformer blocks.  Each helper takes the shape of the *input*
fused tensor (and any constant arguments) and returns a ``list`` of
output :class:`TensorShape`s.  When a constraint cannot be discharged
syntactically we record a textual reason on the optional
:class:`UnpackResult` rather than raising, so the analyzer can decide
whether to abstain.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple, Union

from src.tensor_shapes import ShapeDim, TensorShape  # do not edit, only import
from src.v5.symbolic_config import SymInt, SymExpr, sym_to_dim


@dataclass
class UnpackResult:
    """Outcome of attempting to destructure a fused tensor."""
    shapes: List[TensorShape] = field(default_factory=list)
    ok: bool = True
    reason: str = ""

    def __iter__(self):
        return iter(self.shapes)

    def __len__(self) -> int:  # noqa: D401
        return len(self.shapes)


# ────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ────────────────────────────────────────────────────────────────────────────

def _norm_dim(ndim: int, dim: int) -> int:
    if dim < 0:
        return ndim + dim
    return dim


def _dim_value(d: ShapeDim) -> Union[int, str]:
    return d.value


def _is_concrete(d: ShapeDim) -> bool:
    return isinstance(d.value, int)


# ────────────────────────────────────────────────────────────────────────────
# x.split(size_or_sizes, dim)
# ────────────────────────────────────────────────────────────────────────────

def unpack_split(
    input_shape: TensorShape,
    size_or_sizes: Union[int, Sequence[int]],
    dim: int = 0,
) -> UnpackResult:
    """Mirror ``torch.Tensor.split``.

    * If ``size_or_sizes`` is an int, the tensor is split into chunks of
      that exact size along ``dim``; the last chunk may be smaller when
      the dim is not divisible.
    * If ``size_or_sizes`` is a sequence of ints, the dim is split into
      chunks of those exact sizes (must sum to the dim's size when
      concrete).
    """
    d = _norm_dim(input_shape.ndim, dim)
    if not 0 <= d < input_shape.ndim:
        return UnpackResult(ok=False, reason=f"dim {dim} OOB for ndim {input_shape.ndim}")
    target = input_shape.dims[d]

    if isinstance(size_or_sizes, int):
        size = size_or_sizes
        if size <= 0:
            return UnpackResult(ok=False, reason="split size must be positive")
        if _is_concrete(target):
            total = int(target.value)
            n_full, remainder = divmod(total, size)
            sizes: List[int] = [size] * n_full + ([remainder] if remainder else [])
        else:
            # Symbolic dim: we can't know how many chunks at static time,
            # so abstain unless size == 1 (degenerate).
            return UnpackResult(
                ok=False,
                reason=f"cannot split symbolic dim {target.value} by size {size}",
            )
    else:
        sizes = list(size_or_sizes)
        if any(s <= 0 for s in sizes):
            return UnpackResult(ok=False, reason="negative split size")
        if _is_concrete(target):
            if sum(sizes) != int(target.value):
                return UnpackResult(
                    ok=False,
                    reason=f"split sizes {sizes} sum to {sum(sizes)} ≠ dim {target.value}",
                )

    out: List[TensorShape] = []
    for s in sizes:
        new_dims = list(input_shape.dims)
        new_dims[d] = ShapeDim(s)
        out.append(TensorShape(tuple(new_dims)))
    return UnpackResult(shapes=out, ok=True)


# ────────────────────────────────────────────────────────────────────────────
# x.chunk(chunks, dim)
# ────────────────────────────────────────────────────────────────────────────

def unpack_chunk(
    input_shape: TensorShape,
    chunks: int,
    dim: int = 0,
) -> UnpackResult:
    """Mirror ``torch.Tensor.chunk``.

    PyTorch divides ``dim`` into ``chunks`` *roughly equal* chunks; chunk
    sizes are ``ceil(dim/chunks)`` for the first ``k`` chunks and the
    remainder for the last (which may be 0 — in which case fewer chunks
    are returned).  For QKV the typical case is ``chunks=3`` and
    ``dim`` divisible by 3, which we model exactly.
    """
    if chunks <= 0:
        return UnpackResult(ok=False, reason="chunks must be positive")
    d = _norm_dim(input_shape.ndim, dim)
    if not 0 <= d < input_shape.ndim:
        return UnpackResult(ok=False, reason=f"dim {dim} OOB for ndim {input_shape.ndim}")
    target = input_shape.dims[d]

    if _is_concrete(target):
        total = int(target.value)
        # PyTorch: chunk_size = ceil(total / chunks)
        chunk_size = -(-total // chunks)
        if chunk_size == 0:
            chunk_size = 1
        sizes: List[int] = []
        remaining = total
        while remaining > 0 and len(sizes) < chunks:
            take = min(chunk_size, remaining)
            sizes.append(take)
            remaining -= take
        out = []
        for s in sizes:
            new_dims = list(input_shape.dims)
            new_dims[d] = ShapeDim(s)
            out.append(TensorShape(tuple(new_dims)))
        return UnpackResult(shapes=out, ok=True)

    # Symbolic dim: only handle the divisible case symbolically.
    # Result shape uses (dim // chunks) along d.  Encode as a string.
    new_size = f"({target.value}//{chunks})"
    out = []
    for _ in range(chunks):
        new_dims = list(input_shape.dims)
        new_dims[d] = ShapeDim(new_size)
        out.append(TensorShape(tuple(new_dims)))
    # Mark a soft warning so callers can require divisibility.
    return UnpackResult(
        shapes=out, ok=True,
        reason=f"requires {target.value} % {chunks} == 0",
    )


# ────────────────────────────────────────────────────────────────────────────
# x.unbind(dim)
# ────────────────────────────────────────────────────────────────────────────

def unpack_unbind(
    input_shape: TensorShape,
    dim: int = 0,
) -> UnpackResult:
    """Remove ``dim`` and return one tensor per slice.

    The number of returned tensors equals the size of ``dim``.  When dim
    is symbolic we cannot know the count at static time, but the *common*
    QKV pattern is ``.view(B, T, 3, H, D).unbind(2)`` — i.e. dim is a
    literal 3.  We support that case directly.
    """
    d = _norm_dim(input_shape.ndim, dim)
    if not 0 <= d < input_shape.ndim:
        return UnpackResult(ok=False, reason=f"dim {dim} OOB for ndim {input_shape.ndim}")
    target = input_shape.dims[d]
    if not _is_concrete(target):
        return UnpackResult(
            ok=False,
            reason=f"unbind on symbolic dim {target.value} — count unknown",
        )
    n = int(target.value)
    new_dims = tuple(d_ for i, d_ in enumerate(input_shape.dims) if i != d)
    return UnpackResult(shapes=[TensorShape(new_dims) for _ in range(n)], ok=True)


# ────────────────────────────────────────────────────────────────────────────
# einops.rearrange — restricted patterns
# ────────────────────────────────────────────────────────────────────────────

# We only need to handle the common QKV / multihead patterns.  A general
# einops parser is overkill; we implement just enough.

_AXIS = re.compile(r"[A-Za-z_][A-Za-z_0-9]*")


def _parse_einops_side(side: str) -> List[Union[str, List[str]]]:
    """Parse one half of a rearrange pattern into a flat list of either
    plain axis names or lists representing parenthesised groups.
    Whitespace separates top-level tokens.  Example::

        'b t (three h d)' → ['b', 't', ['three', 'h', 'd']]
    """
    side = side.strip()
    out: List[Union[str, List[str]]] = []
    i = 0
    while i < len(side):
        c = side[i]
        if c.isspace():
            i += 1
            continue
        if c == "(":
            j = side.index(")", i)
            inner = _AXIS.findall(side[i + 1: j])
            out.append(inner)
            i = j + 1
        else:
            m = _AXIS.match(side, i)
            if not m:
                raise ValueError(f"unparsed einops token at {i}: {side!r}")
            out.append(m.group(0))
            i = m.end()
    return out


def parse_einops_rearrange(
    pattern: str,
    input_shape: TensorShape,
    axes_lengths: Optional[dict] = None,
) -> UnpackResult:
    """Compute the output shape of ``einops.rearrange(x, pattern, **axes_lengths)``.

    Only patterns whose left-hand-side has at most one parenthesised
    group per axis position are supported, which is sufficient for the
    QKV idioms enumerated in the module docstring.

    The result is wrapped in an :class:`UnpackResult` whose ``shapes``
    list contains a *single* output shape — unless the leading axis of
    the rhs is a literal small constant (like ``three``), in which case
    we emit one shape per slice (mirroring ``unbind`` on that axis).
    """
    if "->" not in pattern:
        return UnpackResult(ok=False, reason="missing '->' in pattern")
    lhs_str, rhs_str = pattern.split("->", 1)
    try:
        lhs = _parse_einops_side(lhs_str)
        rhs = _parse_einops_side(rhs_str)
    except ValueError as e:
        return UnpackResult(ok=False, reason=str(e))

    if len(lhs) != input_shape.ndim:
        return UnpackResult(
            ok=False,
            reason=f"lhs has {len(lhs)} axes, tensor has {input_shape.ndim}",
        )

    axes_lengths = dict(axes_lengths or {})
    # Bind axis names from the lhs to dims.
    bindings: dict = {}
    for tok, dim in zip(lhs, input_shape.dims):
        if isinstance(tok, str):
            bindings[tok] = dim
        else:
            # Parenthesised group: factor the dim by the known axis lengths.
            known = {n: axes_lengths[n] for n in tok if n in axes_lengths}
            unknown = [n for n in tok if n not in known]
            if _is_concrete(dim):
                total = int(dim.value)
                k_prod = 1
                for v in known.values():
                    k_prod *= int(v)
                if k_prod == 0 or total % k_prod != 0:
                    return UnpackResult(
                        ok=False,
                        reason=(f"axis {tok} group product {k_prod} does not "
                                f"divide concrete dim {total}"),
                    )
                if len(unknown) == 0:
                    if total != k_prod:
                        return UnpackResult(
                            ok=False,
                            reason=(f"group {tok} expected {k_prod}, got {total}"),
                        )
                elif len(unknown) == 1:
                    bindings[unknown[0]] = ShapeDim(total // k_prod)
                else:
                    # Underdetermined; mark each unknown as symbolic.
                    for n in unknown:
                        bindings[n] = ShapeDim(n)
                for n, v in known.items():
                    bindings[n] = ShapeDim(int(v))
            else:
                # Symbolic dim: keep symbolic for unknowns, set knowns concretely.
                for n, v in known.items():
                    bindings[n] = ShapeDim(int(v))
                for n in unknown:
                    bindings.setdefault(n, ShapeDim(n))

    # Build output shape(s).
    out_dims: List[ShapeDim] = []
    for tok in rhs:
        if isinstance(tok, str):
            if tok not in bindings:
                if tok in axes_lengths:
                    bindings[tok] = ShapeDim(int(axes_lengths[tok]))
                else:
                    return UnpackResult(ok=False, reason=f"axis {tok} unbound")
            out_dims.append(bindings[tok])
        else:
            # Output-side group: multiply.
            prod_concrete = 1
            sym_parts: List[str] = []
            all_concrete = True
            for n in tok:
                if n not in bindings:
                    return UnpackResult(ok=False, reason=f"axis {n} unbound")
                d = bindings[n]
                if _is_concrete(d):
                    prod_concrete *= int(d.value)
                else:
                    all_concrete = False
                    sym_parts.append(str(d.value))
            if all_concrete:
                out_dims.append(ShapeDim(prod_concrete))
            else:
                expr = "*".join([str(prod_concrete)] + sym_parts) if prod_concrete != 1 \
                       else "*".join(sym_parts)
                out_dims.append(ShapeDim(f"({expr})"))

    full_shape = TensorShape(tuple(out_dims))

    # If the leading rhs axis is a small known constant, emit one shape
    # per slice (this is the QKV "three b h t d" idiom).
    head = rhs[0]
    if isinstance(head, str) and head in bindings:
        head_dim = bindings[head]
        if _is_concrete(head_dim) and 1 <= int(head_dim.value) <= 8:
            n = int(head_dim.value)
            tail = full_shape.dims[1:]
            return UnpackResult(
                shapes=[TensorShape(tail) for _ in range(n)],
                ok=True,
                reason=f"split along leading axis {head}={n}",
            )
    return UnpackResult(shapes=[full_shape], ok=True)


# ────────────────────────────────────────────────────────────────────────────
# Convenience: detect & compute the QKV destructuring from a fused tensor
# ────────────────────────────────────────────────────────────────────────────

def split_qkv(input_shape: TensorShape, num_heads: Union[int, SymInt],
              head_dim: Union[int, SymInt],
              fused_dim: int = -1) -> UnpackResult:
    """Specialised helper: produces the ``(B, H, T, D)`` shape of each of
    Q, K, V given the fused QKV tensor and head metadata.

    Supports the canonical ``(B, T, 3*H*D)`` layout.
    """
    d = _norm_dim(input_shape.ndim, fused_dim)
    target = input_shape.dims[d]
    # Concrete validation if we can.
    if _is_concrete(target) and isinstance(num_heads, int) and isinstance(head_dim, int):
        expected = 3 * num_heads * head_dim
        if int(target.value) != expected:
            return UnpackResult(
                ok=False,
                reason=f"fused dim {target.value} ≠ 3*H*D = {expected}",
            )
    # Output: drop fused dim, append (H, D).
    front = list(input_shape.dims[:d]) + list(input_shape.dims[d + 1:])
    H = sym_to_dim(num_heads)
    D = sym_to_dim(head_dim)
    out_shape = TensorShape(tuple(front + [H, D]))
    return UnpackResult(shapes=[out_shape, out_shape, out_shape], ok=True,
                        reason="qkv split")


__all__ = [
    "UnpackResult",
    "unpack_split",
    "unpack_chunk",
    "unpack_unbind",
    "parse_einops_rearrange",
    "split_qkv",
]
