"""Static verifier for ``einops`` rearrange / reduce / repeat patterns.

``einops`` patterns are a pervasive source of *silent* shape bugs in modern
transformer / diffusion code: a decomposition like ``rearrange(x, '(h w) c ->
h w c', h=8)`` raises an opaque ``EinopsError`` at runtime only when the actual
axis length is not divisible by 8 — a single-shape smoke test passes while a
different sequence length in production crashes.  PyTorch's own shape checker
never sees inside the ``einops`` call.

This module statically reproduces the *exact* set of conditions under which
real ``einops`` raises, **without executing the tensor op**, so the same fault
is caught at verification time.  It is differentially tested against the real
``einops`` package (``tests/test_einops_verify.py``) over a battery of valid and
invalid patterns: for every case the verdict (ok / error) and, when ok, the
output shape, must match what real ``einops`` does on a concrete tensor.

Public API::

    from src.einops_verify import verify_einops

    v = verify_einops("rearrange", "(h w) c -> h w c", (12, 3), h=4)
    assert v.ok and v.output_shape == (4, 3, 3)

    v = verify_einops("rearrange", "(h w) c -> h w c", (12, 3), h=5)
    assert not v.ok and v.error_kind == "non_divisible"

Symbolic dimensions (strings, e.g. ``"batch"``) are supported: divisibility
that cannot be decided statically yields ``ok=True`` with a symbolic output
dim rather than a false positive, preserving TensorGuard's soundness contract
(never refute a program that may be correct).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple, Union

Dim = Union[int, str]

__all__ = ["EinopsVerdict", "verify_einops", "EinopsParseError"]


class EinopsParseError(ValueError):
    """Raised when a pattern is malformed (mismatched parens, bad tokens)."""


@dataclass
class EinopsVerdict:
    """Result of statically verifying one einops call."""

    ok: bool
    output_shape: Optional[Tuple[Dim, ...]] = None
    error: Optional[str] = None
    # A stable machine-readable tag for the failure mode; ``None`` when ok.
    error_kind: Optional[str] = None
    # Resolved size for every named axis (concrete ints only).
    axes: Dict[str, int] = field(default_factory=dict)

    def __bool__(self) -> bool:  # pragma: no cover - convenience
        return self.ok


# ── pattern parsing ────────────────────────────────────────────────────────

# A token is a name, an anonymous constant (e.g. ``1``), an ellipsis, or a
# parenthesised group of names/constants.
_NAME = r"[A-Za-z_]\w*"
_ATOM = rf"{_NAME}|\d+|\.\.\.|\([^()]*\)"


@dataclass
class _Group:
    """A parenthesised axis group on one side of the pattern."""

    members: List[str]  # axis names / anonymous-constant sentinels


# Sentinel prefix used for anonymous integer axes (e.g. a literal ``1``).
_ANON = "\0anon:"


def _tokenize_side(side: str) -> List[Union[str, _Group, type(Ellipsis)]]:
    side = side.strip()
    if "(" in side:
        # reject nested / unbalanced parens early
        depth = 0
        for ch in side:
            if ch == "(":
                depth += 1
                if depth > 1:
                    raise EinopsParseError("nested parentheses are not allowed")
            elif ch == ")":
                depth -= 1
                if depth < 0:
                    raise EinopsParseError("unbalanced parentheses")
        if depth != 0:
            raise EinopsParseError("unbalanced parentheses")

    out: List[Union[str, _Group, type(Ellipsis)]] = []
    for m in re.finditer(_ATOM, side):
        tok = m.group(0)
        if tok == "...":
            out.append(Ellipsis)
        elif tok.startswith("("):
            inner = tok[1:-1].strip()
            members: List[str] = []
            for sub in re.finditer(rf"{_NAME}|\d+|\.\.\.", inner):
                s = sub.group(0)
                if s == "...":
                    raise EinopsParseError("ellipsis inside parentheses")
                members.append(_ANON + s if s.isdigit() else s)
            out.append(_Group(members))
        elif tok.isdigit():
            out.append(_ANON + tok)
        else:
            out.append(tok)
    return out


def _parse(pattern: str):
    if pattern.count("->") != 1:
        raise EinopsParseError("pattern must contain exactly one '->'")
    lhs_s, rhs_s = pattern.split("->")
    return _tokenize_side(lhs_s), _tokenize_side(rhs_s)


# ── helpers ────────────────────────────────────────────────────────────────


def _flatten_names(side) -> List[str]:
    """All *named* (non-anonymous, non-ellipsis) axes on a side, in order."""
    names: List[str] = []
    for tok in side:
        if tok is Ellipsis:
            continue
        if isinstance(tok, _Group):
            names.extend(m for m in tok.members if not m.startswith(_ANON))
        elif not tok.startswith(_ANON):
            names.append(tok)
    return names


def _is_concrete(v: Dim) -> bool:
    return isinstance(v, int)


# ── the verifier ───────────────────────────────────────────────────────────


def verify_einops(
    op: str,
    pattern: str,
    input_shape: Sequence[Dim],
    **axes_lengths: int,
) -> EinopsVerdict:
    """Statically verify one ``einops`` call.

    Parameters
    ----------
    op:
        ``"rearrange"``, ``"reduce"`` or ``"repeat"``.
    pattern:
        The einops pattern string, e.g. ``"b (h w) c -> b h w c"``.
    input_shape:
        Concrete ints and/or symbolic dim names.
    axes_lengths:
        Known axis sizes (the ``h=8`` style kwargs).
    """
    if op not in ("rearrange", "reduce", "repeat"):
        raise ValueError(f"unknown einops op: {op!r}")

    try:
        lhs, rhs = _parse(pattern)
    except EinopsParseError as e:
        return EinopsVerdict(False, error=str(e), error_kind="parse")

    input_shape = list(input_shape)

    # ── ellipsis bookkeeping ────────────────────────────────────────────
    lhs_has_ell = any(t is Ellipsis for t in lhs)
    rhs_has_ell = any(t is Ellipsis for t in rhs)
    if rhs_has_ell and not lhs_has_ell:
        return EinopsVerdict(
            False,
            error="ellipsis on the right but not the left",
            error_kind="ellipsis",
        )

    # Number of top-level axis positions the LHS consumes (group/name/anon = 1).
    lhs_positions = [t for t in lhs if t is not Ellipsis]
    n_fixed = len(lhs_positions)
    if lhs_has_ell:
        if len(input_shape) < n_fixed:
            return EinopsVerdict(
                False,
                error=(
                    f"pattern expects at least {n_fixed} dims, "
                    f"got {len(input_shape)}"
                ),
                error_kind="rank_mismatch",
            )
    else:
        if len(input_shape) != n_fixed:
            return EinopsVerdict(
                False,
                error=(
                    f"pattern expects {n_fixed} dims, got {len(input_shape)}"
                ),
                error_kind="rank_mismatch",
            )

    # Split the input dims into (before-ellipsis, ellipsis-block, after).
    ell_dims: List[Dim] = []
    if lhs_has_ell:
        ell_index = next(i for i, t in enumerate(lhs) if t is Ellipsis)
        before = ell_index
        after = len(lhs) - ell_index - 1
        ell_dims = input_shape[before:len(input_shape) - after]
        fixed_dims = input_shape[:before] + input_shape[len(input_shape) - after:]
    else:
        fixed_dims = input_shape

    # ── solve axis sizes from the LHS ───────────────────────────────────
    axis_size: Dict[str, Dim] = {}
    for name, val in axes_lengths.items():
        axis_size[name] = int(val)

    dim_iter = iter(fixed_dims)
    for tok in lhs_positions:
        dim = next(dim_iter)
        if isinstance(tok, _Group):
            verdict = _solve_group(tok, dim, axis_size)
            if verdict is not None:
                return verdict
        elif tok.startswith(_ANON):
            want = int(tok[len(_ANON):])
            if _is_concrete(dim) and dim != want:
                return EinopsVerdict(
                    False,
                    error=f"anonymous axis expected {want}, got {dim}",
                    error_kind="anon_mismatch",
                )
        else:
            if tok in axis_size and _is_concrete(axis_size[tok]) \
                    and _is_concrete(dim) and axis_size[tok] != dim:
                return EinopsVerdict(
                    False,
                    error=(
                        f"axis '{tok}' was given length "
                        f"{axis_size[tok]} but tensor dim is {dim}"
                    ),
                    error_kind="length_mismatch",
                )
            axis_size[tok] = dim

    # ── identifier-set checks per op ────────────────────────────────────
    lhs_names = set(_flatten_names(lhs))
    rhs_names = set(_flatten_names(rhs))

    # duplicate-axis detection (einops forbids repeats on a side)
    dup = _first_duplicate(_flatten_names(lhs))
    if dup is not None:
        return EinopsVerdict(
            False, error=f"axis '{dup}' repeated on the left",
            error_kind="duplicate",
        )
    dup = _first_duplicate(_flatten_names(rhs))
    if dup is not None:
        return EinopsVerdict(
            False, error=f"axis '{dup}' repeated on the right",
            error_kind="duplicate",
        )

    if op == "rearrange":
        if lhs_names != rhs_names:
            extra = rhs_names ^ lhs_names
            return EinopsVerdict(
                False,
                error=(
                    "identifiers must appear on both sides of rearrange; "
                    f"offending: {sorted(extra)}"
                ),
                error_kind="axis_set_mismatch",
            )
    elif op == "reduce":
        if not rhs_names <= lhs_names:
            extra = sorted(rhs_names - lhs_names)
            return EinopsVerdict(
                False,
                error=f"reduce introduces new axes on the right: {extra}",
                error_kind="axis_set_mismatch",
            )
    else:  # repeat
        if not lhs_names <= rhs_names:
            missing = sorted(lhs_names - rhs_names)
            return EinopsVerdict(
                False,
                error=f"repeat drops axes that must be kept: {missing}",
                error_kind="axis_set_mismatch",
            )
        # new axes introduced by repeat must have a known length
        for name in rhs_names - lhs_names:
            if name not in axis_size:
                return EinopsVerdict(
                    False,
                    error=f"repeat axis '{name}' needs an explicit length",
                    error_kind="missing_length",
                )

    # ── build the output shape from the RHS ─────────────────────────────
    out: List[Dim] = []
    for tok in rhs:
        if tok is Ellipsis:
            out.extend(ell_dims)
            continue
        if isinstance(tok, _Group):
            prod: Dim = 1
            for m in tok.members:
                size = _member_size(m, axis_size)
                if size is None:
                    return EinopsVerdict(
                        False,
                        error=f"axis '{m}' has unknown length",
                        error_kind="missing_length",
                    )
                prod = _mul(prod, size)
            out.append(prod)
        elif tok.startswith(_ANON):
            out.append(int(tok[len(_ANON):]))
        else:
            size = axis_size.get(tok)
            if size is None:
                return EinopsVerdict(
                    False,
                    error=f"axis '{tok}' has unknown length",
                    error_kind="missing_length",
                )
            out.append(size)

    return EinopsVerdict(
        True,
        output_shape=tuple(out),
        axes={k: v for k, v in axis_size.items() if _is_concrete(v)},
    )


def _solve_group(
    group: _Group, dim: Dim, axis_size: Dict[str, Dim]
) -> Optional[EinopsVerdict]:
    """Resolve the sizes of a parenthesised LHS group consuming ``dim``.

    Returns an error verdict on a real fault, otherwise mutates ``axis_size``
    and returns ``None``.
    """
    known_prod = 1
    unknown: List[str] = []
    symbolic = not _is_concrete(dim)
    for m in group.members:
        if m.startswith(_ANON):
            known_prod *= int(m[len(_ANON):])
            continue
        size = axis_size.get(m)
        if size is None:
            unknown.append(m)
        elif _is_concrete(size):
            known_prod = _mul(known_prod, size)
        else:
            symbolic = True

    if len(unknown) > 1:
        return EinopsVerdict(
            False,
            error=(
                f"could not infer sizes for axes {unknown}; "
                "give all but one an explicit length"
            ),
            error_kind="underdetermined",
        )

    if symbolic or not _is_concrete(known_prod):
        # Cannot decide divisibility statically: stay sound (do not refute).
        for m in unknown:
            axis_size[m] = f"({dim}//{known_prod})" if known_prod != 1 else dim
        return None

    if len(unknown) == 0:
        if _is_concrete(dim) and dim != known_prod:
            return EinopsVerdict(
                False,
                error=(
                    f"shape mismatch: group product {known_prod} "
                    f"!= axis length {dim}"
                ),
                error_kind="length_mismatch",
            )
        return None

    # exactly one unknown: it must divide evenly
    (name,) = unknown
    if known_prod == 0 or dim % known_prod != 0:
        return EinopsVerdict(
            False,
            error=(
                f"can't divide axis of length {dim} into chunks "
                f"of {known_prod} (axis '{name}')"
            ),
            error_kind="non_divisible",
        )
    axis_size[name] = dim // known_prod
    return None


def _member_size(m: str, axis_size: Dict[str, Dim]) -> Optional[Dim]:
    if m.startswith(_ANON):
        return int(m[len(_ANON):])
    return axis_size.get(m)


def _mul(a: Dim, b: Dim) -> Dim:
    if _is_concrete(a) and _is_concrete(b):
        return a * b
    return f"({a}*{b})"


def _first_duplicate(names: List[str]) -> Optional[str]:
    seen = set()
    for n in names:
        if n in seen:
            return n
        seen.add(n)
    return None
