"""SMT-synthesized fix proposals (``SMT_FIX_EXPANSION_STEPS`` Phase 0 + Phase 1).

The pattern strategies in :mod:`src.symexec.autofix` *guess* a canonical edit.
This module makes the **proposal** step as smart as the detector: it poses the
repair as a small constraint-satisfaction query, hands it to the same Z3 bridge
the prover uses (:mod:`src.symexec.smt_bridge`), and reads the fixing value off a
*satisfying model*.  Every proposal it returns is still handed back to
``autofix.verify_fix`` (re-run the engine; the targeted bug must be gone and no
new bug **kind** may appear), so synthesis only *proposes* — the re-verification
gate remains the sole authority on soundness.

Design contract (mirrors the pattern strategies so these plug straight into
``autofix._STRATEGIES``):

* a synthesizer takes ``(lines: List[str], bug)`` and returns
  ``(patched_source, strategy_name, description)`` or ``None``;
* it returns ``None`` on *any* ambiguity (≥2 equally-minimal fixes, a target
  token that is not uniquely locatable) or when Z3 is unavailable / ``unknown``;
* it never mutates analysis state, so it cannot affect which bugs report on the
  original program nor the analysis fingerprint.
"""

from __future__ import annotations

import re
from typing import List, Optional, Tuple

from . import smt_bridge as _smt
from .symdim import SymDim

__all__ = [
    "parse_shape",
    "parse_two_shapes",
    "solve_inferred_factor",
    "synth_reshape_target",
    "synth_reshape_fix",
    "synth_matmul_fix",
]


# --------------------------------------------------------------------------- #
# Phase 0 — synthesis primitives.                                              #
# --------------------------------------------------------------------------- #
_SHAPE_RE = re.compile(r"\(([-\d,\s]*)\)")


def parse_shape(text: str) -> Optional[Tuple[int, ...]]:
    """Parse the first ``(d0, d1, ...)`` shape tuple in ``text`` into ints, or
    ``None`` if it contains a non-integer (symbolic) entry."""
    m = _SHAPE_RE.search(text)
    if not m:
        return None
    body = m.group(1).strip()
    if not body:
        return tuple()
    out: List[int] = []
    for part in body.split(","):
        part = part.strip()
        if part == "":
            continue
        try:
            out.append(int(part))
        except ValueError:
            return None
    return tuple(out)


def parse_two_shapes(text: str) -> Optional[Tuple[Tuple[int, ...], Tuple[int, ...]]]:
    """Parse the first two ``(...)`` shape tuples in ``text``; ``None`` if either
    is missing or symbolic."""
    shapes: List[Tuple[int, ...]] = []
    for m in _SHAPE_RE.finditer(text):
        body = m.group(1).strip()
        parsed = parse_shape("(" + body + ")")
        if parsed is None:
            return None
        shapes.append(parsed)
        if len(shapes) == 2:
            return shapes[0], shapes[1]
    return None


def solve_inferred_factor(prod_others: int, numel: int) -> Optional[int]:
    """Solve ``prod_others * f == numel`` for a positive integer ``f`` using the
    Z3 dimension bridge, returning ``f`` when it exists and is unique, else
    ``None``.

    Phrasing this through the solver (rather than a bare ``numel % prod_others``)
    keeps the primitive correct when ``numel`` becomes a *symbolic* dimension
    product in future callers: ``f`` is then read off the satisfying model the
    same way.  ``prod_others`` is the product of the reshape factors we keep; a
    PyTorch ``-1`` in that position is exactly this inferred ``f``.
    """
    if prod_others <= 0 or numel < 0:
        return None
    if not _smt.Z3_AVAILABLE:
        # Sound fallback: only succeed on an exact concrete division.
        return numel // prod_others if numel % prod_others == 0 else None
    f = SymDim.var("_f")
    cons = [_smt.eq(f * prod_others, numel), _smt.ge(f, 1)]
    if _smt.check(cons) != "sat":
        return None
    model = _smt.model(cons)
    if not model or "_f" not in model:
        return None
    cand = model["_f"]
    # Uniqueness: f is pinned iff `f != cand` is unsatisfiable under the system.
    pinned = _smt.check(cons + [_smt.ne(SymDim.var("_f"), cand)]) == "unsat"
    return cand if pinned else None


# --------------------------------------------------------------------------- #
# Phase 1 R1 — reshape: solver-chosen target (preserve the user's intent).     #
# --------------------------------------------------------------------------- #
def synth_reshape_target(
    numel: int, target: Tuple[int, ...]
) -> Optional[Tuple[int, ...]]:
    """Given a tensor element count ``numel`` and the user's incompatible
    ``reshape``/``view`` ``target``, return a *minimally edited* target that is
    numel-compatible — exactly one factor replaced by ``-1`` (PyTorch infers it)
    — or ``None`` when no single-factor repair is unambiguous.

    The synthesized fix preserves every dimension the user intended and only
    relaxes the single position whose kept-product divides ``numel``.  When two
    or more positions qualify, or none does, we abstain (the caller's blunt
    flatten-to-``-1`` remains as a fallback).
    """
    if not target or -1 in target:
        return None
    n = len(target)
    full = 1
    for d in target:
        if d <= 0:
            return None
        full *= d
    if full == numel:
        return None  # not actually a mismatch
    candidates: List[int] = []
    for j in range(n):
        prod_others = 1
        for i, d in enumerate(target):
            if i != j:
                prod_others *= d
        f = solve_inferred_factor(prod_others, numel)
        # Only a *real* edit: the inferred factor must differ from what's there.
        if f is not None and f != target[j]:
            candidates.append(j)
    if len(candidates) != 1:
        return None
    j = candidates[0]
    return tuple(-1 if i == j else d for i, d in enumerate(target))


_RESHAPE_CALL = re.compile(r"\.(reshape|view)\(\s*([^()]*?)\s*\)")
_NUMEL_RE = re.compile(r"tensor of\s+(\d+)\s+elements")


def synth_reshape_fix(lines: List[str], bug) -> Optional[Tuple[str, str, str]]:
    """Synthesizer for ``reshape_size_mismatch``: rewrite the offending
    ``.reshape(...)``/``.view(...)`` call to a solver-chosen, minimally edited,
    numel-compatible target.  Returns ``None`` when the smart repair is
    ambiguous (the caller falls back to flatten-to-``-1``)."""
    i = bug.line - 1
    if not (0 <= i < len(lines)):
        return None
    line = lines[i]
    call = _RESHAPE_CALL.search(line)
    if not call:
        return None
    target = parse_shape("(" + call.group(2) + ")")
    numel_m = _NUMEL_RE.search(getattr(bug, "message", "") or "")
    if target is None or numel_m is None:
        return None
    numel = int(numel_m.group(1))
    new_target = synth_reshape_target(numel, target)
    if new_target is None:
        return None
    new_args = ", ".join(str(d) for d in new_target)
    new_call = f".{call.group(1)}({new_args})"
    new_line = line[: call.start()] + new_call + line[call.end():]
    if new_line == line:
        return None
    patched = lines[:]
    patched[i] = new_line
    return (
        "\n".join(patched),
        "reshape-synth-target",
        f"solver-synthesized reshape target {new_target}: keep the dimensions "
        "you intended and infer the single mismatched factor with `-1`",
    )


# --------------------------------------------------------------------------- #
# Phase 1 R2 — matmul: synthesize a transpose when the operand is stored      #
# transposed (contracted dim equals the other axis).                          #
# --------------------------------------------------------------------------- #
_MATMUL_OP = re.compile(
    r"^(?P<pre>.*?)(?P<lhs>[A-Za-z_][\w.]*)\s*@\s*(?P<rhs>[A-Za-z_][\w.]*)"
    r"(?P<post>.*)$"
)


def _transpose_fixes_contraction(a: Tuple[int, ...], b: Tuple[int, ...]) -> bool:
    """Does transposing ``b``'s last two axes make ``a @ b`` contract-compatible?

    ``a @ b`` requires ``a[-1] == b[-2]``.  Transposing ``b`` swaps its last two
    axes, so the new requirement is ``a[-1] == b[-1]``.  We confirm the *original*
    is a genuine mismatch and the *transposed* form is compatible, routing the
    check through the Z3 bridge so the reasoning extends to symbolic dims.
    """
    if len(a) < 1 or len(b) < 2:
        return False
    k = SymDim.const_dim(a[-1])
    p = SymDim.const_dim(b[-2])
    q = SymDim.const_dim(b[-1])
    mismatch = _smt.check([_smt.ne(k, p)]) == "sat" and (
        _smt.check([_smt.eq(k, p)]) == "unsat"
        if not _smt.Z3_AVAILABLE else a[-1] != b[-2]
    )
    fixed = a[-1] == b[-1] if not _smt.Z3_AVAILABLE else (
        _smt.check([_smt.eq(k, q)]) != "unsat" and a[-1] == b[-1]
    )
    return bool(a[-1] != b[-2] and a[-1] == b[-1]) and mismatch and fixed


def synth_matmul_fix(lines: List[str], bug) -> Optional[Tuple[str, str, str]]:
    """Synthesizer for ``matmul_dim_mismatch``: when the right operand is stored
    transposed (its last axis equals the left operand's contracted dim), rewrite
    ``a @ b`` to ``a @ b.transpose(-1, -2)``.  Abstains unless the line has a
    single, simply-named ``@`` and the transpose provably fixes the contraction.
    """
    i = bug.line - 1
    if not (0 <= i < len(lines)):
        return None
    line = lines[i]
    if line.count("@") != 1:
        return None
    m = _MATMUL_OP.match(line)
    if not m:
        return None
    shapes = parse_two_shapes(getattr(bug, "message", "") or "")
    if shapes is None:
        return None
    a_shape, b_shape = shapes
    if not _transpose_fixes_contraction(a_shape, b_shape):
        return None
    rhs = m.group("rhs")
    new_line = (
        m.group("pre") + m.group("lhs") + " @ "
        + f"{rhs}.transpose(-1, -2)" + m.group("post")
    )
    if new_line == line:
        return None
    patched = lines[:]
    patched[i] = new_line
    return (
        "\n".join(patched),
        "matmul-transpose",
        f"the right operand `{rhs}` is stored transposed; `@ {rhs}.transpose"
        "(-1, -2)` aligns the contracted dimension",
    )
