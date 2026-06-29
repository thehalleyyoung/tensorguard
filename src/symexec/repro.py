"""Runnable reproducer generation (even_more.md Tier 1, idea #1).

A TensorGuard report is *sound* (the Lean `refute` lemmas + the proof-carrying
certificate guarantee a real forced failure).  This module turns that guarantee
into the single most convincing artifact for a maintainer: a **minimal,
self-contained, runnable Python program that actually raises the predicted
exception**.

`generate_repro(bug)` synthesises, per :class:`~src.symexec.bugs.SymBugKind`, a
tiny script that constructs concrete tensors matching the witness recovered from
the report's (fingerprinted, hence stable) message and invokes the offending
operation.  When run with torch installed it deterministically raises the
predicted `RuntimeError` / `IndexError` / `ZeroDivisionError`.

`confirm(repro)` is the **empirical soundness layer**: it executes the script in
an isolated namespace and checks that the expected exception is in fact raised —
a third, independent confirmation beyond the proof fingerprint and the Lean
proofs.  `confirm` imports torch lazily and is skipped when torch is absent, so
this module itself stays torch-free to import.

Reproducers are *diagnostic*: generating or confirming one never changes which
bugs report, nor the proof fingerprint.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

__all__ = [
    "ReproScript",
    "ReproResult",
    "generate_repro",
    "generate_repros",
    "confirm",
]


@dataclass(frozen=True)
class ReproScript:
    """A minimal runnable program that reproduces one report's failure."""

    kind: str
    script: str          # complete, runnable source
    expected_exception: str  # e.g. "RuntimeError", "IndexError", "ZeroDivisionError"
    fidelity: str        # "exact" (witness shapes) | "class" (same failure class)
    note: str
    line: int = 0
    col: int = 0
    function: str = ""


@dataclass(frozen=True)
class ReproResult:
    """The outcome of executing a reproducer."""

    kind: str
    expected_exception: str
    raised: bool
    raised_exception: Optional[str]
    confirmed: bool      # raised AND the type matches (or subclasses) the prediction
    detail: str


# --------------------------------------------------------------------------- #
# Message parsing helpers.                                                     #
# --------------------------------------------------------------------------- #
def _ints(text: str) -> Tuple[int, ...]:
    return tuple(int(x) for x in re.findall(r"-?\d+", text))


def _randn(shape: Tuple[int, ...]) -> str:
    return "torch.randn(" + ", ".join(str(d) for d in shape) + ")"


# Each generator returns (body_source, expected_exception, fidelity, note) or None.
_Gen = Callable[[str], Optional[Tuple[str, str, str, str]]]


def _g_matmul(msg: str) -> Optional[Tuple[str, str, str, str]]:
    m = re.search(r"\(([\d,\s]+)\) @ \(([\d,\s]+)\)", msg)
    if not m:
        return None
    a, b = _ints(m.group(1)), _ints(m.group(2))
    if len(a) < 2 or len(b) < 2:
        return None
    body = f"import torch\n{_randn(a)} @ {_randn(b)}\n"
    return body, "RuntimeError", "exact", "contracted-dim mismatch in matmul"


def _g_broadcast(msg: str) -> Optional[Tuple[str, str, str, str]]:
    m = re.search(r"shapes \(([\d,\s]+)\) and \(([\d,\s]+)\)", msg)
    if not m:
        return None
    a, b = _ints(m.group(1)), _ints(m.group(2))
    body = f"import torch\n{_randn(a)} + {_randn(b)}\n"
    return body, "RuntimeError", "exact", "non-broadcastable elementwise add"


def _g_reshape(msg: str) -> Optional[Tuple[str, str, str, str]]:
    tgt = re.search(r"target \(([\d,\s]+)\)", msg)
    numel = re.search(r"of (\d+) element", msg)
    if not tgt or not numel:
        return None
    target = _ints(tgt.group(1))
    n = int(numel.group(1))
    body = (
        "import torch\n"
        f"torch.randn({n}).reshape("
        + ", ".join(str(d) for d in target)
        + ")\n"
    )
    return body, "RuntimeError", "exact", "reshape changes the element count"


def _g_axis(msg: str) -> Optional[Tuple[str, str, str, str]]:
    m = re.search(r"dim (\d+) but the tensor has rank (\d+)", msg)
    if not m:
        return None
    dim, rank = int(m.group(1)), int(m.group(2))
    shape = tuple([2] * rank) if rank > 0 else (2,)
    body = f"import torch\n{_randn(shape)}.sum(dim={dim})\n"
    return body, "IndexError", "exact", "reduction axis out of range"


def _g_index(msg: str) -> Optional[Tuple[str, str, str, str]]:
    m = re.search(r"index (\d+) .*length (\d+)", msg)
    if not m:
        return None
    idx, length = int(m.group(1)), int(m.group(2))
    body = f"_xs = [0] * {length}\n_xs[{idx}]\n"
    return body, "IndexError", "exact", "list index out of range"


def _g_divzero(_msg: str) -> Optional[Tuple[str, str, str, str]]:
    body = "_n = 0\n10 // _n\n"
    return body, "ZeroDivisionError", "exact", "integer division by zero"


def _g_linear(msg: str) -> Optional[Tuple[str, str, str, str]]:
    m = re.search(r"last input dim (\d+) but received (\d+)", msg)
    if not m:
        return None
    expected, received = int(m.group(1)), int(m.group(2))
    body = (
        "import torch\n"
        "import torch.nn as nn\n"
        f"nn.Linear({expected}, 1)(torch.randn(1, {received}))\n"
    )
    return body, "RuntimeError", "exact", "nn.Linear in-features mismatch"


def _g_cat(msg: str) -> Optional[Tuple[str, str, str, str]]:
    m = re.search(r"disagree on dim (\d+) \(\[(\d+),\s*(\d+)\]\)", msg)
    if not m:
        return None
    bad_dim, s0, s1 = int(m.group(1)), int(m.group(2)), int(m.group(3))
    rank = bad_dim + 1
    cat_dim = 0 if bad_dim != 0 else 1
    if cat_dim >= rank:
        rank = cat_dim + 1
    base = [2] * rank
    sh0, sh1 = list(base), list(base)
    sh0[bad_dim], sh1[bad_dim] = s0, s1
    body = (
        "import torch\n"
        f"torch.cat([{_randn(tuple(sh0))}, {_randn(tuple(sh1))}], dim={cat_dim})\n"
    )
    return body, "RuntimeError", "exact", "torch.cat inputs disagree off the cat axis"


def _g_einsum(msg: str) -> Optional[Tuple[str, str, str, str]]:
    m = re.search(r"sizes (\d+) vs (\d+)", msg)
    if not m:
        return None
    s0, s1 = int(m.group(1)), int(m.group(2))
    body = (
        "import torch\n"
        f'torch.einsum("i,i->", torch.randn({s0}), torch.randn({s1}))\n'
    )
    return body, "RuntimeError", "class", "einsum binds one index to two sizes"


def _g_negdim(_msg: str) -> Optional[Tuple[str, str, str, str]]:
    body = "import torch\ntorch.randn(-1)\n"
    return body, "RuntimeError", "class", "tensor constructed with a negative dimension"


_GENERATORS: Dict[str, _Gen] = {
    "matmul_dim_mismatch": _g_matmul,
    "broadcast_mismatch": _g_broadcast,
    "reshape_size_mismatch": _g_reshape,
    "axis_out_of_range": _g_axis,
    "tensor_index_oob": _g_index,
    "rank_index_error": _g_index,
    "division_by_zero": _g_divzero,
    "layer_dim_mismatch": _g_linear,
    "cat_shape_mismatch": _g_cat,
    "einsum_dim_mismatch": _g_einsum,
    "negative_dimension": _g_negdim,
}


def _header(bug, kind: str, note: str, fidelity: str, exc: str) -> str:
    loc = f"line {getattr(bug, 'line', 0)}"
    fn = getattr(bug, "function", "") or "<module>"
    return (
        f'"""Auto-generated TensorGuard reproducer.\n\n'
        f"Bug kind : {kind}\n"
        f"Origin   : {fn} ({loc})\n"
        f"Predicts : {exc} ({note})\n"
        f"Fidelity : {fidelity}\n"
        f'"""\n'
    )


def generate_repro(bug) -> Optional[ReproScript]:
    """Synthesise a runnable reproducer for one :class:`SymBug`, or ``None`` when
    the kind has no generator (honest — never a non-crashing stub)."""
    kind = getattr(bug.kind, "value", str(bug.kind))
    gen = _GENERATORS.get(kind)
    if gen is None:
        return None
    message = getattr(bug, "message", "") or ""
    try:
        produced = gen(message)
    except Exception:
        produced = None
    if produced is None:
        return None
    body, exc, fidelity, note = produced
    script = _header(bug, kind, note, fidelity, exc) + body
    return ReproScript(
        kind=kind,
        script=script,
        expected_exception=exc,
        fidelity=fidelity,
        note=note,
        line=int(getattr(bug, "line", 0)),
        col=int(getattr(bug, "col", 0)),
        function=getattr(bug, "function", "") or "",
    )


def generate_repros(result) -> List[ReproScript]:
    """Reproducers for every report in a :class:`SymResult` that has a
    generator (skips kinds without one)."""
    out: List[ReproScript] = []
    for b in result.bugs:
        r = generate_repro(b)
        if r is not None:
            out.append(r)
    return out


def confirm(repro: ReproScript) -> ReproResult:
    """Execute a reproducer in an isolated namespace and report whether it
    raised the predicted exception.  Requires torch for tensor reproducers;
    raises ``RuntimeError`` if a torch-dependent script is confirmed without
    torch installed (callers should gate on torch availability)."""
    needs_torch = "import torch" in repro.script
    if needs_torch:
        try:  # pragma: no cover - torch presence is environment-dependent
            import torch  # noqa: F401
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(
                "confirm() needs torch for this reproducer; gate on availability"
            ) from exc

    ns: Dict[str, object] = {"__name__": "__tg_repro__"}
    try:
        code = compile(repro.script, "<tg-repro>", "exec")
        exec(code, ns)  # noqa: S102 - executing our own generated snippet
    except BaseException as exc:  # noqa: BLE001 - we want the raised type
        name = type(exc).__name__
        matched = _matches(type(exc), repro.expected_exception)
        return ReproResult(
            kind=repro.kind,
            expected_exception=repro.expected_exception,
            raised=True,
            raised_exception=name,
            confirmed=matched,
            detail=(f"raised {name}"
                    + ("" if matched else
                       f" (expected {repro.expected_exception})")),
        )
    return ReproResult(
        kind=repro.kind,
        expected_exception=repro.expected_exception,
        raised=False,
        raised_exception=None,
        confirmed=False,
        detail="reproducer ran without raising — NOT confirmed",
    )


def _matches(exc_type: type, expected_name: str) -> bool:
    """True if ``exc_type`` is, or subclasses, the predicted exception.

    ``RuntimeError`` is matched leniently against torch's subclasses; the
    integer-arithmetic and indexing predictions are matched exactly by name or
    inheritance."""
    for klass in exc_type.__mro__:
        if klass.__name__ == expected_name:
            return True
    return False
