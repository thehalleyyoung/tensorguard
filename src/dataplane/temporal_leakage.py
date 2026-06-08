"""Static scanner for *temporal lookahead* leakage (no-lookahead contract).

Where :mod:`datarefine.pipeline_leakage` certifies **fit/transform isolation**
(a *set*-disjointness property over train/holdout rows) and
:mod:`datarefine.split_contracts` certifies **split disjointness / partition
lengths / bounds**, this scanner certifies a *strictly different* and more subtle
contract: **temporal causality**.

    No-lookahead: a feature computed at output row ``i`` of an ordered (time
    indexed) frame must depend only on input rows at index ``<= i``.  A feature
    that reads a *future* row is *lookahead bias* --- silent at training/backtest
    time (the metrics look great) and unavailable at streaming inference, so the
    model fails in production.  No test, type checker, or linter catches it.

The crucial design move (mirroring how ``split_contracts`` only understands
*literal* index ranges, never arbitrary indexing) is that we **do not model
pandas / numpy / Python semantics**.  Lookahead is a *local property of the
feature-construction operator*: ``x.shift(-k)`` reads ``k`` rows into the future
*by the operator's own definition*, regardless of where (or whether) a split
happens later.  So we keep a small **operator catalog** annotating each operator
with its **forward reach** (an integer), recognise the operator chain
syntactically, **compose** the reaches by summation, and lower the single integer
``forward_reach`` to a ``temporal_causality`` obligation that the z3-backed
:class:`~datarefine.certification.StructuralCertifier` decides --- z3 *searches*
the index axis for a concrete ``(feature_row, future_source_row)`` counterexample.

This is the theoretical generalisation: the contract lattice moves from
*set-membership disjointness* to *temporal causality (no future read)* while the
machinery stays a syntactic-signature recogniser plus a small decidable
obligation.  Anything not in the catalog (and any trailing / causal operator)
contributes reach ``0`` and is **not** flagged --- precision over soundness.

Operator catalog (signed contribution to the *forward edge* of the dependency
window; the window's forward edge is the largest future offset the feature reads,
and operators compose by **summation** because translations and window
extensions are additive on the index axis):

* ``shift(k)`` / ``tshift(k)``   index translation        -> ``-k``
  (``shift(-1)`` -> ``+1`` future; ``shift(2)`` -> ``-2`` past, *compensating*)
* ``np.roll(a, s)``              index translation        -> ``-s``
* ``rolling(w, center=True)``    symmetric window         -> ``+(w // 2)``  (trailing: 0)
* ``diff(periods)`` / ``pct_change(periods)``   negative literal ``-k`` -> ``+k``  (else 0)
* ``ewm`` / ``expanding`` / ``cum*``                              -> 0 (causal)

A feature **leaks** iff the composed forward edge ``R = sum(deltas) >= 1``.  A net
non-positive edge is causal/trailing and is **not** flagged --- crucially this
makes the real-world *compensation* idiom ``rolling(w, center=True)...shift(w//2)``
(centered window deliberately delayed back into the past) certify clean, exactly
as a careful author intends.  ``R == 0`` makes the violation predicate
unsatisfiable, so z3 *admits* it.

A feature-role left-hand side is required: ``df['feat'] = ...`` / ``feat = ...``.
A *target*-role LHS (``y``, ``target``, ``label``, ``next_*``, ``*_ahead`` ...) is
**exempt** --- building a forward-shifted *prediction target* (``y =
price.shift(-1)``) is correct ML practice, not leakage.
"""
from __future__ import annotations

import ast
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable

from .certification import StructuralCertifier
from .obligations import obligation

__all__ = [
    "TemporalFinding",
    "TemporalLeakageScanner",
    "scan_source",
    "scan_path",
]

_CERT = StructuralCertifier()

# LHS names that denote a *prediction target* rather than a feature; a forward
# shift used to build these is legitimate (next-step label), so we stay silent.
_TARGET_ROLE_RE = re.compile(
    r"(?:^|_)(y|target|targets|label|labels|outcome|response|gt|truth|"
    r"future|next|ahead|forward|fwd|horizon|fut)(?:$|_)",
    re.IGNORECASE,
)

# Operators that are causal by construction (reach 0); listed only so a reader
# sees they were considered and deliberately not flagged.
_CAUSAL_OPS = frozenset({"ewm", "expanding", "cumsum", "cumprod", "cummax",
                         "cummin", "cumcount", "expanding_mean"})


@dataclass
class TemporalFinding:
    file: str
    line: int
    pattern: str
    constraint: str
    verdict: str
    detail: str
    snippet: str
    forward_reach: int
    operators: list
    witness: dict | None = None

    def as_dict(self) -> dict:
        return asdict(self)


def _name_of(node: ast.AST | None) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return None


def _lit_int(node: ast.AST | None) -> int | None:
    """Return a literal int, resolving unary minus (``-1`` is ``UnaryOp(USub)``)."""
    if isinstance(node, ast.Constant) and isinstance(node.value, bool) is False \
            and isinstance(node.value, int):
        return node.value
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        inner = _lit_int(node.operand)
        return -inner if inner is not None else None
    return None


def _lit_true(node: ast.AST | None) -> bool:
    return isinstance(node, ast.Constant) and node.value is True


def _arg(call: ast.Call, name: str, pos: int | None) -> ast.AST | None:
    for kw in call.keywords:
        if kw.arg == name:
            return kw.value
    if pos is not None and pos < len(call.args):
        return call.args[pos]
    return None


def _forward_edge_delta(method: str, call: ast.Call) -> int:
    """Signed contribution of one ``.method(...)`` call to the dependency
    window's *forward edge*.

    Positive => the feature reaches further into the future; negative => a
    backward translation that *compensates* prior lookahead.  Anything not
    recognised contributes 0 (precision-first: unknown operators never fire).
    """
    if method in ("shift", "tshift"):
        # shift(k): out[i] = x[i-k]; forward edge moves by -k. Default periods=1.
        k = _lit_int(_arg(call, "periods", 0))
        if k is None and not call.args and not call.keywords:
            k = 1  # bare .shift() defaults to periods=1 (backward)
        return -k if k is not None else 0
    if method == "roll":  # np.roll(a, s): out[i] = a[i-s]; forward edge moves -s
        s = _lit_int(_arg(call, "shift", 1))
        return -s if s is not None else 0
    if method == "rolling":
        if not _lit_true(_arg(call, "center", None)):
            return 0
        w = _lit_int(_arg(call, "window", 0))
        if w is not None and w >= 2:
            return w // 2
        return 0
    if method in ("diff", "pct_change"):
        # diff(periods): out[i] = x[i] - x[i-periods]; reads the future only when
        # periods < 0 (then forward edge += |periods|). It keeps row i, so a
        # positive/default periods is causal (0), not a backward translation.
        k = _lit_int(_arg(call, "periods", 0))
        if k is not None and k < 0:
            return -k
        return 0
    return 0


def _chain_calls(head: ast.Call) -> list[tuple[str, ast.Call]]:
    """Walk a method chain from ``head`` inward, yielding ``(method, call)``.

    ``a.shift(-2).rolling(3, center=True).mean()`` -> the outer ``mean`` call's
    ``.func.value`` is the ``rolling`` call, whose ``.func.value`` is the
    ``shift`` call.  We collect every call whose ``func`` is an attribute access.
    """
    out: list[tuple[str, ast.Call]] = []
    node: ast.AST = head
    while isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
        out.append((node.func.attr, node))
        node = node.func.value
    return out


def _is_chain_head(call: ast.Call, inner_calls: set[int]) -> bool:
    return id(call) not in inner_calls


def _collect_inner_calls(value: ast.AST) -> set[int]:
    """Ids of calls that are the ``.func.value`` receiver of an enclosing call
    (i.e. *not* chain heads), so each method chain is counted once."""
    inner: set[int] = set()
    for node in ast.walk(value):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            recv = node.func.value
            if isinstance(recv, ast.Call):
                inner.add(id(recv))
    return inner


def _max_chain_reach(value: ast.AST) -> tuple[int, list]:
    """Largest net forward edge over the method chains in ``value``.

    Operator deltas compose by summation along a chain; a net edge ``<= 0`` is
    causal.  Returns ``(reach, operators)`` where ``operators`` is the signed
    ``[method, delta]`` list of the worst chain (provenance, including any
    compensating backward shift, for the witness/detail).
    """
    inner = _collect_inner_calls(value)
    best_reach = 0
    best_ops: list = []
    for node in ast.walk(value):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        if not _is_chain_head(node, inner):
            continue
        total = 0
        ops: list = []
        for method, call in _chain_calls(node):
            d = _forward_edge_delta(method, call)
            if d:
                ops.append([method, d])
            total += d
        if total > best_reach:
            best_reach = total
            best_ops = list(reversed(ops))
    return best_reach, best_ops


def _target_role_names(target: ast.AST) -> list[str]:
    """Candidate names for the assignment target (column key / var / attr)."""
    names: list[str] = []
    if isinstance(target, ast.Subscript):
        sl = target.slice
        if isinstance(sl, ast.Constant) and isinstance(sl.value, str):
            names.append(sl.value)
        base = _name_of(target.value)
        if base:
            names.append(base)
    elif isinstance(target, ast.Name):
        names.append(target.id)
    elif isinstance(target, ast.Attribute):
        names.append(target.attr)
    return names


def _looks_like_target(target: ast.AST) -> bool:
    for n in _target_role_names(target):
        if _TARGET_ROLE_RE.search(n):
            return True
    return False


class TemporalLeakageScanner:
    def __init__(self, source: str, filename: str = "<string>") -> None:
        self.source = source
        self.file = filename
        self.src_lines = source.splitlines()
        self.findings: list[TemporalFinding] = []

    def _certify(self, reach: int, operators: list) -> tuple[str, dict | None]:
        ob = obligation(
            "temporal", self.file, "temporal_causality",
            constraint="temporal_causality",
            forward_reach=reach,
            backward_reach=0,
            operators=[list(op) for op in operators],
            horizon=reach + 2,
        )
        verdict = _CERT.certify(ob)
        witness = None
        diags = verdict.diagnostics or ()
        if diags and isinstance(diags[0], dict):
            witness = diags[0].get("model")
        return verdict.status, witness

    def _emit(self, line: int, reach: int, operators: list,
              status: str, witness: dict | None) -> None:
        if status != "rejected":
            return
        snippet = self.src_lines[line - 1].strip() if 0 < line <= len(self.src_lines) else ""
        chain = " -> ".join(f"{m}({d:+d})" for m, d in operators) or "lookahead operator"
        detail = (f"feature reads {reach} row(s) into the future via {chain}; at "
                  f"streaming inference row {reach} is unavailable when row 0 is "
                  f"scored (lookahead bias).")
        self.findings.append(TemporalFinding(
            file=self.file, line=line, pattern="T1:feature_reads_future_row",
            constraint="temporal_causality", verdict=status, detail=detail,
            snippet=snippet, forward_reach=reach, operators=operators,
            witness=witness))

    def scan(self) -> list[TemporalFinding]:
        try:
            tree = ast.parse(self.source, filename=self.file)
        except SyntaxError:
            return []
        for node in ast.walk(tree):
            targets: list[ast.AST]
            if isinstance(node, ast.Assign):
                targets = list(node.targets)
                value = node.value
            elif isinstance(node, ast.AnnAssign) and node.value is not None:
                targets = [node.target]
                value = node.value
            else:
                continue
            reach, operators = _max_chain_reach(value)
            if reach < 1:
                continue
            # A forward-shifted *target* (next-step label) is legitimate.
            if any(_looks_like_target(t) for t in targets):
                continue
            line = getattr(node, "lineno", 0)
            status, witness = self._certify(reach, operators)
            self._emit(line, reach, operators, status, witness)
        return self.findings


def scan_source(source: str, filename: str = "<string>") -> list[TemporalFinding]:
    return TemporalLeakageScanner(source, filename).scan()


def scan_path(path: str | Path) -> list[TemporalFinding]:
    p = Path(path)
    try:
        source = p.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return []
    return scan_source(source, filename=str(p))


def scan_tree(root: str | Path) -> list[TemporalFinding]:
    root = Path(root)
    out: list[TemporalFinding] = []
    paths: Iterable[Path] = [root] if root.is_file() else root.rglob("*.py")
    for p in paths:
        out.extend(scan_path(p))
    return out
