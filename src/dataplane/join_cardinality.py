"""Static scanner for the pandas merge / join **cardinality-preservation** contract.

This is the first contract in the lattice about *relational multiplicity* rather
than membership (`split_disjointness`, `group_disjointness`), order
(`temporal_causality`), or effect (`sampling_independence`).  A relational join
``L >< _k R`` preserves the left cardinality **iff** the join key ``k`` is unique
on the right (an ``m:1`` / ``1:1`` join).  When ``k`` is *not* unique on the
right the rows **fan out** --- ``|result| = sum_l mult_R(k_l)`` --- silently
duplicating samples and inflating every downstream count / mean / split.  This is
the canonical, runtime-silent pandas ``merge`` footgun that the library added the
``validate=`` keyword specifically to prevent; almost nobody passes it.

Two concrete bug shapes are recognised, each requiring a *cardinality-sensitive
consumer* of the merged frame so every finding is a real defect (not merely an
un-annotated merge):

* **M1 --- fan-out before a split.**  An un-guarded merge whose result is then
  fed to ``train_test_split`` / ``random_split`` / ``.sample(frac<1)``.  The
  duplicated rows land in **both** train and test --- a duplicate-leakage that
  silently inflates the held-out metric.

* **M2 --- fan-out before an aggregate.**  An un-guarded merge whose result feeds
  a cardinality-sensitive reducer (``len`` / ``.shape[0]`` / ``groupby().size()``
  / ``.sum()`` / ``.mean()`` / ``.count()`` / ``.value_counts()`` / ``.nunique()``).
  The duplicated rows over-count groups and bias every per-group statistic.

As with the other scanners we **do not model pandas semantics**: the contract is
recognised from *local, syntactic* signals --- a ``pd.merge`` / ``df.merge`` /
``df.join`` with an **explicit** join key and **no** ``validate=`` argument, a
right operand that is **not** made unique on the key (no preceding / inline
``drop_duplicates`` / ``groupby`` / ``pivot_table`` / ``set_index`` / ``unique``),
and a cardinality-sensitive consumer of the result.  Each candidate is lowered to
a ``join_cardinality`` obligation that the z3-backed
:class:`~datarefine.certification.StructuralCertifier` decides over the
multiplicity arithmetic (returning a concrete fan-out factor as the witness),
with an independent re-check.

The precision boundary is honest and identical in spirit to ``group_leakage`` /
``dataloader_determinism``: this is a *recall* (co-occurrence) recognizer, not a
soundness oracle --- it does not *prove* the right key is non-unique (a merge
against an implicitly-unique dimension table with no in-module dedup is the
expected false-positive class), but the certifier still vetoes via the
differential oracle (disagreement => ``unknown``, never silently admitted).
"""
from __future__ import annotations

import ast
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable

from .certification import StructuralCertifier
from .obligations import obligation

__all__ = [
    "JoinFinding",
    "JoinCardinalityScanner",
    "scan_source",
    "scan_path",
    "scan_tree",
]

_CERT = StructuralCertifier()

# Method calls that establish key-uniqueness on the right operand (so the merge
# cannot fan out): the right frame is collapsed to one row per key.
_UNIQUE_OPS = frozenset({
    "drop_duplicates", "groupby", "pivot_table", "pivot", "set_index",
    "unique", "nunique", "first", "last", "agg", "aggregate", "value_counts",
})
# Cardinality-sensitive reducers whose correctness depends on row multiplicity.
_REDUCERS = frozenset({
    "sum", "mean", "count", "size", "value_counts", "nunique", "median",
    "std", "var", "min", "max", "prod",
})
# Splitters that move the merged result into train/test (duplicate leakage).
_SPLITTERS = frozenset({"train_test_split", "random_split"})


@dataclass
class JoinFinding:
    file: str
    line: int
    pattern: str
    constraint: str
    verdict: str
    detail: str
    snippet: str
    join_key: str
    how: str
    consumer: str
    witness: dict | None = None

    def as_dict(self) -> dict:
        return asdict(self)


def _name_of(node: ast.AST | None) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return None


def _kw(call: ast.Call, name: str) -> ast.AST | None:
    for kw in call.keywords:
        if kw.arg == name:
            return kw.value
    return None


def _const_str(node: ast.AST | None) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None


def _join_key(call: ast.Call) -> str | None:
    """Explicit join key from ``on=`` / ``left_on=`` (None when the merge would
    fall back to the implicit common-column join, which we do not flag)."""
    for kw_name in ("on", "left_on"):
        v = _kw(call, kw_name)
        if v is None:
            continue
        s = _const_str(v)
        if s is not None:
            return s
        if isinstance(v, (ast.List, ast.Tuple)) and v.elts:
            first = _const_str(v.elts[0])
            if first is not None:
                return first
        if isinstance(v, ast.Name):
            return v.id
    return None


def _expr_makes_unique(node: ast.AST | None) -> bool:
    """True if ``node`` is/contains a uniqueness-establishing op on a frame."""
    if node is None:
        return False
    for n in ast.walk(node):
        if isinstance(n, ast.Call) and _name_of(n.func) in _UNIQUE_OPS:
            return True
    return False


class JoinCardinalityScanner:
    def __init__(self, source: str, filename: str = "<string>") -> None:
        self.source = source
        self.file = filename
        self.src_lines = source.splitlines()
        self.findings: list[JoinFinding] = []
        self._assign: dict[str, ast.AST] = {}

    def _certify(self, **payload) -> tuple[str, dict | None]:
        ob = obligation("join", self.file, "join_cardinality",
                        constraint="join_cardinality", **payload)
        verdict = _CERT.certify(ob)
        witness = None
        diags = verdict.diagnostics or ()
        if diags and isinstance(diags[0], dict):
            witness = diags[0].get("model")
        return verdict.status, witness

    def _snippet(self, line: int) -> str:
        return self.src_lines[line - 1].strip() if 0 < line <= len(self.src_lines) else ""

    def _collect_assignments(self, tree: ast.AST) -> None:
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign) and node.value is not None:
                for t in node.targets:
                    name = _name_of(t)
                    if name:
                        self._assign[name] = node.value

    # -- merge recognition --------------------------------------------------
    def _merge_parts(self, call: ast.Call):
        """Return ``(right_node, key, how, validated)`` for a merge/join call, or
        ``None`` when it is not an explicit-key merge we should reason about."""
        fname = _name_of(call.func)
        if fname not in ("merge", "join"):
            return None
        how = _const_str(_kw(call, "how")) or ("left" if fname == "join" else "inner")
        if how == "cross":
            return None  # an intentional cartesian product, not a fan-out bug
        key = _join_key(call)
        if key is None:
            return None  # implicit common-column join -> precision-first skip
        validated = _kw(call, "validate") is not None
        # right operand: pd.merge(L, R, ...) vs L.merge(R, ...) / L.join(R, ...)
        if isinstance(call.func, ast.Attribute):           # L.merge(R, ...)
            right = call.args[0] if call.args else None
        else:                                              # pd.merge(L, R, ...)
            right = call.args[1] if len(call.args) >= 2 else None
        return right, key, how, validated

    def _right_is_unique(self, right: ast.AST | None) -> bool:
        if right is None:
            return False
        if _expr_makes_unique(right):
            return True
        if isinstance(right, ast.Name):
            return _expr_makes_unique(self._assign.get(right.id))
        return False

    # -- consumer detection -------------------------------------------------
    def _consumer(self, tree: ast.AST, result: str | None) -> str | None:
        """``"split"`` (preferred) or ``"aggregate"`` if the merged result is
        consumed in a cardinality-sensitive way, else ``None``."""
        if result is None:
            return None
        aggregate = False
        for n in ast.walk(tree):
            if isinstance(n, ast.Call):
                fname = _name_of(n.func)
                # train_test_split(merged, ...) / random_split(merged, ...)
                if fname in _SPLITTERS and any(
                        isinstance(a, ast.Name) and a.id == result for a in n.args):
                    return "split"
                # merged.sample(frac<1) / merged.sample(n=...)
                if fname == "sample" and isinstance(n.func, ast.Attribute) \
                        and _name_of(n.func.value) == result:
                    return "split"
                # len(merged)
                if fname == "len" and any(
                        isinstance(a, ast.Name) and a.id == result for a in n.args):
                    aggregate = True
                # merged....reducer(...)
                if fname in _REDUCERS and isinstance(n.func, ast.Attribute) \
                        and self._roots_at(n.func.value, result):
                    aggregate = True
            # merged.shape[...]
            if isinstance(n, ast.Attribute) and n.attr == "shape" \
                    and _name_of(n.value) == result:
                aggregate = True
        return "aggregate" if aggregate else None

    @staticmethod
    def _roots_at(node: ast.AST, name: str) -> bool:
        cur = node
        while isinstance(cur, (ast.Attribute, ast.Subscript, ast.Call)):
            if isinstance(cur, ast.Call):
                cur = cur.func
            elif isinstance(cur, ast.Attribute):
                cur = cur.value
            else:  # Subscript
                cur = cur.value
        return isinstance(cur, ast.Name) and cur.id == name

    def _result_name(self, tree: ast.AST, call: ast.Call) -> str | None:
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign) and node.value is call:
                return _name_of(node.targets[0]) if node.targets else None
        return None

    def scan(self) -> list[JoinFinding]:
        try:
            tree = ast.parse(self.source, filename=self.file)
        except SyntaxError:
            return []
        self._collect_assignments(tree)
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            parts = self._merge_parts(node)
            if parts is None:
                continue
            right, key, how, validated = parts
            if validated:
                continue
            if self._right_is_unique(right):
                continue
            result = self._result_name(tree, node)
            consumer = self._consumer(tree, result)
            if consumer is None:
                continue
            status, witness = self._certify(
                validated=False, right_key_unique=False,
                cardinality_consumed=True, join_key=key, how=how)
            if status != "rejected":
                continue
            line = getattr(node, "lineno", 0)
            if consumer == "split":
                pattern = "M1:fanout_before_split"
                detail = (
                    f"merge on {key!r} (how={how}) has no validate= guard and the "
                    f"right key is not deduplicated, so rows can fan out; the merged "
                    f"frame is then split/sampled, leaking duplicated rows into both "
                    f"train and test. Pass validate='m:1'/'one_to_one', or "
                    f"drop_duplicates(subset=['{key}']) the right frame first."
                )
            else:
                pattern = "M2:fanout_before_aggregate"
                detail = (
                    f"merge on {key!r} (how={how}) has no validate= guard and the "
                    f"right key is not deduplicated, so rows can fan out; the merged "
                    f"frame then feeds a count/mean/sum aggregate that is silently "
                    f"inflated. Pass validate='m:1', or deduplicate the right key."
                )
            self.findings.append(JoinFinding(
                file=self.file, line=line, pattern=pattern,
                constraint="join_cardinality", verdict=status, detail=detail,
                snippet=self._snippet(line), join_key=key, how=how,
                consumer=consumer, witness=witness))
        return self.findings


def scan_source(source: str, filename: str = "<string>") -> list[JoinFinding]:
    return JoinCardinalityScanner(source, filename).scan()


def scan_path(path: str | Path) -> list[JoinFinding]:
    p = Path(path)
    try:
        source = p.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return []
    return scan_source(source, filename=str(p))


def scan_tree(root: str | Path) -> list[JoinFinding]:
    root = Path(root)
    out: list[JoinFinding] = []
    paths: Iterable[Path] = [root] if root.is_file() else root.rglob("*.py")
    for p in paths:
        out.extend(scan_path(p))
    return out
