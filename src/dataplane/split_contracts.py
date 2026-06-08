"""Static scanner for split / partition / bounds data-contract bugs.

Where :mod:`datarefine.pipeline_leakage` finds *fit-before-split* leakage, this
scanner targets a different family of real, common torch / pandas data bugs and
routes each through DataRefine's **z3-backed** structural certifier so the SMT
backend (not the scanner) decides them:

* ``S1`` overlapping integer index ranges across named train/val/test partitions
  (``Subset(ds, range(a, b))``, ``df.iloc[a:b]``, ``x[a:b]``) -> a
  ``split_disjointness`` obligation over *intervals*; z3 searches for an actual
  overlapping row index and returns it as the witness.

* ``S2`` ``random_split(ds, [f1, f2, ...])`` whose fractional lengths do not sum
  to 1 (a hard ``torch`` runtime error for the >=1.13 fractional API) or contain
  a non-positive partition -> a ``partition_lengths`` obligation decided by z3
  linear real arithmetic.

* ``S3`` an out-of-range scalar literal in a call whose argument has a known
  admissible interval (``nn.Dropout(p=1.5)``, ``train_test_split(test_size=1.5)``)
  -> a ``bounds`` obligation decided by z3.

Every finding's verdict is the certifier's; the scanner only locates candidate
sites and builds the obligation. The differential oracle re-decides each formula
with the pure-Python procedure too, so an admitted clean case is agreed by two
independent deciders and a disagreement is surfaced as ``unknown`` rather than a
false positive.
"""
from __future__ import annotations

import ast
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable

from .certification import StructuralCertifier
from .obligations import obligation

__all__ = [
    "ContractFinding",
    "SplitContractScanner",
    "scan_source",
    "scan_path",
    "scan_split_contracts",
    "scan_split_contracts_path",
]

_CERT = StructuralCertifier()

# variable-name tokens that signal a train/val/test partition role
_SPLIT_ROLE_TOKENS = ("train", "val", "valid", "test", "holdout", "dev", "eval")
_SUBSET_NAMES = frozenset({"Subset"})
# Index-list consumers: `SubsetRandomSampler(indices[a:b])` /
# `Subset(ds, indices[a:b])` carve a partition out of an index list. If two
# split-role partitions index into the *same* list over overlapping ranges, the
# same rows land in more than one split -- exactly the disjointness obligation z3
# already decides for S1.
_INDEX_SAMPLER_NAMES = frozenset({"SubsetRandomSampler", "Subset"})
_RANDOM_SPLIT_NAMES = frozenset({"random_split"})
# call -> {arg_name or positional index: (lower, upper)} admissible bounds
_BOUNDED_ARGS = {
    "Dropout": {"p": (0.0, 1.0), 0: (0.0, 1.0)},
    "Dropout2d": {"p": (0.0, 1.0), 0: (0.0, 1.0)},
    "Dropout3d": {"p": (0.0, 1.0), 0: (0.0, 1.0)},
    "AlphaDropout": {"p": (0.0, 1.0), 0: (0.0, 1.0)},
    "train_test_split": {"test_size": (0.0, 1.0), "train_size": (0.0, 1.0)},
}


@dataclass
class ContractFinding:
    file: str
    line: int
    pattern: str
    constraint: str
    verdict: str
    detail: str
    snippet: str
    witness: dict | None = None

    def as_dict(self) -> dict:
        return asdict(self)


def _name_of(node: ast.AST | None) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return None


def _call_func_name(call: ast.Call) -> str | None:
    return _name_of(call.func)


def _receiver_name(call: ast.Call) -> str | None:
    if isinstance(call.func, ast.Attribute):
        return _name_of(call.func.value)
    return None


def _literal_int(node: ast.AST | None) -> int | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, bool) is False \
            and isinstance(node.value, int):
        return node.value
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        inner = _literal_int(node.operand)
        return -inner if inner is not None else None
    return None


def _literal_number(node: ast.AST | None) -> float | None:
    if isinstance(node, ast.Constant) and not isinstance(node.value, bool) \
            and isinstance(node.value, (int, float)):
        return float(node.value)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        inner = _literal_number(node.operand)
        return -inner if inner is not None else None
    return None


def _has_split_role(name: str) -> bool:
    low = name.lower()
    return any(tok in low for tok in _SPLIT_ROLE_TOKENS)


def _range_bounds(node: ast.AST) -> tuple[int, int] | None:
    """Extract [lo, hi) from ``range(lo, hi)`` / ``range(hi)``."""
    if isinstance(node, ast.Call) and _call_func_name(node) == "range":
        args = [a for a in node.args]
        if len(args) == 1:
            hi = _literal_int(args[0])
            return (0, hi) if hi is not None else None
        if len(args) >= 2:
            lo, hi = _literal_int(args[0]), _literal_int(args[1])
            if lo is not None and hi is not None:
                return (lo, hi)
    return None


def _slice_bounds(node: ast.AST) -> tuple[str | None, tuple[int, int]] | None:
    """Extract (base, [lo, hi)) from ``base[lo:hi]`` / ``base.iloc[lo:hi]``."""
    if not isinstance(node, ast.Subscript) or not isinstance(node.slice, ast.Slice):
        return None
    sl = node.slice
    if sl.step is not None:
        return None
    lo = _literal_int(sl.lower) if sl.lower is not None else 0
    hi = _literal_int(sl.upper) if sl.upper is not None else None
    if lo is None or hi is None:  # open-ended slice: end unknown, stay sound
        return None
    target = node.value
    base = _name_of(target.value) if isinstance(target, ast.Attribute) else _name_of(target)
    return base, (lo, hi)


class SplitContractScanner:
    def __init__(self, source: str, filename: str = "<string>") -> None:
        self.source = source
        self.file = filename
        self.src_lines = source.splitlines()
        self.findings: list[ContractFinding] = []

    # -- obligation routing -------------------------------------------------
    def _certify(self, kind: str, constraint: str, **payload) -> tuple[str, dict | None]:
        ob = obligation(kind, self.file, constraint, constraint=constraint, **payload)
        verdict = _CERT.certify(ob)
        witness = None
        diags = verdict.diagnostics or ()
        if diags and isinstance(diags[0], dict):
            witness = diags[0].get("model")
        return verdict.status, witness

    def _emit(self, line: int, pattern: str, constraint: str, detail: str,
              status: str, witness: dict | None) -> None:
        if status != "rejected":
            return
        snippet = self.src_lines[line - 1].strip() if 0 < line <= len(self.src_lines) else ""
        self.findings.append(ContractFinding(
            file=self.file, line=line, pattern=pattern, constraint=constraint,
            verdict=status, detail=detail, snippet=snippet, witness=witness))

    # -- driver -------------------------------------------------------------
    def scan(self) -> list[ContractFinding]:
        try:
            tree = ast.parse(self.source, filename=self.file)
        except SyntaxError:
            return []
        self._scan_overlapping_ranges(tree)
        self._scan_random_split(tree)
        self._scan_bounds(tree)
        return self.findings

    # S1 -- overlapping integer index ranges across named partitions --------
    def _scan_overlapping_ranges(self, tree: ast.AST) -> None:
        # base object -> {partition_name: (lo, hi), line}
        groups: dict[str, dict[str, tuple[int, int]]] = {}
        group_line: dict[str, int] = {}
        for node in ast.walk(tree):
            if not isinstance(node, ast.Assign) or len(node.targets) != 1:
                continue
            tname = _name_of(node.targets[0])
            if not tname or not _has_split_role(tname):
                continue
            base: str | None = None
            bounds: tuple[int, int] | None = None
            val = node.value
            if isinstance(val, ast.Call) and _call_func_name(val) in _SUBSET_NAMES:
                base = _name_of(val.args[0]) if val.args else None
                idx = val.args[1] if len(val.args) > 1 else None
                rb = _range_bounds(idx) if idx is not None else None
                if rb is not None:
                    bounds = rb
                elif idx is not None:
                    # Subset(ds, indices[a:b]): the partition indexes into an index
                    # list; key the group by that list so two slices of the *same*
                    # list are compared for overlap.
                    sb = _slice_bounds(idx)
                    if sb is not None:
                        base, bounds = sb
            elif isinstance(val, ast.Call) and _call_func_name(val) in _INDEX_SAMPLER_NAMES \
                    and val.args:
                # SubsetRandomSampler(indices[a:b])
                sb = _slice_bounds(val.args[0])
                if sb is not None:
                    base, bounds = sb
            else:
                sb = _slice_bounds(val)
                if sb is not None:
                    base, bounds = sb
            if base is None or bounds is None:
                continue
            groups.setdefault(base, {})[tname] = bounds
            group_line[base] = max(group_line.get(base, 0), getattr(node, "lineno", 0))
        for base, parts in groups.items():
            if len(parts) < 2:
                continue
            interval_partitions = {n: [lo, hi] for n, (lo, hi) in parts.items()}
            status, witness = self._certify(
                "split", "split_disjointness",
                interval_partitions=interval_partitions)
            line = group_line.get(base, 0)
            detail = (f"index partitions over `{base}` "
                      + ", ".join(f"{n}=[{lo},{hi})" for n, (lo, hi) in sorted(parts.items()))
                      + " overlap; the same rows appear in more than one split.")
            self._emit(line, "S1:overlapping_index_ranges", "split_disjointness",
                       detail, status, witness)

    # S2 -- random_split fractional lengths -> partition_lengths ------------
    def _scan_random_split(self, tree: ast.AST) -> None:
        for node in ast.walk(tree):
            if not (isinstance(node, ast.Call)
                    and _call_func_name(node) in _RANDOM_SPLIT_NAMES
                    and len(node.args) >= 2):
                continue
            lengths_node = node.args[1]
            if not isinstance(lengths_node, (ast.List, ast.Tuple)):
                continue
            nums = [_literal_number(e) for e in lengths_node.elts]
            if any(n is None for n in nums) or not nums:
                continue
            line = getattr(node, "lineno", 0)
            looks_fractional = all(0.0 < n < 1.0 for n in nums if n is not None) and \
                any(not float(n).is_integer() for n in nums if n is not None)
            if looks_fractional:
                status, witness = self._certify(
                    "split", "partition_lengths",
                    lengths=[n for n in nums], fractions=True, total=1.0)
                detail = (f"`random_split(..., {[n for n in nums]})` uses fractional lengths "
                          f"that sum to {sum(n for n in nums):.6g}, not 1.0 "
                          f"(torch's fractional API requires the fractions to sum to 1).")
                self._emit(line, "S2:random_split_fraction_sum", "partition_lengths",
                           detail, status, witness)
            elif any(n is not None and n <= 0 for n in nums):
                status, witness = self._certify(
                    "split", "partition_lengths", lengths=[n for n in nums])
                detail = (f"`random_split(..., {[n for n in nums]})` contains a non-positive "
                          f"partition length.")
                self._emit(line, "S2:random_split_nonpositive", "partition_lengths",
                           detail, status, witness)

    # S3 -- out-of-range scalar literal -> bounds ---------------------------
    def _scan_bounds(self, tree: ast.AST) -> None:
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            fname = _call_func_name(node)
            if fname not in _BOUNDED_ARGS:
                continue
            spec = _BOUNDED_ARGS[fname]
            line = getattr(node, "lineno", 0)
            # keyword args
            for kw in node.keywords:
                if kw.arg in spec:
                    val = _literal_number(kw.value)
                    if val is None:
                        continue
                    lo, hi = spec[kw.arg]
                    self._emit_bound(line, fname, kw.arg, val, lo, hi)
            # positional args (only index 0 is registered)
            if node.args and 0 in spec:
                val = _literal_number(node.args[0])
                if val is not None:
                    lo, hi = spec[0]
                    self._emit_bound(line, fname, "0", val, lo, hi)

    def _emit_bound(self, line: int, fname: str, arg: str, val: float,
                    lo: float, hi: float) -> None:
        status, witness = self._certify(
            "schema", "bounds", lower=lo, upper=hi, values=[val])
        detail = (f"`{fname}(...)` argument `{arg}={val}` is outside its admissible "
                  f"range [{lo}, {hi}].")
        self._emit(line, "S3:out_of_range_argument", "bounds", detail, status, witness)


def scan_source(source: str, filename: str = "<string>") -> list[ContractFinding]:
    """Scan one Python source string; return certifier-rejected contract findings."""
    return SplitContractScanner(source, filename).scan()


def scan_path(path: str | Path) -> list[ContractFinding]:
    """Scan a file or directory tree of ``*.py`` files."""
    p = Path(path)
    out: list[ContractFinding] = []
    files: Iterable[Path] = [p] if p.is_file() else p.rglob("*.py")
    for f in files:
        try:
            out.extend(scan_source(f.read_text(encoding="utf-8", errors="ignore"), str(f)))
        except (OSError, UnicodeDecodeError):
            continue
    return out


# Distinct top-level aliases (``scan_source``/``scan_path`` are already public
# names owned by datarefine.pipeline_leakage).
scan_split_contracts = scan_source
scan_split_contracts_path = scan_path
