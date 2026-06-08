"""Static scanner for *group leakage* (quotient-disjointness contract).

Where :mod:`datarefine.split_contracts` certifies **split disjointness** over raw
row indices and :mod:`datarefine.pipeline_leakage` certifies **fit/transform
isolation** over rows, this scanner certifies the single most impactful
generalisation of those contracts for the data-processing layer: **quotient
disjointness** --- disjointness of the train/test partitions *after collapsing
the rows by an equivalence relation* ``~`` induced by a *group key*.

    Group leakage: when rows are not independent (multiple rows share a
    *patient*, *user*, *speaker*, *molecule scaffold*, *session*, *device*, ...),
    a row-level random split (``train_test_split``, ``KFold``, ``random_split``,
    ``df.sample(frac=...)``) puts rows of the **same group** in both train and
    test.  The model memorises the group, the held-out metric is silently
    inflated, and nothing --- no test, no type checker, no runtime exception ---
    reveals it.  The fix is a *group-aware* splitter (``GroupKFold``,
    ``GroupShuffleSplit``, ``StratifiedGroupKFold``) or a ``groups=`` argument.

The contract lattice move is exact: ``split_disjointness`` is the special case of
``group_disjointness`` where ``~`` is the **identity** relation (every row its own
singleton group).  Lifting ``~`` from identity to a coarser grouping is what turns
"the same *row* is in two partitions" into "the same *group* is in two
partitions".  z3 decides the lifted obligation by searching an abstract
member->partition assignment for a straddling group (see
:func:`datarefine.smt_backend._decide_group`).

As with the other scanners we **do not model pandas / numpy / Python semantics**.
Group leakage is recognised from two *local syntactic signals*:

1. a **group key is visibly present** in the module --- a frame subscript with a
   group-role name (``df['patient_id']``), a ``groupby('user')`` key, or a
   variable named like a group (``groups``, ``speaker_ids``); and
2. a **group-blind row-level split** is performed --- ``train_test_split`` /
   ``random_split`` / a non-group CV splitter (``KFold``, ``ShuffleSplit``, ...) /
   ``df.sample(frac=...)`` --- *without* a group-aware splitter or a ``groups=``
   argument.

When both hold, the rows carry a grouping the split ignores, so the certifier is
asked to decide quotient disjointness with ``group_aware=False`` and rejects with
a concrete witness.  A call that *does* use a group-aware splitter or passes
``groups=`` is certified ``group_aware=True`` --- z3 admits it (a true negative),
so the scanner stays silent.  This is precision-first: with no visible group key
the module is silent regardless of how it splits.
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
    "GroupFinding",
    "GroupLeakageScanner",
    "scan_source",
    "scan_path",
    "scan_tree",
]

_CERT = StructuralCertifier()

# Word-stems that denote a *grouping* of correlated rows (not a row identifier).
# A token is a group key iff one of its ``_``/case-delimited words is a stem
# (singular or plural).  ``id``/``index``/``row`` are deliberately excluded: a
# bare row id is the identity relation, already covered by split_disjointness.
_GROUP_STEMS = frozenset({
    "patient", "subject", "speaker", "user", "customer", "client", "account",
    "household", "session", "series", "entity", "group", "grp", "cluster",
    "scaffold", "molecule", "compound", "ligand", "protein", "store", "shop",
    "sku", "device", "sensor", "study", "site", "donor", "individual",
    "person", "player", "team", "author", "seller", "driver", "trip",
    "trajectory", "video", "audio", "recording", "document", "case",
    "encounter", "visit", "block", "batch", "plate", "well", "voter",
})

_WORD_SPLIT = re.compile(r"[^a-z0-9]+")

# An assignment-target name is only trusted as a group signal when it both names
# a group role *and* looks like a key holder (ends in a plural / id / key / col /
# group suffix).  This rejects homographs used as ordinary scalars (``study`` =
# an Optuna study, ``session`` = an HTTP session, ``case`` = a switch case) which
# would otherwise be false positives, while keeping ``groups``, ``speaker_ids``,
# ``patient_id``, ``user_group``.  Frame subscripts and ``groupby`` keys do not
# need this guard --- a string column key is already concrete evidence.
_KEY_HOLDER_RE = re.compile(
    r"(?:^groups?$|s$|_ids?$|_keys?$|_col(?:umn)?$|_groups?$|_labels?$)",
    re.IGNORECASE,
)

# Row-level, group-blind splitters: each partitions *rows* with no notion of a
# group, so a multi-row group straddles the cut.
_ROW_SPLITTERS = frozenset({"train_test_split", "random_split"})
_BLIND_CV = frozenset({
    "KFold", "StratifiedKFold", "ShuffleSplit", "StratifiedShuffleSplit",
    "RepeatedKFold", "RepeatedStratifiedKFold",
})
# Group-aware splitters: keep every row of a group inside one partition.
_GROUP_AWARE = frozenset({
    "GroupKFold", "GroupShuffleSplit", "StratifiedGroupKFold",
    "LeaveOneGroupOut", "LeavePGroupsOut",
})


@dataclass
class GroupFinding:
    file: str
    line: int
    pattern: str
    constraint: str
    verdict: str
    detail: str
    snippet: str
    group_key: str
    splitter: str
    witness: dict | None = None

    def as_dict(self) -> dict:
        return asdict(self)


def _name_of(node: ast.AST | None) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return None


def _group_role(token: str) -> bool:
    """Whether ``token`` names a grouping of correlated rows."""
    for word in _WORD_SPLIT.split(token.lower()):
        if not word:
            continue
        if word in _GROUP_STEMS or word.rstrip("s") in _GROUP_STEMS:
            return True
    return False


def _str_const(node: ast.AST | None) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None


def _has_groups_kwarg(call: ast.Call) -> bool:
    """A ``groups=`` / ``group=`` keyword threads the grouping into the split."""
    return any(kw.arg in ("groups", "group") and kw.value is not None
               for kw in call.keywords)


def _is_subsample(call: ast.Call) -> bool:
    """``df.sample(frac=f)`` with ``0 < f < 1`` is a row-level split; ``frac=1``
    (or ``frac=1.0``) is a full shuffle and ``n=`` without ``frac`` is ambiguous,
    so neither counts as a split."""
    for kw in call.keywords:
        if kw.arg == "frac" and isinstance(kw.value, ast.Constant):
            try:
                f = float(kw.value.value)
            except (TypeError, ValueError):
                return False
            return 0.0 < f < 1.0
    return False


class GroupLeakageScanner:
    def __init__(self, source: str, filename: str = "<string>") -> None:
        self.source = source
        self.file = filename
        self.src_lines = source.splitlines()
        self.findings: list[GroupFinding] = []

    # -- group-signal collection -------------------------------------------
    def _collect_group_keys(self, tree: ast.AST) -> list[str]:
        """Group-role tokens visibly present in the module (de-duplicated,
        first-seen order)."""
        keys: list[str] = []
        seen: set[str] = set()

        def add(tok: str | None) -> None:
            if tok and _group_role(tok) and tok not in seen:
                seen.add(tok)
                keys.append(tok)

        for node in ast.walk(tree):
            # df['patient_id'] / data["speaker"]
            if isinstance(node, ast.Subscript):
                add(_str_const(node.slice))
            # df.groupby('user') / groupby(by='user') / groupby(['a','user'])
            elif isinstance(node, ast.Call) and _name_of(node.func) == "groupby":
                args: list[ast.AST] = list(node.args)
                args += [kw.value for kw in node.keywords if kw.arg in ("by", "level")]
                for a in args:
                    add(_str_const(a))
                    if isinstance(a, (ast.List, ast.Tuple)):
                        for el in a.elts:
                            add(_str_const(el))
            # groups = ...  / speaker_ids = ...  (assignment target named as a
            # group *key holder*; bare homographs like ``study`` are rejected)
            elif isinstance(node, ast.Assign):
                for tgt in node.targets:
                    name = _name_of(tgt)
                    if name and _KEY_HOLDER_RE.search(name):
                        add(name)
            elif isinstance(node, ast.AnnAssign):
                name = _name_of(node.target)
                if name and _KEY_HOLDER_RE.search(name):
                    add(name)
        return keys

    # -- certification ------------------------------------------------------
    def _certify(self, key: str, aware: bool) -> tuple[str, dict | None]:
        ob = obligation(
            "group", self.file, "group_disjointness",
            constraint="group_disjointness",
            group_key=key, group_size=2, partitions=2, group_aware=aware,
        )
        verdict = _CERT.certify(ob)
        witness = None
        diags = verdict.diagnostics or ()
        if diags and isinstance(diags[0], dict):
            witness = diags[0].get("model")
        return verdict.status, witness

    def _emit(self, line: int, key: str, splitter: str,
              status: str, witness: dict | None) -> None:
        if status != "rejected":
            return
        snippet = self.src_lines[line - 1].strip() if 0 < line <= len(self.src_lines) else ""
        detail = (
            f"row-level split via {splitter}() ignores the group key {key!r} "
            f"present in this module; rows sharing a {key} value can land in both "
            f"train and test, so the held-out score is inflated by group "
            f"memorisation. Use a group-aware splitter (GroupKFold / "
            f"GroupShuffleSplit) or pass groups={key}."
        )
        self.findings.append(GroupFinding(
            file=self.file, line=line, pattern="G1:group_member_split_across_partitions",
            constraint="group_disjointness", verdict=status, detail=detail,
            snippet=snippet, group_key=key, splitter=splitter, witness=witness))

    def scan(self) -> list[GroupFinding]:
        try:
            tree = ast.parse(self.source, filename=self.file)
        except SyntaxError:
            return []
        group_keys = self._collect_group_keys(tree)
        if not group_keys:
            return []  # no visible grouping -> identity relation -> silent
        key = group_keys[0]
        # A locally-defined function shadowing a splitter name (e.g. a custom
        # session/time-aware ``train_test_split``) has its own semantics; the
        # library catalog no longer applies, so we skip such calls.  The same
        # holds for a splitter name imported from a non-sklearn module.
        local_defs = {n.name for n in ast.walk(tree)
                      if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}
        for n in ast.walk(tree):
            if isinstance(n, ast.ImportFrom):
                mod = n.module or ""
                if "sklearn" not in mod and "torch" not in mod:
                    for alias in n.names:
                        local_defs.add(alias.asname or alias.name)

        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            callee = _name_of(node.func)
            if callee is None or callee in local_defs:
                continue
            splitter: str | None = None
            if callee in _ROW_SPLITTERS or callee in _BLIND_CV:
                splitter = callee
            elif callee == "sample" and _is_subsample(node):
                splitter = "sample"
            if splitter is None:
                continue
            # A call that threads the grouping in is group-aware -> z3 admits.
            aware = _has_groups_kwarg(node)
            line = getattr(node, "lineno", 0)
            status, witness = self._certify(key, aware)
            self._emit(line, key, splitter, status, witness)
        return self.findings


def scan_source(source: str, filename: str = "<string>") -> list[GroupFinding]:
    return GroupLeakageScanner(source, filename).scan()


def scan_path(path: str | Path) -> list[GroupFinding]:
    p = Path(path)
    try:
        source = p.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return []
    return scan_source(source, filename=str(p))


def scan_tree(root: str | Path) -> list[GroupFinding]:
    root = Path(root)
    out: list[GroupFinding] = []
    paths: Iterable[Path] = [root] if root.is_file() else root.rglob("*.py")
    for p in paths:
        out.extend(scan_path(p))
    return out
