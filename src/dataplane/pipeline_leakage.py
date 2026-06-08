"""Static train/test-leakage scanner for real Python training scripts.

This is a first-class DataRefine capability: it locates the dominant
preprocessing / featurization leakage patterns by a *local* AST signature, then
constructs a :func:`fit_transform_isolation` obligation for each candidate and
asks the :class:`StructuralCertifier` to decide it. The verdict on every finding
is produced by DataRefine's certifier, not by the scanner -- the scanner only
locates candidate sites and builds the obligation.

The decidable insight that makes this tractable without a full Python semantic
verifier: a transformer / resampler / manual normalisation that runs *before*
the train/test split, on data that then flows into ``train_test_split``,
necessarily consumed the held-out rows. That is a finite set-intersection fact,
which the structural certifier already decides.

Detected patterns (all grounded in ``fit_transform_isolation``):

* ``A`` -- ``Xs = scaler.fit_transform(X)`` whose output flows into the split.
* ``B`` -- ``scaler.fit(X)`` where the same ``X`` is a split input.
* ``C`` -- ``X = (X - X.mean()) / X.std()`` (manual full-data scaling) then split.
* ``D`` -- ``scaler.fit(pd.concat([train, test]))`` (fit on combined frames).
* ``E`` -- ``X_res, y_res = SMOTE().fit_resample(X, y)`` then split.

A ``model.fit(X_train, y_train)`` is never flagged: ``X_train`` is a split
*output*, so it can never appear among the split inputs.

PyTorch coverage
----------------

The split site is not limited to scikit-learn's ``train_test_split``. PyTorch
training scripts overwhelmingly split with ``torch.utils.data.random_split`` (or
``Subset``) over a ``TensorDataset`` / ``DataLoader`` that wraps the very tensor
that was normalised. The scanner therefore also treats ``random_split`` /
``Subset`` as split functions and looks *through* dataset wrappers
(``TensorDataset``, ``DataLoader``, ``Subset``, ``ConcatDataset``,
``torch.tensor(...)``) so that, e.g.::

    X = scaler.fit_transform(X)                  # full-data fit  (leak)
    ds = TensorDataset(torch.tensor(X), torch.tensor(y))
    train_ds, val_ds = random_split(ds, [800, 200])

is recognised: ``X`` (the fit output) flows through ``ds`` into ``random_split``,
so the validation tensors were scaled with their own statistics. The same holds
for manual ``(X - X.mean()) / X.std()`` normalisation before ``random_split``.
"""
from __future__ import annotations

import ast
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable

from .certification import StructuralCertifier
from .obligations import obligation

__all__ = [
    "LeakageFinding",
    "PipelineLeakageScanner",
    "scan_source",
    "scan_path",
]

_FIT_ATTRS = frozenset({"fit", "fit_transform"})
# Methods that *apply* a previously-fitted estimator's state to data. If the
# estimator was fit on the full dataset, these outputs carry the leak.
_TRANSFORM_ATTRS = frozenset({"transform", "predict", "predict_proba",
                              "decision_function", "predict_log_proba"})
_RESAMPLE_ATTRS = frozenset({"fit_resample", "fit_sample"})
_STATS_ATTRS = frozenset({"mean", "std", "min", "max", "median", "var", "quantile", "sum"})
# Statistics that, computed over a full dataset before a split, constitute a
# normalisation leak (the torch `transforms.Normalize` family). `sum` is excluded
# because it is rarely a normalisation constant and shows up in unrelated code.
_NORM_STAT_ATTRS = frozenset({"mean", "std", "var", "median", "quantile", "min", "max"})
# Numeric-library module roots whose `mod.mean(data)` form summarises the
# *argument*, not the module -- so `np`/`torch` must never be read as the data.
_NUMERIC_MODULES = frozenset({"np", "numpy", "torch", "pd", "pandas", "cp", "jnp", "tf"})
_CONCAT_NAMES = frozenset({"concat", "vstack", "hstack", "concatenate", "append"})
# Calls that *merge* tensors/arrays back into one -- the inverse of a split.
# `torch.cat([train_x, test_x])` reconstitutes the full dataset, so any statistic
# computed over the merge has seen every partition (including the held-out one).
_MERGE_NAMES = frozenset({"cat", "concatenate", "vstack", "hstack", "stack", "concat"})
import re as _re
# A variable name that plays an explicit train / val / test / holdout partition
# role. Used to recognise a `cat([train_x, test_x])` partition-merge with high
# precision (so model-internal `torch.cat([out1, out2])` is never mistaken for it).
_SPLIT_ROLE_RE = _re.compile(
    r"(?:^|_)(train|trn|tr|val|valid|validation|test|tst|holdout|hold_out|dev|eval)(?:_|$|\d)",
    _re.IGNORECASE)


def _split_role(name: str | None) -> str | None:
    if not name:
        return None
    m = _SPLIT_ROLE_RE.search(name)
    return m.group(1).lower() if m else None
_SPLIT_NAMES = frozenset({"train_test_split"})
_TORCH_SPLIT_NAMES = frozenset({"random_split", "Subset"})
_CV_NAMES = frozenset({
    "cross_val_score", "cross_validate", "cross_val_predict",
})
_DATASET_WRAPPERS = frozenset({
    "TensorDataset", "DataLoader", "Subset", "ConcatDataset",
    "StackDataset", "tensor", "as_tensor", "from_numpy", "Tensor",
})
_TRANSFORMER_HINTS = (
    "scal", "encod", "imput", "transform", "pca", "normal", "vector",
    "select", "binar", "discretiz", "poly", "minmax", "standard", "robust",
    "power", "quantil", "tfidf", "count", "ohe", "onehot", "preprocess",
)

_CERT = StructuralCertifier()


@dataclass(frozen=True)
class LeakageFinding:
    """One certified train/test-leakage site in a source file."""

    file: str
    line: int
    pattern: str
    estimator: str
    data_var: str
    split_line: int
    snippet: str
    constraint: str
    verdict: str
    explanation: str

    @property
    def is_leak(self) -> bool:
        return self.verdict == "rejected"

    def as_dict(self) -> dict:
        return asdict(self)


def _name_of(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    if isinstance(node, ast.Subscript):
        return _name_of(node.value)
    return None


def _root_name(node: ast.AST) -> str | None:
    """The *root* variable a chained expression hangs off, so the receiver of
    ``dataset.data.float().mean()`` resolves to ``dataset`` (not ``data``). Used
    to attribute a normalisation statistic to the dataset it is computed over."""
    cur = node
    while True:
        if isinstance(cur, ast.Name):
            return cur.id
        if isinstance(cur, ast.Attribute):
            cur = cur.value
        elif isinstance(cur, ast.Subscript):
            cur = cur.value
        elif isinstance(cur, ast.Call):
            cur = cur.func
        else:
            return None


def _call_func_name(call: ast.Call) -> str | None:
    f = call.func
    if isinstance(f, ast.Name):
        return f.id
    if isinstance(f, ast.Attribute):
        return f.attr
    return None


def _receiver_name(call: ast.Call) -> str:
    f = call.func
    if isinstance(f, ast.Attribute):
        return _name_of(f.value) or "<estimator>"
    return "<estimator>"


def _looks_like_transformer(attr: str, receiver: str, known_vars: frozenset[str]) -> bool:
    if attr == "fit_transform":
        return True
    if receiver in known_vars:
        return True
    r = receiver.lower()
    return any(h in r for h in _TRANSFORMER_HINTS)


def _collect_transformer_vars(tree: ast.AST) -> frozenset[str]:
    out: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Call):
            cname = _call_func_name(node.value)
            if cname and any(h in cname.lower() for h in _TRANSFORMER_HINTS):
                for tgt in node.targets:
                    nm = _name_of(tgt)
                    if nm:
                        out.add(nm)
    return frozenset(out)


def _names_within(node: ast.AST) -> set[str]:
    """Every bare ``Name`` id appearing in an expression subtree (so a tensor
    wrapped in ``TensorDataset(torch.tensor(X), ...)`` still yields ``X``)."""
    return {n.id for n in ast.walk(node) if isinstance(n, ast.Name)}


def _scope_nodes(body: list[ast.stmt]) -> list[ast.AST]:
    """All AST nodes belonging to *this* scope, NOT descending into nested
    function / class / lambda bodies (those are analysed as separate scopes).

    This is the precision guard that stops a ``fit`` in one function from being
    correlated with a ``split`` in an unrelated function of the same module --
    the failure mode that makes large library files explode with false
    positives."""
    out: list[ast.AST] = []
    stack: list[ast.AST] = list(body)
    while stack:
        node = stack.pop()
        out.append(node)
        # A nested function / class / lambda opens a *new* scope; it is analysed
        # separately, so do not descend into its body here. (Checking the popped
        # node -- not its children -- is essential: the children of a FunctionDef
        # are ordinary statements, so a child-only check would still walk one
        # level into every nested function and re-introduce cross-scope leakage.)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef,
                             ast.ClassDef, ast.Lambda)):
            continue
        for child in ast.iter_child_nodes(node):
            stack.append(child)
    return out


def _collect_dataset_vars(nodes: Iterable[ast.AST]) -> dict[str, set[str]]:
    """Map a dataset/loader variable to the tensor names it wraps.

    ``ds = TensorDataset(torch.tensor(X), torch.tensor(y))`` -> ``{ds: {X, y}}``.
    These let the scanner follow data through a wrapper into ``random_split``."""
    out: dict[str, set[str]] = {}
    for node in nodes:
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Call):
            fn = _call_func_name(node.value)
            if fn in _DATASET_WRAPPERS:
                inner: set[str] = set()
                for a in node.value.args:
                    inner |= _names_within(a)
                for tgt in node.targets:
                    elts = tgt.elts if isinstance(tgt, ast.Tuple) else [tgt]
                    for e in elts:
                        if (nm := _name_of(e)):
                            out[nm] = inner
    return out


def _expand_dataset_vars(names: set[str], dataset_vars: dict[str, set[str]]) -> set[str]:
    """Resolve dataset-wrapper variables to the tensor names they carry."""
    extra: set[str] = set()
    for nm in names:
        extra |= dataset_vars.get(nm, set())
    return names | extra


def _derivation_map(nodes: Iterable[ast.AST]) -> dict[str, set[str]]:
    """Map a variable to the root names of the expression it was assigned from:
    ``Xn = (X - mean) / std`` -> ``{Xn: {X, mean, std}}``. Used to follow a
    normalisation statistic computed over ``X`` forward into a ``Xn`` that is
    wrapped and split."""
    out: dict[str, set[str]] = {}
    for node in nodes:
        if isinstance(node, ast.Assign):
            roots = {n.id for n in ast.walk(node.value) if isinstance(n, ast.Name)}
            for tgt in node.targets:
                elts = tgt.elts if isinstance(tgt, ast.Tuple) else [tgt]
                for e in elts:
                    if (nm := _name_of(e)):
                        out.setdefault(nm, set()).update(roots)
    return out


def _backward_closure(names: set[str], deriv: dict[str, set[str]]) -> set[str]:
    """All variables that ``names`` were (transitively) derived from."""
    seen: set[str] = set()
    stack = list(names)
    while stack:
        n = stack.pop()
        if n in seen:
            continue
        seen.add(n)
        stack.extend(src for src in deriv.get(n, ()) if src not in seen)
    return seen


def _binding_lines(nodes: Iterable[ast.AST]) -> dict[str, list[int]]:
    """Map a plain variable name to every line where it is *freshly* re-bound to a
    value that does **not** depend on its own prior value (``x = df.iloc[...]``,
    ``for x in ...``, ``with ... as x``).

    Self-derived rebindings (``x = scaler.fit_transform(x)``, ``x += 1``) are
    deliberately *excluded*: they transform the same data, so the flow from an
    earlier fit to a later split is preserved through them.

    Used as a data-flow precision guard: a ``fit`` on ``x`` and a later ``split``
    of ``x`` are only the same data if ``x`` was not *freshly* rebound in between.
    This kills the dominant false-positive class (a tutorial that fits a scaler on
    one ``X`` -- e.g. a single column -- then pages later writes ``X = df.iloc[...]``
    before an unrelated ``train_test_split``) while still flagging a chain of
    self-derived transforms (``X = imputer-impute(X); X = ohe.fit_transform(X)``)
    that carries leaked statistics into the split."""
    out: dict[str, list[int]] = {}

    def _mentions(name: str, value: ast.AST | None) -> bool:
        if value is None:
            return False
        return any(isinstance(n, ast.Name) and n.id == name for n in ast.walk(value))

    def _add_fresh(target: ast.AST | None, line: int, value: ast.AST | None) -> None:
        # Only a *full* rebind of a plain Name that does NOT reference the
        # variable itself breaks the fit->split dataflow. Subscript / attribute
        # targets (``X[:, 0] = ...``) are in-place partial mutations, not fresh
        # rebinds, so they never break the flow.
        if isinstance(target, ast.Name) and not _mentions(target.id, value):
            out.setdefault(target.id, []).append(line)

    for node in nodes:
        line = getattr(node, "lineno", 0)
        if isinstance(node, ast.Assign):
            for tgt in node.targets:
                elts = tgt.elts if isinstance(tgt, ast.Tuple) else [tgt]
                for e in elts:
                    _add_fresh(e, line, node.value)
        elif isinstance(node, ast.AnnAssign):
            _add_fresh(node.target, line, node.value)
        # AugAssign (x += ...) is inherently self-referential -> never breaks flow.
        elif isinstance(node, (ast.For, ast.AsyncFor)):
            tgt = node.target
            elts = tgt.elts if isinstance(tgt, ast.Tuple) else [tgt]
            for e in elts:
                _add_fresh(e, line, None)
        elif isinstance(node, ast.withitem) and node.optional_vars is not None:
            _add_fresh(node.optional_vars, line, None)
    return out


def _rebound_between(var: str, lo: int, hi: int, binds: dict[str, list[int]]) -> bool:
    """True if ``var`` is re-bound to a fresh value strictly between lines ``lo``
    and ``hi`` -- i.e. the data observed by a fit at ``lo`` is not the data split
    at ``hi``."""
    return any(lo < ln < hi for ln in binds.get(var, ()))


def _partition_ids(tag: str | int) -> tuple[str, str]:
    """The two disjoint row partitions a split at ``tag`` produces."""
    return f"rows::{tag}::train", f"rows::{tag}::holdout"


def _certify(estimator: str, fit_row_ids: list[str], holdout_row_ids: list[str],
             fit_sources: tuple[str, ...] = (), outcomes: tuple[str, ...] = ()) -> str:
    """Hand the observed provenance to the real :class:`StructuralCertifier` and
    return *its* verdict. The scanner does NOT decide leak-vs-clean: it only
    reports which row partitions the fit actually consumed (a fit on the full
    pre-split data carries the holdout partition id; a fit on an already-split
    train partition does not). The certifier then decides by computing
    ``disjoint(fit_row_ids, holdout_row_ids)`` -- so a clean split-then-fit
    pipeline is *certified*, and only a genuine overlap is *rejected*."""
    ob = obligation(
        "provenance", estimator, "fit_isolation",
        constraint="fit_transform_isolation",
        fit_row_ids=list(fit_row_ids),
        holdout_row_ids=list(holdout_row_ids),
        fit_feature_sources=list(fit_sources),
        outcome_columns=list(outcomes),
    )
    return _CERT.certify(ob).status


class _ScopeVisitor:
    def __init__(self, file: str, src_lines: list[str], known_vars: frozenset[str]):
        self.file = file
        self.src_lines = src_lines
        self.known_vars = known_vars
        self.findings: list[LeakageFinding] = []

    def analyse(self, body: list[ast.stmt]) -> None:
        nodes = _scope_nodes(body)

        dataset_vars = _collect_dataset_vars(nodes)

        # Pattern I runs independently of an explicit split call: a torch.cat /
        # np.concatenate of named train/test partitions *is* the (inverse of a)
        # split, so it must be scanned even when no train_test_split appears.
        self._scan_concat_normalization(nodes)

        split_calls: list[tuple[int, set[str], str]] = []
        for node in nodes:
            if not isinstance(node, ast.Call):
                continue
            fn = _call_func_name(node)
            if fn in _SPLIT_NAMES:
                inputs = {nm for a in node.args if (nm := _name_of(a))}
            elif fn in _TORCH_SPLIT_NAMES and node.args:
                # random_split(dataset, [n_train, n_val]) / Subset(dataset, idx):
                # only the first positional arg is the dataset to follow.
                inputs = _names_within(node.args[0])
            elif fn in _CV_NAMES and (len(node.args) >= 2 or any(
                    kw.arg in ("X", "y") for kw in node.keywords)):
                # cross_val_score(estimator, X, y, ...) / cross_validate / GridSearchCV
                # re-split X internally on every fold. A transformer fit on the
                # whole X *before* the CV call therefore leaks held-out folds into
                # the fitted state -- the canonical "scale-then-cross-validate"
                # bug. The first positional arg is the estimator, so the data are
                # the remaining positional args (plus an X=/y= keyword form).
                inputs = set()
                for a in node.args[1:]:
                    inputs |= _names_within(a)
                for kw in node.keywords:
                    if kw.arg in ("X", "y"):
                        inputs |= _names_within(kw.value)
            else:
                continue
            inputs = _expand_dataset_vars(inputs, dataset_vars)
            split_calls.append((getattr(node, "lineno", 0), inputs, fn or "split"))
        if not split_calls:
            return

        fit_calls: list[tuple[int, str, str, str]] = []
        for node in nodes:
            if not isinstance(node, ast.Call):
                continue
            attr = _call_func_name(node)
            if attr not in _FIT_ATTRS or not node.args:
                continue
            recv = _receiver_name(node)
            if not _looks_like_transformer(attr, recv, self.known_vars):
                continue
            dv = _name_of(node.args[0])
            if dv:
                fit_calls.append((getattr(node, "lineno", 0), attr, dv, recv))

        produced_by_ft: dict[str, tuple[int, str, str]] = {}
        for node in nodes:
            if isinstance(node, ast.Assign) and isinstance(node.value, ast.Call):
                call = node.value
                if _call_func_name(call) == "fit_transform" and call.args:
                    dv = _name_of(call.args[0])
                    recv = _receiver_name(call)
                    for tgt in node.targets:
                        lhs = _name_of(tgt)
                        if lhs and dv:
                            produced_by_ft[lhs] = (getattr(node, "lineno", 0), dv, recv)

        # Variables that are *outputs* of a split (e.g. `X_train` from
        # train_test_split / random_split). Fitting a transformer on such a
        # partition is the CORRECT order, not a leak -- so a fit whose data
        # variable is itself a split output must never be flagged. This is the
        # precision guard that prevents large multi-branch loaders (one big
        # function reusing names like `x_trn` across mutually-exclusive dataset
        # branches) from producing cascades of false positives.
        split_output_vars: set[str] = set()
        for node in nodes:
            if isinstance(node, ast.Assign) and isinstance(node.value, ast.Call) \
                    and _call_func_name(node.value) in (_SPLIT_NAMES | _TORCH_SPLIT_NAMES):
                for tgt in node.targets:
                    elts = tgt.elts if isinstance(tgt, ast.Tuple) else [tgt]
                    for e in elts:
                        if (nm := _name_of(e)):
                            split_output_vars.add(nm)

        seen: set[tuple[int, int]] = set()

        # Lines where each variable is re-bound -- guards against pairing a fit
        # and a split that merely share a name across an intervening reassignment.
        binds = _binding_lines(nodes)
        # Backward-derivation map for the torch normalisation pattern (F).
        deriv = _derivation_map(nodes)
        # Variables passed into a `transforms.Normalize(...)`-style call -- the
        # lazy torch path by which a precomputed mean/std is applied to the data.
        norm_arg_vars: set[str] = set()
        for node in nodes:
            if isinstance(node, ast.Call):
                fn = _call_func_name(node)
                if fn and "normal" in fn.lower():
                    for a in node.args:
                        norm_arg_vars |= _names_within(a)
                    for kw in node.keywords:
                        norm_arg_vars |= _names_within(kw.value)
        # Variables that participate in normalisation *arithmetic* -- i.e. appear
        # inside a subtraction or division anywhere in the scope. A full-data
        # statistic only leaks if it is actually subtracted/divided into the data
        # (manual path); a stat that merely flows into a one-hot / index
        # construction (`arange(max(labels)+1) == labels`) never normalises.
        norm_op_vars: set[str] = set()
        for node in nodes:
            if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Sub, ast.Div)):
                norm_op_vars |= _names_within(node)

        # Pattern A
        for split_line, inputs, split_fn in split_calls:
            for var in inputs:
                if var in produced_by_ft:
                    ft_line, dv, recv = produced_by_ft[var]
                    if ft_line < split_line and (ft_line, split_line) not in seen \
                            and not _rebound_between(var, ft_line, split_line, binds):
                        seen.add((ft_line, split_line))
                        train_id, holdout_id = _partition_ids(split_line)
                        partitioned = dv in split_output_vars or var in split_output_vars
                        fit_ids = [train_id] if partitioned else [train_id, holdout_id]
                        self._emit("A:fit_transform_output_split", ft_line, recv, dv, split_line,
                                   f"`{var} = {recv}.fit_transform({dv})` is fit on the full data, "
                                   f"then `{var}` is split via `{split_fn}` at line {split_line}: "
                                   f"test-set statistics leak into the training features.",
                                   fit_ids, [holdout_id])

        # Pattern B
        for fit_line, attr, dv, recv in fit_calls:
            for split_line, inputs, split_fn in split_calls:
                # A cross-validation call that merely receives the *raw* `dv` is
                # not a leak: the estimator handed to CV is typically a Pipeline
                # that re-fits the transformer inside each fold, and any earlier
                # standalone `transformer.fit_transform(dv)` is a discarded demo.
                # Only pattern A (the transform *output* flowing into the CV) is a
                # genuine scale-then-cross-validate leak.
                if split_fn in _CV_NAMES:
                    continue
                if dv in inputs and fit_line < split_line and (fit_line, split_line) not in seen \
                        and not _rebound_between(dv, fit_line, split_line, binds):
                    seen.add((fit_line, split_line))
                    train_id, holdout_id = _partition_ids(split_line)
                    partitioned = dv in split_output_vars
                    fit_ids = [train_id] if partitioned else [train_id, holdout_id]
                    self._emit("B:fit_before_split", fit_line, recv, dv, split_line,
                               f"`{recv}.{attr}({dv})` at line {fit_line} fits on the full `{dv}`, "
                               f"which is then split via `{split_fn}` at line {split_line}: the "
                               f"transform saw every row, including the held-out test rows.",
                               fit_ids, [holdout_id])

        # Pattern C -- manual full-data normalisation
        for node in nodes:
            if not isinstance(node, ast.Assign):
                continue
            # Normalisation *subtracts or divides* the data by a full-data
            # statistic. Requiring a Sub/Div (not just any BinOp) excludes
            # one-hot / index constructions like `arange(max(labels)+1) == labels`,
            # which also reference a stat but do not leak a distribution.
            if not any(isinstance(s, ast.BinOp) and isinstance(s.op, (ast.Sub, ast.Div))
                       for s in ast.walk(node.value)):
                continue
            bases, bare = set(), set()
            for sub in ast.walk(node.value):
                if isinstance(sub, ast.Call) and isinstance(sub.func, ast.Attribute) \
                        and sub.func.attr in _STATS_ATTRS:
                    recv_root = _root_name(sub.func.value)
                    if recv_root in _NUMERIC_MODULES:
                        # function style `np.mean(data)` -> stat over the argument
                        if sub.args and (a := _root_name(sub.args[0])):
                            bases.add(a)
                    elif recv_root:
                        bases.add(recv_root)
                elif isinstance(sub, ast.Name):
                    bare.add(sub.id)
            self_norm = (bases & bare) - _NUMERIC_MODULES
            if not self_norm:
                continue
            base = sorted(self_norm)[0]
            line = getattr(node, "lineno", 0)
            for tgt in node.targets:
                t = _name_of(tgt)
                if not t:
                    continue
                for split_line, inputs, split_fn in split_calls:
                    if t in inputs and line < split_line and (line, split_line) not in seen \
                            and not _rebound_between(t, line, split_line, binds):
                        seen.add((line, split_line))
                        train_id, holdout_id = _partition_ids(split_line)
                        partitioned = t in split_output_vars
                        fit_ids = [train_id] if partitioned else [train_id, holdout_id]
                        self._emit("C:manual_full_data_scaling", line, "manual_scale", t, split_line,
                                   f"`{t}` is normalised using full-data statistics "
                                   f"(e.g. `{base}.mean()/{base}.std()`) at line {line}, then split "
                                   f"via `{split_fn}` at line {split_line}: the scaling statistics "
                                   f"were computed over the held-out rows too.",
                                   fit_ids, [holdout_id])

        # Pattern D -- fit on concatenated frames
        for node in nodes:
            if not isinstance(node, ast.Call):
                continue
            attr = _call_func_name(node)
            if attr not in _FIT_ATTRS or not node.args:
                continue
            recv = _receiver_name(node)
            if not _looks_like_transformer(attr, recv, self.known_vars):
                continue
            arg0 = node.args[0]
            if isinstance(arg0, ast.Call) and _call_func_name(arg0) in _CONCAT_NAMES:
                line = getattr(node, "lineno", 0)
                if (line, -1) not in seen:
                    seen.add((line, -1))
                    train_id, holdout_id = _partition_ids(f"concat@{line}")
                    self._emit("D:fit_on_concatenated_data", line, recv, "concat(...)", line,
                               f"`{recv}.{attr}(...)` at line {line} is fit on a concatenation of "
                               f"multiple frames (train+test combined): the fitted state is "
                               f"contaminated by the held-out rows.",
                               [train_id, holdout_id], [holdout_id])

        # Pattern E -- resampling before the split
        for node in nodes:
            if not (isinstance(node, ast.Assign) and isinstance(node.value, ast.Call)
                    and _call_func_name(node.value) in _RESAMPLE_ATTRS):
                continue
            recv = _receiver_name(node.value)
            outs: set[str] = set()
            for tgt in node.targets:
                elts = tgt.elts if isinstance(tgt, ast.Tuple) else [tgt]
                for e in elts:
                    if (nm := _name_of(e)):
                        outs.add(nm)
            line = getattr(node, "lineno", 0)
            for split_line, inputs, split_fn in split_calls:
                if (outs & inputs) and line < split_line and (line, split_line) not in seen:
                    seen.add((line, split_line))
                    train_id, holdout_id = _partition_ids(split_line)
                    self._emit("E:resample_before_split", line, recv,
                               sorted(outs & inputs)[0], split_line,
                               f"`{recv}.fit_resample(...)` at line {line} resamples the full "
                               f"dataset, and the result is split via `{split_fn}` at line "
                               f"{split_line}: synthetic / duplicated rows leak across the "
                               f"train/test boundary.",
                               [train_id, holdout_id], [holdout_id])

        # Pattern F -- torch full-dataset normalisation statistics before a split.
        # The canonical PyTorch leak: per-channel mean/std are computed over the
        # WHOLE dataset (or its backing tensor), wired into `transforms.Normalize`
        # / applied to the data, and only THEN is the dataset carved into
        # train/val/test with `random_split` / `Subset`. The normalisation
        # constants therefore saw every held-out sample. We anchor on the strong,
        # low-false-positive signal: a statistic (`.mean()/.std()/.var()/...`)
        # whose receiver root is a variable that is *itself* later split. A bare
        # `loss.mean()` is never flagged because `loss` is not a split input.
        for node in nodes:
            if not isinstance(node, ast.Assign):
                continue
            stat_bases: set[str] = set()
            for sub in ast.walk(node.value):
                if isinstance(sub, ast.Call) and isinstance(sub.func, ast.Attribute) \
                        and sub.func.attr in _NORM_STAT_ATTRS:
                    recv_root = _root_name(sub.func.value)
                    if recv_root in _NUMERIC_MODULES:
                        # function style `np.mean(data)` summarises the argument.
                        if sub.args and (a := _root_name(sub.args[0])):
                            stat_bases.add(a)
                    elif recv_root:
                        stat_bases.add(recv_root)
            stat_bases -= _NUMERIC_MODULES
            if not stat_bases:
                continue
            stat_targets = {nm for tgt in node.targets
                            for e in (tgt.elts if isinstance(tgt, ast.Tuple) else [tgt])
                            if (nm := _name_of(e))}
            line = getattr(node, "lineno", 0)
            for split_line, inputs, split_fn in split_calls:
                if split_fn in _CV_NAMES:
                    continue
                # Follow the stat base forward through derived tensors
                # (`Xn = (X - mean)/std; ds = TensorDataset(Xn, y)`) into the split.
                reachable = _backward_closure(inputs, deriv)
                hit = (stat_bases & reachable) - split_output_vars
                if not hit:
                    continue
                # Precision anchor: a statistic computed over the full data is a
                # *leak* only if it is actually USED to normalise the data --
                # either fed into a `transforms.Normalize(...)` (the lazy torch
                # path) or flowing back into the tensors that are split (the
                # manual path). A bare unused `summary = df.mean()` is benign.
                used = bool(stat_targets & norm_arg_vars) or \
                    (bool(stat_targets & reachable) and bool(stat_targets & norm_op_vars))
                if not used:
                    continue
                base = sorted(hit)[0]
                if line < split_line and (line, split_line) not in seen \
                        and not _rebound_between(base, line, split_line, binds):
                    seen.add((line, split_line))
                    train_id, holdout_id = _partition_ids(split_line)
                    self._emit("F:torch_full_dataset_normalization", line, "normalize", base,
                               split_line,
                               f"normalisation statistics (e.g. `{base}.mean()/{base}.std()`) "
                               f"are computed over the full `{base}` at line {line}, which is then "
                               f"split via `{split_fn}` at line {split_line}: the mean/std saw every "
                               f"held-out sample, so the normalisation leaks the validation/test "
                               f"distribution into training.",
                               [train_id, holdout_id], [holdout_id])

        # Pattern G -- two-step fit()/transform(): a transformer is `.fit()` on the
        # full data and then a SEPARATE `.transform(...)` call (or .predict /
        # .decision_function) produces the array that is split. Patterns A and B
        # only see `fit_transform(dv)` or `fit(dv)` where `dv` itself is split;
        # the very common `pca.fit(X); Xp = pca.transform(X); split(Xp)` form slips
        # past both because the split input (`Xp`) is neither the fit argument nor
        # a fit_transform output. The fitted state still saw every held-out row.
        transform_outputs: dict[str, list[tuple[str, int]]] = {}
        for node in nodes:
            if isinstance(node, ast.Assign) and isinstance(node.value, ast.Call):
                call = node.value
                if _call_func_name(call) in _TRANSFORM_ATTRS:
                    rc = _receiver_name(call)
                    if rc:
                        for tgt in node.targets:
                            if (lhs := _name_of(tgt)):
                               transform_outputs.setdefault(rc, []).append(
                                   (lhs, getattr(node, "lineno", 0)))
        for fit_line, attr, dv, recv in fit_calls:
            if attr != "fit":          # fit_transform is already pattern A
                continue
            if dv in split_output_vars:  # fit on a train subset -> correct order
                continue
            for out_var, t_line in transform_outputs.get(recv, []):
                if t_line < fit_line:    # transform must follow the fit
                    continue
                for split_line, inputs, split_fn in split_calls:
                    if split_fn in _CV_NAMES:
                        continue
                    reachable = _backward_closure(inputs, deriv)
                    if out_var not in (reachable | inputs):
                        continue
                    if t_line < split_line and (fit_line, split_line) not in seen \
                            and not _rebound_between(dv, fit_line, split_line, binds):
                        seen.add((fit_line, split_line))
                        train_id, holdout_id = _partition_ids(split_line)
                        self._emit("G:fit_then_transform_split", fit_line, recv, dv, split_line,
                                  f"`{recv}.fit({dv})` at line {fit_line} fits on the full `{dv}`; "
                                  f"`{out_var} = {recv}.transform(...)` then feeds the split via "
                                  f"`{split_fn}` at line {split_line}. Although `fit` and `transform` "
                                  f"are separate calls, the fitted state still saw every held-out "
                                  f"row, so the transformed features leak the test distribution.",
                                  [train_id, holdout_id], [holdout_id])

        # Pattern H -- full-data mean / target *encoding* before a split.
        # `df['enc'] = df.groupby('cat')['target'].transform('mean')` (or the
        # two-step `m = df.groupby('cat')['y'].mean().to_dict(); df['enc'] =
        # df['cat'].map(m)`) computes a per-category aggregate over the WHOLE
        # frame and BROADCASTS it back onto every row as a new feature; splitting
        # afterwards leaks the held-out rows' target distribution into the encoded
        # training column. The defining signal is the *broadcast back to rows*
        # (`groupby(...).transform(...)` or `.map(<groupby-derived dict>)`), which
        # preserves the row count. A plain `df.groupby([...]).mean().reset_index()`
        # merely *collapses* repeated measurements into one row per group (each
        # output row uses only its own group's raw values) and is NOT a leak -- so
        # the aggregate-then-collapse form must stay silent.
        groupby_derived: set[str] = set()
        for node in nodes:
            if isinstance(node, ast.Assign) and any(
                    isinstance(s, ast.Call) and isinstance(s.func, ast.Attribute)
                    and s.func.attr == "groupby" for s in ast.walk(node.value)):
                for tgt in node.targets:
                    if (nm := _name_of(tgt)):
                        groupby_derived.add(nm)

        def _encoding_base(value: ast.AST) -> str | None:
            for sub in ast.walk(value):
                if not (isinstance(sub, ast.Call) and isinstance(sub.func, ast.Attribute)):
                    continue
                # broadcast form: groupby(...)...transform(...)
                if sub.func.attr == "transform":
                    for inner in ast.walk(sub.func.value):
                        if isinstance(inner, ast.Call) and isinstance(inner.func, ast.Attribute) \
                                and inner.func.attr == "groupby":
                            return _root_name(inner.func.value)
                # map form: <col>.map(<groupby-derived mapping>)
                if sub.func.attr == "map" and sub.args:
                    a0 = sub.args[0]
                    if (isinstance(a0, ast.Name) and a0.id in groupby_derived) or any(
                            isinstance(s, ast.Call) and isinstance(s.func, ast.Attribute)
                            and s.func.attr == "groupby" for s in ast.walk(a0)):
                        return _root_name(sub.func.value)
            return None

        for node in nodes:
            if not isinstance(node, ast.Assign):
                continue
            gb_base = _encoding_base(node.value)
            if not gb_base:
                continue
            tgt_roots: set[str] = set()
            for tgt in node.targets:
                if isinstance(tgt, ast.Subscript):
                    if (r := _root_name(tgt.value)):
                        tgt_roots.add(r)
                elif (nm := _name_of(tgt)):
                    tgt_roots.add(nm)
            line = getattr(node, "lineno", 0)
            for split_line, inputs, split_fn in split_calls:
                if split_fn in _CV_NAMES:
                    continue
                reachable = _backward_closure(inputs, deriv) | inputs
                carrier = (tgt_roots | {gb_base}) & reachable
                if not carrier:
                    continue
                if line < split_line and (line, split_line) not in seen \
                        and not _rebound_between(gb_base, line, split_line, binds):
                    seen.add((line, split_line))
                    train_id, holdout_id = _partition_ids(split_line)
                    enc = sorted(carrier)[0]
                    self._emit("H:group_encoding_before_split", line, "group_encode", enc,
                               split_line,
                               f"a full-data group encoding (`{gb_base}.groupby(...).transform(...)` "
                               f"or a groupby-derived `.map(...)`) at line {line} broadcasts a "
                               f"per-category aggregate onto `{enc}`, which is then split via "
                               f"`{split_fn}` at line {split_line}: the aggregate (e.g. a per-category "
                               f"target mean) was computed over the held-out rows, leaking their "
                               f"distribution into the training features.",
                               [train_id, holdout_id], [holdout_id])

    def _scan_concat_normalization(self, nodes: list[ast.AST]) -> None:
        # Pattern I -- normalisation over an explicit train/test *merge*.
        # `merged = torch.cat([train_x, test_x]); m = merged.mean(); s = merged.std();
        #  train_x = (train_x - m) / s` computes the normalisation constants over a
        # tensor that already contains the held-out rows. There is no
        # `train_test_split` call here -- the partition is implicit in the operands
        # of the `cat` -- so patterns A-H never see it. The defining, low-false-
        # positive signal is a merge (`torch.cat` / `np.concatenate` / `vstack` /
        # `stack`) of >= 2 operands whose names carry distinct split roles
        # (train/val/test/...). Model-internal `torch.cat([out1, out2])` over
        # feature maps carries no split role and is ignored.
        merges: list[tuple[str, int, set[str]]] = []  # (merged_var, line, roles)
        for node in nodes:
            if not (isinstance(node, ast.Assign) and isinstance(node.value, ast.Call)):
                continue
            call = node.value
            if _call_func_name(call) not in _MERGE_NAMES:
                continue
            operands: list[ast.AST] = []
            for a in call.args:
                if isinstance(a, (ast.List, ast.Tuple)):
                    operands.extend(a.elts)
                else:
                    operands.append(a)
            roles: dict[str, str] = {}
            for op in operands:
                nm = _name_of(op)
                role = _split_role(nm)
                if nm and role:
                    roles[nm] = role
            if len({r for r in roles.values()}) < 2:
                continue  # need >= 2 *distinct* partition roles to be a real merge
            merged_var = None
            for tgt in node.targets:
                if (nm := _name_of(tgt)):
                    merged_var = nm
            if merged_var:
                merges.append((merged_var, getattr(node, "lineno", 0), set(roles)))
        if not merges:
            return
        # Variables that participate in a Sub/Div (normalisation arithmetic).
        norm_op_vars: set[str] = set()
        for node in nodes:
            if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Sub, ast.Div)):
                norm_op_vars |= _names_within(node)
        # Variables passed to a `*normal*` transform (lazy torch path).
        norm_arg_vars: set[str] = set()
        for node in nodes:
            if isinstance(node, ast.Call) and (fn := _call_func_name(node)) and "normal" in fn.lower():
                for a in node.args:
                    norm_arg_vars |= _names_within(a)
                for kw in node.keywords:
                    norm_arg_vars |= _names_within(kw.value)
        deriv = _derivation_map(nodes)
        for merged_var, line, roles in merges:
            # Collect the variables that hold a statistic computed over the merge
            # (`m = merged.mean()` / `merged.std(0)` / `np.mean(merged)`).
            stat_targets: set[str] = set()
            direct_norm = False
            for node in nodes:
                if not isinstance(node, ast.Assign):
                    continue
                base_hit = False
                for sub in ast.walk(node.value):
                    if isinstance(sub, ast.Call) and isinstance(sub.func, ast.Attribute) \
                            and sub.func.attr in _NORM_STAT_ATTRS:
                        recv_root = _root_name(sub.func.value)
                        if recv_root == merged_var:
                            base_hit = True
                        elif recv_root in _NUMERIC_MODULES and sub.args \
                                and _root_name(sub.args[0]) == merged_var:
                            base_hit = True
                if base_hit:
                    for tgt in node.targets:
                        if (nm := _name_of(tgt)):
                            stat_targets.add(nm)
                    # `merged = (merged - merged.mean()) / merged.std()` -- normalise in place.
                    if any(isinstance(s, ast.BinOp) and isinstance(s.op, (ast.Sub, ast.Div))
                           for s in ast.walk(node.value)):
                        direct_norm = True
            # The statistic must actually be USED to normalise: either the merge is
            # normalised in place, or a stat-of-merge feeds a Sub/Div or a Normalize.
            used = direct_norm or bool(stat_targets & (norm_op_vars | norm_arg_vars))
            if not stat_targets or not used:
                continue
            train_id, holdout_id = _partition_ids(f"merge@{line}")
            roles_txt = "/".join(sorted(roles))
            self._emit("I:concat_partition_normalization", line, "merge_normalize",
                       merged_var, line,
                       f"`{merged_var}` merges the {roles_txt} partitions (a torch.cat / "
                       f"concatenate of already-split tensors) at line {line}, and its "
                       f"mean/std are then used to normalise the data: the normalisation "
                       f"constants were computed over the held-out partition, leaking its "
                       f"distribution into training.",
                       [train_id, holdout_id], [holdout_id])

    def _emit(self, pattern: str, line: int, est: str, data_var: str,
              split_line: int, explanation: str,
              fit_row_ids: list[str], holdout_row_ids: list[str]) -> str:
        # The verdict is the certifier's, computed from the observed provenance.
        # We only record a finding when the certifier *rejects* the obligation;
        # a clean (split-then-fit) pipeline is certified and produces nothing.
        verdict = _certify(est, fit_row_ids, holdout_row_ids)
        if verdict != "rejected":
            return verdict
        snippet = self.src_lines[line - 1].strip() if 0 < line <= len(self.src_lines) else ""
        self.findings.append(LeakageFinding(
            file=self.file, line=line, pattern=pattern, estimator=est,
            data_var=data_var, split_line=split_line, snippet=snippet,
            constraint="fit_transform_isolation", verdict=verdict, explanation=explanation,
        ))
        return verdict


def scan_source(source: str, filename: str = "<string>") -> list[LeakageFinding]:
    """Scan one Python source string; return certified leakage findings."""
    try:
        tree = ast.parse(source, filename=filename)
    except SyntaxError:
        return []
    src_lines = source.splitlines()
    known = _collect_transformer_vars(tree)
    scopes: list[list[ast.stmt]] = [tree.body]
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            scopes.append(node.body)
    findings: list[LeakageFinding] = []
    for body in scopes:
        v = _ScopeVisitor(filename, src_lines, known)
        v.analyse(body)
        findings.extend(v.findings)
    uniq: dict[tuple[str, int, int], LeakageFinding] = {}
    for f in findings:
        uniq[(f.file, f.line, f.split_line)] = f
    return sorted(uniq.values(), key=lambda f: (f.file, f.line))


def scan_path(path: str | Path) -> list[LeakageFinding]:
    """Scan a file or recursively scan a directory of ``*.py`` files."""
    p = Path(path)
    files: Iterable[Path] = [p] if p.is_file() else sorted(p.rglob("*.py"))
    out: list[LeakageFinding] = []
    for fp in files:
        try:
            out.extend(scan_source(fp.read_text(encoding="utf-8", errors="ignore"), str(fp)))
        except (OSError, ValueError):
            continue
    return out


@dataclass(frozen=True)
class PipelineLeakageScanner:
    """Object wrapper around :func:`scan_source` / :func:`scan_path`."""

    name: str = "pipeline_leakage"

    def scan_source(self, source: str, filename: str = "<string>") -> list[LeakageFinding]:
        return scan_source(source, filename)

    def scan_path(self, path: str | Path) -> list[LeakageFinding]:
        return scan_path(path)
