"""Static data-plane misuse scanner for PyTorch source (North Star, slice 1).

PromptABI's existing analyzers prove structural contracts at the *text* interface
of an LLM system (tokenizer / template / stop / tool-call boundary). This module
opens the **data plane**: it scans the source of PyTorch ``Dataset`` /
``DataLoader`` / training code for *silent data-misuse bugs* -- code that runs
without error but feeds the model data that violates an unstated contract.

See ``NORTH_STAR.md`` for the full thesis and taxonomy. This first slice decides
three of the most famous, most clearly "wrong way to use data" classes, all of
which throw no exception and are therefore routinely shipped:

* **worker-rng-duplication** (family C, sampling/determinism). A ``Dataset``
  whose ``__getitem__`` draws augmentation randomness from ``numpy.random`` or the
  stdlib ``random`` module, consumed by a ``DataLoader(num_workers>0)`` with no
  ``worker_init_fn``. Every worker process inherits the *same* parent RNG state,
  so the "random" augmentations are identical across workers -- a well-documented
  PyTorch footgun (``torch`` RNG is re-seeded per worker; ``numpy``/``random`` are
  not).
* **drop-last-on-eval** (family C). ``DataLoader(..., drop_last=True)`` for a
  validation/test/eval loader silently discards the final partial batch, so every
  metric is computed on a truncated, batch-size-dependent subset.
* **fit-before-split-leakage** (family B, contamination). A ``fit`` /
  ``fit_transform`` / dataset-wide statistic computed over the *full* data
  *before* a ``train_test_split`` / ``random_split`` leaks test-set statistics
  into the training normalization.

Honesty contract (mirrors ``parser_source_scan`` / ``upstream_bug_campaign``):
every finding is a structurally-present pattern with an explicit guarantee tier
and a replayable witness slice. The scanner abstains (emits nothing) whenever the
relevant structure is dynamic or unresolved -- no guessing. It only inspects the
source you point it at and makes no network calls.
"""

from __future__ import annotations

import ast
import json
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Iterable, Sequence

TORCH_DATA_MISUSE_VERSION = 1


class TorchDataMisuseKind(StrEnum):
    WORKER_RNG_DUPLICATION = "worker-rng-duplication"
    DROP_LAST_ON_EVAL = "drop-last-on-eval"
    FIT_BEFORE_SPLIT_LEAKAGE = "fit-before-split-leakage"


class GuaranteeTier(StrEnum):
    """How strong the claim is (mirrors the rest of PromptABI)."""

    SOUND = "sound"
    BOUNDED = "bounded"
    HEURISTIC = "heuristic"


@dataclass(frozen=True)
class TorchDataFinding:
    kind: TorchDataMisuseKind
    guarantee: GuaranteeTier
    message: str
    line: int
    col: int
    witness: tuple[str, ...]
    suggestion: str
    path: str | None = None

    def as_dict(self) -> dict[str, object]:
        return {
            "kind": str(self.kind),
            "guarantee": str(self.guarantee),
            "message": self.message,
            "path": self.path,
            "line": self.line,
            "col": self.col,
            "witness": list(self.witness),
            "suggestion": self.suggestion,
        }


@dataclass(frozen=True)
class TorchDataMisuseReport:
    path: str | None
    findings: tuple[TorchDataFinding, ...]
    abstained: bool = False
    note: str = ""

    @property
    def ok(self) -> bool:
        return not self.findings

    def as_dict(self) -> dict[str, object]:
        return {
            "analysis": "torch-data-misuse",
            "version": TORCH_DATA_MISUSE_VERSION,
            "path": self.path,
            "ok": self.ok,
            "abstained": self.abstained,
            "note": self.note,
            "findings": [f.as_dict() for f in self.findings],
        }


# --------------------------------------------------------------------------- #
# AST helpers.
# --------------------------------------------------------------------------- #


def _call_name(node: ast.Call) -> str:
    """Return the dotted callee name of a Call (best-effort, last two parts)."""

    func = node.func
    parts: list[str] = []
    while isinstance(func, ast.Attribute):
        parts.append(func.attr)
        func = func.value
    if isinstance(func, ast.Name):
        parts.append(func.id)
    parts.reverse()
    return ".".join(parts)


def _attr_root(node: ast.AST) -> str | None:
    """Return the leftmost Name id of an attribute chain (e.g. ``np`` in np.random.rand)."""

    cur = node
    while isinstance(cur, ast.Attribute):
        cur = cur.value
    if isinstance(cur, ast.Name):
        return cur.id
    return None


def _kw(node: ast.Call, name: str) -> ast.expr | None:
    for kw in node.keywords:
        if kw.arg == name:
            return kw.value
    return None


def _literal_int(node: ast.expr | None) -> int | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, int) and not isinstance(node.value, bool):
        return node.value
    return None


def _is_true(node: ast.expr | None) -> bool:
    return isinstance(node, ast.Constant) and node.value is True


def _base_name(base: ast.expr) -> str:
    if isinstance(base, ast.Name):
        return base.id
    if isinstance(base, ast.Attribute):
        return base.attr
    return ""


_EVAL_HINT_TOKENS = ("val", "valid", "eval", "test", "dev", "holdout")
# RNG sources that are NOT re-seeded per DataLoader worker (torch IS, so excluded).
_UNSAFE_RNG_ROOTS = {"np", "numpy", "npr", "random"}


def _name_hints_eval(name: str | None) -> bool:
    if not name:
        return False
    low = name.lower()
    return any(tok in low for tok in _EVAL_HINT_TOKENS)


# --------------------------------------------------------------------------- #
# Detectors.
# --------------------------------------------------------------------------- #


class _Scanner(ast.NodeVisitor):
    def __init__(self, source_lines: Sequence[str], path: str | None) -> None:
        self._lines = source_lines
        self._path = path
        self.findings: list[TorchDataFinding] = []
        # name of every class that looks like a Dataset whose __getitem__ uses unsafe RNG
        self.unsafe_rng_datasets: dict[str, tuple[int, str]] = {}
        # all DataLoader call nodes (collected, evaluated after class scan)
        self._dataloader_calls: list[ast.Call] = []
        # statement-order effect log for leakage: list of (order, kind, var, node)
        self._effects: list[tuple[int, str, str, ast.AST]] = []
        # variable -> constructed class name (e.g. ds -> AugDataset)
        self._var_class: dict[str, str] = {}
        self._order = 0

    def _src(self, node: ast.AST) -> str:
        line = getattr(node, "lineno", 0)
        if 1 <= line <= len(self._lines):
            return self._lines[line - 1].strip()
        return ""

    # -- Dataset class scan -------------------------------------------------- #

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        base_names = {_base_name(b) for b in node.bases}
        looks_dataset = any(b.endswith("Dataset") or b == "Dataset" for b in base_names)
        if looks_dataset or any(_base_name(b) == "IterableDataset" for b in node.bases):
            self._scan_dataset_class(node)
        self.generic_visit(node)

    def _scan_dataset_class(self, node: ast.ClassDef) -> None:
        getitem = next(
            (
                m
                for m in node.body
                if isinstance(m, ast.FunctionDef) and m.name in {"__getitem__", "__next__", "__iter__"}
            ),
            None,
        )
        if getitem is None:
            return
        seeds_locally = False
        unsafe_call: tuple[int, str] | None = None
        for sub in ast.walk(getitem):
            if isinstance(sub, ast.Call):
                root = _attr_root(sub.func)
                name = _call_name(sub)
                if root in _UNSAFE_RNG_ROOTS and (".random" in name or root == "random"):
                    # numpy.random.* / npr.* / random.*  -> unsafe draw
                    if unsafe_call is None:
                        unsafe_call = (getattr(sub, "lineno", node.lineno), self._src(sub))
                if name.endswith("seed") and root in _UNSAFE_RNG_ROOTS:
                    seeds_locally = True
        if unsafe_call is not None and not seeds_locally:
            self.unsafe_rng_datasets[node.name] = unsafe_call

    # -- collect DataLoader calls + leakage effects -------------------------- #

    def visit_Assign(self, node: ast.Assign) -> None:
        # Track `var = ClassName(...)` so a DataLoader's dataset variable resolves.
        if (
            isinstance(node.value, ast.Call)
            and node.targets
            and isinstance(node.targets[0], ast.Name)
        ):
            cls = _base_name(node.value.func)
            if cls:
                self._var_class[node.targets[0].id] = cls
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        name = _call_name(node)
        short = name.rsplit(".", 1)[-1]
        if short == "DataLoader":
            self._dataloader_calls.append(node)
        if short in {"fit", "fit_transform"}:
            target = _attr_root(node.func)  # e.g. scaler in scaler.fit(X)
            arg_var = _first_arg_name(node)
            if arg_var is not None:
                self._effects.append((self._order, "fit", arg_var, node))
        if short in {"train_test_split", "random_split"}:
            arg_var = _first_arg_name(node)
            if arg_var is not None:
                self._effects.append((self._order, "split", arg_var, node))
        self._order += 1
        self.generic_visit(node)

    # -- DataLoader-dependent checks ----------------------------------------- #

    def finalize(self) -> None:
        self._check_worker_rng()
        self._check_drop_last_eval()
        self._check_fit_before_split()

    def _check_worker_rng(self) -> None:
        if not self.unsafe_rng_datasets:
            return
        for call in self._dataloader_calls:
            workers = _literal_int(_kw(call, "num_workers"))
            if workers is None or workers < 1:
                continue
            if _kw(call, "worker_init_fn") is not None:
                continue
            raw_ds = _dataset_arg_name(call)
            resolved_cls: str | None = None
            if raw_ds is not None:
                if raw_ds in self.unsafe_rng_datasets:
                    resolved_cls = raw_ds  # DataLoader(MyUnsafeDataset(...))
                elif raw_ds in self._var_class:
                    resolved_cls = self._var_class[raw_ds]  # ds -> AugDataset
            # If we resolved the dataset to a *known, non-unsafe* class, this loader
            # is fine -> abstain. Otherwise (unresolved var, or resolved to an unsafe
            # class) we flag, since an unsafe-RNG dataset exists in this module.
            if resolved_cls is not None and resolved_cls not in self.unsafe_rng_datasets:
                continue
            cls = resolved_cls or next(iter(self.unsafe_rng_datasets))
            rng_line, rng_src = self.unsafe_rng_datasets[cls]
            self.findings.append(
                TorchDataFinding(
                    kind=TorchDataMisuseKind.WORKER_RNG_DUPLICATION,
                    guarantee=GuaranteeTier.HEURISTIC,
                    message=(
                        f"DataLoader(num_workers={workers}) consumes Dataset {cls!r} whose "
                        f"__getitem__ draws randomness from numpy/random with no worker_init_fn; "
                        f"every worker inherits the same RNG state -> identical augmentations"
                    ),
                    line=getattr(call, "lineno", 0),
                    col=getattr(call, "col_offset", 0),
                    path=self._path,
                    witness=(
                        f"Dataset {cls} __getitem__ at line {rng_line}: {rng_src}",
                        f"DataLoader(num_workers={workers}) at line {getattr(call, 'lineno', 0)}: {self._src(call)}",
                        "no worker_init_fn -> all workers share the parent numpy/random seed",
                        "consequence: the same 'random' augmentation is produced in every worker",
                    ),
                    suggestion=(
                        "Pass worker_init_fn=lambda wid: np.random.seed(torch.initial_seed() % 2**32) "
                        "(or seed per-worker), or use torch's RNG (torch.randint/torch.randn) which is "
                        "re-seeded per worker automatically."
                    ),
                )
            )

    def _check_drop_last_eval(self) -> None:
        for call in self._dataloader_calls:
            if not _is_true(_kw(call, "drop_last")):
                continue
            target = getattr(call, "_assign_target", None)
            ds_name = _dataset_arg_name(call)
            if not (_name_hints_eval(target) or _name_hints_eval(ds_name)):
                continue
            label = target or ds_name or "this loader"
            self.findings.append(
                TorchDataFinding(
                    kind=TorchDataMisuseKind.DROP_LAST_ON_EVAL,
                    guarantee=GuaranteeTier.HEURISTIC,
                    message=(
                        f"drop_last=True on an evaluation/validation loader ({label!r}) silently "
                        f"discards the final partial batch -> metrics over a truncated subset"
                    ),
                    line=getattr(call, "lineno", 0),
                    col=getattr(call, "col_offset", 0),
                    path=self._path,
                    witness=(
                        f"DataLoader at line {getattr(call, 'lineno', 0)}: {self._src(call)}",
                        f"target/dataset name {label!r} matches an eval/val/test hint",
                        "drop_last=True drops the last partial batch from iteration",
                        "consequence: reported metrics depend on batch_size and omit some samples",
                    ),
                    suggestion="Set drop_last=False for evaluation/validation/test loaders.",
                )
            )

    def _check_fit_before_split(self) -> None:
        # For each variable, if a fit/stat over it precedes a split of it -> leakage.
        fits: dict[str, tuple[int, ast.AST]] = {}
        for order, kind, var, node in self._effects:
            if kind == "fit" and var not in fits:
                fits[var] = (order, node)
        for order, kind, var, node in self._effects:
            if kind != "split":
                continue
            if var in fits and fits[var][0] < order:
                fit_order, fit_node = fits[var]
                self.findings.append(
                    TorchDataFinding(
                        kind=TorchDataMisuseKind.FIT_BEFORE_SPLIT_LEAKAGE,
                        guarantee=GuaranteeTier.HEURISTIC,
                        message=(
                            f"statistics were fit on the full data {var!r} (line "
                            f"{getattr(fit_node, 'lineno', 0)}) before it was split (line "
                            f"{getattr(node, 'lineno', 0)}) -> test statistics leak into training"
                        ),
                        line=getattr(node, "lineno", 0),
                        col=getattr(node, "col_offset", 0),
                        path=self._path,
                        witness=(
                            f"fit over full data at line {getattr(fit_node, 'lineno', 0)}: {self._src(fit_node)}",
                            f"split of the same variable at line {getattr(node, 'lineno', 0)}: {self._src(node)}",
                            "the fit saw test rows -> normalization/encoding carries test information",
                        ),
                        suggestion=(
                            "Split first, then fit the transform on the training split only and apply "
                            "it to validation/test (e.g. scaler.fit(X_train); scaler.transform(X_test))."
                        ),
                    )
                )


def _first_arg_name(node: ast.Call) -> str | None:
    if node.args:
        a = node.args[0]
        if isinstance(a, ast.Name):
            return a.id
    return None


def _dataset_arg_name(call: ast.Call) -> str | None:
    """Resolve the dataset argument variable/class name of a DataLoader call."""

    arg: ast.expr | None = None
    if call.args:
        arg = call.args[0]
    else:
        arg = _kw(call, "dataset")
    if isinstance(arg, ast.Name):
        return arg.id
    if isinstance(arg, ast.Call):
        # DataLoader(MyDataset(...)) -> the class name
        return _base_name(arg.func)
    return None


def _annotate_assign_targets(tree: ast.AST) -> None:
    """Tag DataLoader call nodes with the name they are assigned to (for eval hints)."""

    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Call):
            name = _call_name(node.value).rsplit(".", 1)[-1]
            if name == "DataLoader" and node.targets and isinstance(node.targets[0], ast.Name):
                setattr(node.value, "_assign_target", node.targets[0].id)


# --------------------------------------------------------------------------- #
# Public API.
# --------------------------------------------------------------------------- #


def analyze_torch_data_source(source: str, *, path: str | None = None) -> TorchDataMisuseReport:
    """Scan PyTorch data-plane *source* for silent data-misuse bugs."""

    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        return TorchDataMisuseReport(
            path=path, findings=(), abstained=True, note=f"unparseable source: {exc}"
        )
    _annotate_assign_targets(tree)
    scanner = _Scanner(source.splitlines(), path)
    scanner.visit(tree)
    scanner.finalize()
    findings = tuple(
        sorted(scanner.findings, key=lambda f: (f.line, f.col, str(f.kind)))
    )
    return TorchDataMisuseReport(path=path, findings=findings)


def analyze_torch_data_file(path: str | Path) -> TorchDataMisuseReport:
    p = Path(path)
    return analyze_torch_data_source(p.read_text(encoding="utf-8"), path=str(p))


def analyze_torch_data_tree(root: str | Path) -> tuple[TorchDataMisuseReport, ...]:
    """Scan every ``*.py`` under ``root`` (file or directory)."""

    base = Path(root)
    files: Iterable[Path]
    if base.is_file():
        files = [base]
    else:
        files = sorted(base.rglob("*.py"))
    return tuple(analyze_torch_data_file(f) for f in files)


def render_torch_data_report_json(reports: TorchDataMisuseReport | Sequence[TorchDataMisuseReport]) -> str:
    if isinstance(reports, TorchDataMisuseReport):
        payload: object = reports.as_dict()
    else:
        payload = [r.as_dict() for r in reports]
    return json.dumps(payload, indent=2, ensure_ascii=False)


def render_torch_data_report_text(reports: TorchDataMisuseReport | Sequence[TorchDataMisuseReport]) -> str:
    if isinstance(reports, TorchDataMisuseReport):
        reports = [reports]
    lines: list[str] = []
    total = 0
    for r in reports:
        if r.abstained:
            lines.append(f"{r.path or '<source>'}: abstained ({r.note})")
            continue
        if r.ok:
            continue
        lines.append(f"{r.path or '<source>'}:")
        for f in r.findings:
            total += 1
            lines.append(f"  [{f.kind}] ({f.guarantee}) line {f.line}: {f.message}")
            for step in f.witness:
                lines.append(f"      - {step}")
            lines.append(f"    fix: {f.suggestion}")
    if not lines:
        return "torch-data-misuse: no findings"
    lines.append(f"\n{total} finding(s) across {len(reports)} file(s).")
    return "\n".join(lines)
