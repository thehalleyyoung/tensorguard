"""Static scanner for the Dataset / DataLoader **stochastic-effect** contract.

Where the other scanners certify *value* properties of the data --- set
disjointness (`pipeline_leakage`), interval disjointness (`split_contracts`),
temporal causality (`temporal_leakage`), quotient disjointness (`group_leakage`)
--- this scanner lifts the lattice one level further, to an **effect** property of
the *sampling path itself*: the randomness an ordinary ``Dataset.__getitem__`` /
``DataLoader`` draws must be **independent across samples** and the evaluation
path must be **deterministic**.  Two runtime-silent bug classes fall out of one
contract:

* **D1 --- correlated worker RNG (the NumPy-fork bug).**  A map-style ``Dataset``
  whose ``__getitem__`` draws from a *process-global* RNG (``np.random.*`` /
  ``random.*``) loaded by a ``DataLoader(num_workers >= 1)`` **without a
  ``worker_init_fn``** is broken: ``fork`` copies the parent's RNG state into
  every worker, so all workers emit the *same* sequence of "random" augmentations
  every epoch.  Training proceeds, no exception is raised, and the effective
  augmentation diversity silently collapses.  This exact footgun has been
  documented across *thousands* of open-source PyTorch projects (Pärnamaa 2021);
  ``torch``'s own RNG is reseeded per worker and is therefore *not* flagged.

* **D2 --- nondeterministic evaluation.**  A validation / test transform pipeline
  that contains a *random* augmentation (``RandomResizedCrop``, ``ColorJitter``,
  ``RandAugment`` ...) makes the held-out metric nondeterministic and biased ---
  test-time augmentation accidentally left on.  Recognised either from an
  eval-role pipeline variable (``test_transform = Compose([... Random* ...])``) or
  from a random aug passed as the ``transform=`` of a dataset built with
  ``train=False`` / ``split in {test, val}``.

As with every other scanner we **do not model pandas / numpy / torch / Python
semantics**.  Both bugs are recognised from *local syntactic signals*: an
explicit module-global RNG draw inside a ``__getitem__``-bearing class, a
``DataLoader`` call's literal ``num_workers`` and the presence/absence of
``worker_init_fn``, and a small catalog of stochastic torchvision augmentations.
Each candidate is lowered to a ``sampling_independence`` obligation that the
z3-backed :class:`~datarefine.certification.StructuralCertifier` decides (the
worker count is a genuine integer the solver reasons about), with a concrete
witness and an independent re-check.
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
    "SamplingFinding",
    "DataLoaderDeterminismScanner",
    "scan_source",
    "scan_path",
    "scan_tree",
]

_CERT = StructuralCertifier()

# Module-global RNG modules whose draws share one process state across a fork.
# ``torch`` is excluded: the DataLoader reseeds each worker's torch RNG.
_RNG_MODULES = frozenset({"random", "np", "numpy"})
# Attributes that are *not* per-call draws (seeders / generator constructors /
# state accessors), so they do not signal a forked-RNG draw.
_RNG_NON_DRAW = frozenset({
    "seed", "default_rng", "Generator", "RandomState", "SeedSequence",
    "get_state", "set_state", "manual_seed",
})

# Stochastic torchvision / aug transforms whose presence on an eval path makes
# the metric nondeterministic.
_RANDOM_AUG = frozenset({
    "RandomCrop", "RandomResizedCrop", "RandomHorizontalFlip",
    "RandomVerticalFlip", "RandomRotation", "RandomAffine", "RandomPerspective",
    "RandomErasing", "ColorJitter", "RandomApply", "RandomChoice", "RandomOrder",
    "RandomGrayscale", "RandomInvert", "RandomPosterize", "RandomSolarize",
    "RandomAdjustSharpness", "RandomAutocontrast", "RandomEqualize",
    "GaussianBlur", "RandAugment", "AutoAugment", "TrivialAugmentWide", "AugMix",
    "ElasticTransform", "RandomResize",
})

_EVAL_ROLE_RE = re.compile(
    r"(?:^|_)(?:val|valid|validation|test|eval|evaluation|inference|infer|"
    r"predict)(?:$|_)",
    re.IGNORECASE,
)


@dataclass
class SamplingFinding:
    file: str
    line: int
    pattern: str
    constraint: str
    verdict: str
    detail: str
    snippet: str
    num_workers: int | None = None
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
    if isinstance(node, ast.Constant) and isinstance(node.value, bool) is False \
            and isinstance(node.value, int):
        return node.value
    return None


def _kw(call: ast.Call, name: str) -> ast.AST | None:
    for kw in call.keywords:
        if kw.arg == name:
            return kw.value
    return None


def _is_false(node: ast.AST | None) -> bool:
    return isinstance(node, ast.Constant) and node.value is False


def _global_rng_draw(call: ast.Call) -> str | None:
    """Return a dotted label if ``call`` is a *process-global* RNG draw.

    Recognises ``random.<draw>(...)`` and ``np.random.<draw>(...)`` /
    ``numpy.random.<draw>(...)``; returns ``None`` for seeders, generator
    constructors, ``torch`` draws, and draws on a local ``Generator`` instance.
    """
    f = call.func
    if not isinstance(f, ast.Attribute):
        return None
    attr = f.attr
    if attr in _RNG_NON_DRAW:
        return None
    base = f.value
    # random.<draw>
    if isinstance(base, ast.Name) and base.id == "random":
        return f"random.{attr}"
    # np.random.<draw> / numpy.random.<draw>
    if isinstance(base, ast.Attribute) and base.attr == "random" \
            and isinstance(base.value, ast.Name) and base.value.id in ("np", "numpy"):
        return f"{base.value.id}.random.{attr}"
    return None


def _contains_random_aug(value: ast.AST) -> str | None:
    """Name of the first stochastic augmentation called inside ``value``."""
    for n in ast.walk(value):
        if isinstance(n, ast.Call):
            name = _name_of(n.func)
            if name in _RANDOM_AUG:
                return name
    return None


class DataLoaderDeterminismScanner:
    def __init__(self, source: str, filename: str = "<string>") -> None:
        self.source = source
        self.file = filename
        self.src_lines = source.splitlines()
        self.findings: list[SamplingFinding] = []

    def _certify(self, **payload) -> tuple[str, dict | None]:
        ob = obligation("sampling", self.file, "sampling_independence",
                        constraint="sampling_independence", **payload)
        verdict = _CERT.certify(ob)
        witness = None
        diags = verdict.diagnostics or ()
        if diags and isinstance(diags[0], dict):
            witness = diags[0].get("model")
        return verdict.status, witness

    def _snippet(self, line: int) -> str:
        return self.src_lines[line - 1].strip() if 0 < line <= len(self.src_lines) else ""

    # -- D1: correlated worker RNG -----------------------------------------
    def _dataset_global_rng(self, tree: ast.AST) -> tuple[str, str] | None:
        """First ``(class_name, rng_label)`` for a map-style dataset whose body
        draws from a process-global RNG."""
        for node in ast.walk(tree):
            if not isinstance(node, ast.ClassDef):
                continue
            has_getitem = any(isinstance(m, ast.FunctionDef) and m.name == "__getitem__"
                              for m in node.body)
            if not has_getitem:
                continue
            for inner in ast.walk(node):
                if isinstance(inner, ast.Call):
                    label = _global_rng_draw(inner)
                    if label is not None:
                        return node.name, label
        return None

    def _scan_dataloaders(self, tree: ast.AST, rng: tuple[str, str]) -> None:
        cls, label = rng
        for node in ast.walk(tree):
            if not (isinstance(node, ast.Call) and _name_of(node.func) == "DataLoader"):
                continue
            workers = _lit_int(_kw(node, "num_workers"))
            if workers is None or workers < 1:
                continue  # default 0 -> no fork; unknown -> precision-first skip
            has_init = _kw(node, "worker_init_fn") is not None
            line = getattr(node, "lineno", 0)
            status, witness = self._certify(
                global_rng=True, num_workers=workers,
                worker_init_fn=has_init, is_eval=False)
            if status != "rejected":
                continue
            detail = (
                f"Dataset {cls!r} draws from the process-global RNG via {label}(); "
                f"DataLoader(num_workers={workers}) forks workers that inherit one "
                f"RNG state and no worker_init_fn reseeds them, so every worker "
                f"emits identical 'random' augmentations each epoch. Add "
                f"worker_init_fn to reseed numpy/random per worker, or draw from "
                f"torch.randint / a per-worker Generator."
            )
            self.findings.append(SamplingFinding(
                file=self.file, line=line,
                pattern="D1:correlated_worker_rng",
                constraint="sampling_independence", verdict=status,
                detail=detail, snippet=self._snippet(line),
                num_workers=workers, witness=witness))

    # -- D2: nondeterministic eval -----------------------------------------
    def _scan_eval_augmentations(self, tree: ast.AST) -> None:
        for node in ast.walk(tree):
            # test_transform = Compose([... Random* ...])
            if isinstance(node, (ast.Assign, ast.AnnAssign)):
                targets = (node.targets if isinstance(node, ast.Assign)
                           else [node.target])
                names = [n for n in (_name_of(t) for t in targets) if n]
                if not any(_EVAL_ROLE_RE.search(n) for n in names):
                    continue
                if node.value is None:
                    continue
                aug = _contains_random_aug(node.value)
                if aug is not None:
                    self._emit_eval(getattr(node, "lineno", 0), aug,
                                    "eval-role transform pipeline")
            # dataset(train=False, transform=Compose([... Random* ...]))
            elif isinstance(node, ast.Call):
                is_eval_ds = _is_false(_kw(node, "train"))
                split = _kw(node, "split")
                if isinstance(split, ast.Constant) and isinstance(split.value, str) \
                        and _EVAL_ROLE_RE.search(split.value):
                    is_eval_ds = True
                if not is_eval_ds:
                    continue
                tfm = _kw(node, "transform") or _kw(node, "transforms")
                if tfm is None:
                    continue
                aug = _contains_random_aug(tfm)
                if aug is not None:
                    self._emit_eval(getattr(node, "lineno", 0), aug,
                                    "eval dataset transform")

    def _emit_eval(self, line: int, aug: str, where: str) -> None:
        status, witness = self._certify(is_eval=True, stochastic_eval=True)
        if status != "rejected":
            return
        detail = (
            f"{where} applies the stochastic augmentation {aug}() on an "
            f"evaluation/test path; the held-out metric becomes nondeterministic "
            f"and biased (test-time augmentation left on). Use deterministic "
            f"eval transforms (Resize/CenterCrop/Normalize) only."
        )
        self.findings.append(SamplingFinding(
            file=self.file, line=line, pattern="D2:stochastic_eval_transform",
            constraint="sampling_independence", verdict=status,
            detail=detail, snippet=self._snippet(line), witness=witness))

    def scan(self) -> list[SamplingFinding]:
        try:
            tree = ast.parse(self.source, filename=self.file)
        except SyntaxError:
            return []
        rng = self._dataset_global_rng(tree)
        if rng is not None:
            self._scan_dataloaders(tree, rng)
        self._scan_eval_augmentations(tree)
        return self.findings


def scan_source(source: str, filename: str = "<string>") -> list[SamplingFinding]:
    return DataLoaderDeterminismScanner(source, filename).scan()


def scan_path(path: str | Path) -> list[SamplingFinding]:
    p = Path(path)
    try:
        source = p.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return []
    return scan_source(source, filename=str(p))


def scan_tree(root: str | Path) -> list[SamplingFinding]:
    root = Path(root)
    out: list[SamplingFinding] = []
    paths: Iterable[Path] = [root] if root.is_file() else root.rglob("*.py")
    for p in paths:
        out.extend(scan_path(p))
    return out
