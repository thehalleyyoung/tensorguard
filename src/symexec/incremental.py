"""Incremental analysis: re-analyze only what changed (roadmap Step 84).

Re-running the engine over a whole project after editing one file repeats all the
work for the unchanged files.  This module caches per-file results and, on a
re-run, recomputes a file only when its analysis could actually differ — i.e.
when either

* the file's own source changed, **or**
* a project symbol the file directly imports (and whose body is therefore inlined
  into the file's import-augmented analysis, see :mod:`.package`) changed.

The atomic re-analysis unit is the *file*.  This is deliberate and is what keeps
incremental output **byte-identical** to a fresh whole-package run: a file's
:class:`~src.symexec.engine.SymResult` — including its proof fingerprint, which
folds in the per-run abstain-coverage profile accumulated across all of the
file's analysis passes in one shared interpreter — is a deterministic function of
exactly ``(file source, directly-imported project definitions)``.  A finer
*function*-level merge cannot reproduce that fingerprint, because abstain counts
depend on cross-pass interpreter-cache interactions; so we re-analyse the whole
file but invalidate at *symbol-dependency* granularity, and additionally expose
function-level change *detection* (:func:`unit_index` / :func:`diff_units`) for
reporting which functions an edit touched.

Soundness/correctness: a reused result is the *exact cached object* from a prior
analysis of identical inputs, so reuse can never change a verdict — it only skips
recomputation.  Anything whose determinant changed is recomputed from scratch via
the same per-module path the whole-package driver uses.  The module is torch-free.
"""

from __future__ import annotations

import ast
import hashlib
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from .engine import SymResult, analyze_source
from .package import PackageIndex, PackageResult, _analyze_one

__all__ = [
    "IncrementalCache",
    "IncrementalStats",
    "UnitChange",
    "analyze_package_incremental",
    "analyze_source_incremental",
    "unit_index",
    "diff_units",
]


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# --------------------------------------------------------------------------- #
# Function-level change detection                                             #
# --------------------------------------------------------------------------- #

def unit_index(module: ast.Module) -> Dict[str, str]:
    """Map each top-level unit of ``module`` to a position-sensitive hash.

    Each top-level ``class``/``def`` is keyed by its name; everything else at
    module level (imports, assignments, the ``__main__`` block, …) is folded into
    a single ``"<module>"`` entry.  Hashes include source positions
    (``include_attributes=True``) so a line shift — which would move reported bug
    lines — counts as a change."""
    index: Dict[str, str] = {}
    other: List[str] = []
    for node in module.body:
        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            index[node.name] = _sha(ast.dump(node, include_attributes=True))
        else:
            other.append(ast.dump(node, include_attributes=True))
    index["<module>"] = _sha("\x00".join(other))
    return index


@dataclass(frozen=True)
class UnitChange:
    """The set of top-level units that differ between two versions of a file."""

    added: Tuple[str, ...] = ()
    removed: Tuple[str, ...] = ()
    modified: Tuple[str, ...] = ()

    @property
    def any(self) -> bool:
        return bool(self.added or self.removed or self.modified)

    @property
    def affected(self) -> Tuple[str, ...]:
        return tuple(sorted(set(self.added) | set(self.removed) | set(self.modified)))


def diff_units(old: Dict[str, str], new: Dict[str, str]) -> UnitChange:
    """Diff two :func:`unit_index` maps into added / removed / modified units."""
    old_keys, new_keys = set(old), set(new)
    added = sorted(new_keys - old_keys)
    removed = sorted(old_keys - new_keys)
    modified = sorted(k for k in old_keys & new_keys if old[k] != new[k])
    return UnitChange(tuple(added), tuple(removed), tuple(modified))


# --------------------------------------------------------------------------- #
# Cache & stats                                                               #
# --------------------------------------------------------------------------- #

@dataclass
class _Entry:
    content_hash: str
    dep_sig: str
    result: SymResult


@dataclass
class IncrementalStats:
    """What a single incremental run reused vs. recomputed."""

    reused: Tuple[str, ...] = ()
    reanalyzed: Tuple[str, ...] = ()

    @property
    def total(self) -> int:
        return len(self.reused) + len(self.reanalyzed)


class IncrementalCache:
    """A persistent (across calls) cache of per-file analysis results.

    Hold one instance and pass it to successive :func:`analyze_package_incremental`
    / :func:`analyze_source_incremental` calls; entries are keyed by file path
    (package analysis) or filename (single file) and validated against a
    content + dependency signature on every call."""

    def __init__(self) -> None:
        self._entries: Dict[str, _Entry] = {}

    def __len__(self) -> int:
        return len(self._entries)

    def __contains__(self, key: str) -> bool:
        return key in self._entries

    def invalidate(self, key: str) -> None:
        self._entries.pop(key, None)

    def clear(self) -> None:
        self._entries.clear()

    # internal -------------------------------------------------------------
    def _get(self, key: str, content_hash: str, dep_sig: str) -> Optional[SymResult]:
        e = self._entries.get(key)
        if e is not None and e.content_hash == content_hash and e.dep_sig == dep_sig:
            return e.result
        return None

    def _put(self, key: str, content_hash: str, dep_sig: str, result: SymResult) -> None:
        self._entries[key] = _Entry(content_hash, dep_sig, result)


# --------------------------------------------------------------------------- #
# Dependency signature                                                        #
# --------------------------------------------------------------------------- #

def _dep_signature(index: PackageIndex, qualname: str) -> str:
    """Position-sensitive hash of every project definition that
    :meth:`PackageIndex.augmented_module` would inline into ``qualname`` — i.e.
    the cross-file code its analysis result depends on.  Empty (stable) when the
    module imports nothing project-local."""
    parts: List[str] = []
    for bound, node in sorted(
        index.direct_dependency_defs(qualname), key=lambda x: x[0]
    ):
        parts.append(bound)
        parts.append(ast.dump(node, include_attributes=True))
    return _sha("\x00".join(parts))


# --------------------------------------------------------------------------- #
# Drivers                                                                      #
# --------------------------------------------------------------------------- #

def analyze_package_incremental(
    root: str,
    cache: Optional[IncrementalCache] = None,
    *,
    budget_ms: Optional[float] = None,
) -> Tuple[PackageResult, IncrementalStats]:
    """Analyze a project under ``root``, reusing cached per-file results for
    files whose source *and* directly-imported project symbols are unchanged.

    Returns ``(PackageResult, IncrementalStats)``.  When ``cache`` is ``None`` a
    fresh one is used (every file is analysed and the cache is discarded — the
    result is identical to :func:`~src.symexec.package.analyze_package`).  The
    cache always reflects the latest analysis afterwards, so a follow-up call
    after an edit re-analyses only the affected files."""
    if cache is None:
        cache = IncrementalCache()
    index = PackageIndex.build(root)
    results: Dict[str, SymResult] = {}
    reused: List[str] = []
    reanalyzed: List[str] = []

    current_keys = set()
    for qual in sorted(index.modules):
        info = index.modules[qual]
        key = info.path
        current_keys.add(key)
        content_hash = _sha(info.source)
        dep_sig = _dep_signature(index, qual)
        cached = cache._get(key, content_hash, dep_sig)
        if cached is not None:
            results[key] = cached
            reused.append(key)
            continue
        res = _analyze_one(index, qual, budget_ms=budget_ms)
        cache._put(key, content_hash, dep_sig, res)
        results[key] = res
        reanalyzed.append(key)

    # Drop cache entries for files that no longer exist in the project.
    for stale in [k for k in cache._entries if k not in current_keys]:
        cache.invalidate(stale)

    stats = IncrementalStats(reused=tuple(reused), reanalyzed=tuple(reanalyzed))
    return PackageResult(root=root, results=results, index=index), stats


def analyze_source_incremental(
    source: str,
    filename: str = "<unknown>",
    cache: Optional[IncrementalCache] = None,
    *,
    budget_ms: Optional[float] = None,
) -> Tuple[SymResult, bool]:
    """Single-file incremental analysis keyed by file content.

    Returns ``(result, reused)``.  A standalone file has no project-local
    dependencies, so its result depends only on its own source — the cached
    result is reused verbatim whenever the source is unchanged."""
    if cache is None:
        cache = IncrementalCache()
    content_hash = _sha(source)
    cached = cache._get(filename, content_hash, "")
    if cached is not None:
        return cached, True
    res = analyze_source(source, filename=filename, budget_ms=budget_ms)
    cache._put(filename, content_hash, "", res)
    return res, False
