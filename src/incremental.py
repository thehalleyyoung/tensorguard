"""Step 48 -- incremental re-verification.

Re-running the full verifier on an unchanged model is wasted work.  This module
productionises *incremental analysis*: it caches verification verdicts keyed by a
**dependency-aware structural fingerprint** and, on a source diff, re-verifies
only the models whose fingerprint actually changed.

Soundness rests on a precise dependency notion.  TensorGuard verifies the *root*
``nn.Module`` of a source (the class not instantiated by any other), inlining
its base classes and user-defined submodules (Steps 43/44).  A root's verdict
therefore depends only on:

  * the root class's own source, and
  * the source of every user-defined class reachable from it (base classes and
    instantiated/called submodules), transitively.

Library layers (``nn.Linear`` etc.) have fixed semantics and never enter the
fingerprint.  The cache key combines the structural hash of every class in this
transitive closure with a hash of the verification options (input shapes, the
``check_*`` flags).  Consequently:

  * editing an *unrelated* sibling class leaves the fingerprint unchanged -> the
    cached verdict is reused (sound: the root cannot depend on it);
  * editing the root or any class it transitively uses changes the fingerprint
    -> the model is re-verified.

The cache is plain JSON, so it survives across processes / CI runs.
"""
from __future__ import annotations

import ast
import hashlib
import json
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

from src.model_checker import (
    _collect_module_classes,
    _find_root_module,
    verify_model,
)

__all__ = [
    "class_structural_hashes",
    "class_dependency_graph",
    "root_dependency_closure",
    "model_fingerprint",
    "changed_models",
    "IncrementalResult",
    "IncrementalVerifier",
]


def _parse(source: str) -> ast.AST:
    return ast.parse(source)


def _module_class_map(source: str) -> Dict[str, ast.ClassDef]:
    tree = _parse(source)
    classes = _collect_module_classes(tree)
    return {c.name: c for c in classes}


def class_structural_hashes(source: str) -> Dict[str, str]:
    """SHA-256 of the normalized AST dump of every module class."""
    out: Dict[str, str] = {}
    for name, node in _module_class_map(source).items():
        dumped = ast.dump(node, annotate_fields=True, include_attributes=False)
        out[name] = hashlib.sha256(dumped.encode("utf-8")).hexdigest()
    return out


def _referenced_names(node: ast.ClassDef) -> Set[str]:
    """All bare names / attribute roots / base-class names used in *node*."""
    names: Set[str] = set()
    for base in node.bases:
        if isinstance(base, ast.Name):
            names.add(base.id)
        elif isinstance(base, ast.Attribute):
            names.add(base.attr)
    for sub in ast.walk(node):
        if isinstance(sub, ast.Name):
            names.add(sub.id)
        elif isinstance(sub, ast.Attribute):
            names.add(sub.attr)
    return names


def class_dependency_graph(source: str) -> Dict[str, Set[str]]:
    """Map each module class to the set of *module-local* classes it uses."""
    cmap = _module_class_map(source)
    local = set(cmap)
    graph: Dict[str, Set[str]] = {}
    for name, node in cmap.items():
        refs = _referenced_names(node) & local
        refs.discard(name)
        graph[name] = refs
    return graph


def root_dependency_closure(source: str) -> Tuple[str, Set[str]]:
    """Return (root_class_name, transitive closure of classes it depends on)."""
    cmap = _module_class_map(source)
    if not cmap:
        raise ValueError("no nn.Module subclass found in source")
    root = _find_root_module(list(cmap.values())).name
    graph = class_dependency_graph(source)
    seen: Set[str] = set()
    stack: List[str] = [root]
    while stack:
        cur = stack.pop()
        if cur in seen:
            continue
        seen.add(cur)
        stack.extend(graph.get(cur, set()) - seen)
    return root, seen


def _options_hash(
    input_shapes: Optional[Dict[str, tuple]],
    options: Dict[str, object],
) -> str:
    payload = {
        "input_shapes": {k: list(v) for k, v in (input_shapes or {}).items()},
        "options": {k: options[k] for k in sorted(options)},
    }
    blob = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def model_fingerprint(
    source: str,
    input_shapes: Optional[Dict[str, tuple]] = None,
    options: Optional[Dict[str, object]] = None,
) -> str:
    """Dependency-aware fingerprint of the *root* model + its options.

    Two sources with the same fingerprint are guaranteed to produce the same
    verification verdict for the root model under the same options.
    """
    _root, closure = root_dependency_closure(source)
    hashes = class_structural_hashes(source)
    parts = ["%s=%s" % (n, hashes[n]) for n in sorted(closure)]
    struct = hashlib.sha256("\n".join(parts).encode("utf-8")).hexdigest()
    opt = _options_hash(input_shapes, options or {})
    return hashlib.sha256(("%s:%s" % (struct, opt)).encode("utf-8")).hexdigest()


def changed_models(
    old_source: str,
    new_source: str,
    input_shapes: Optional[Dict[str, tuple]] = None,
    options: Optional[Dict[str, object]] = None,
) -> bool:
    """True iff the root model must be re-verified after old -> new."""
    try:
        old_fp = model_fingerprint(old_source, input_shapes, options)
    except ValueError:
        return True
    new_fp = model_fingerprint(new_source, input_shapes, options)
    return old_fp != new_fp


@dataclass
class IncrementalResult:
    """Outcome of an incremental verification call."""
    fingerprint: str
    from_cache: bool
    safe: bool
    num_violations: int
    root: str


class IncrementalVerifier:
    """Verdict cache keyed by dependency-aware model fingerprints.

    Parameters
    ----------
    cache_path : str, optional
        JSON file used to persist verdicts across processes.  When omitted the
        cache lives only in memory.
    """

    _RELEVANT_OPTIONS = (
        "default_device", "default_phase", "check_devices", "check_phases",
        "check_gradients", "check_dtypes", "high_confidence_only",
        "verification_mode", "infer_inputs",
    )

    def __init__(self, cache_path: Optional[str] = None) -> None:
        self.cache_path = cache_path
        self._cache: Dict[str, Dict[str, object]] = {}
        if cache_path and os.path.exists(cache_path):
            try:
                with open(cache_path) as fh:
                    self._cache = json.load(fh)
            except (OSError, ValueError):
                self._cache = {}
        self.hits = 0
        self.misses = 0

    def _options(self, verify_kwargs: Dict[str, object]) -> Dict[str, object]:
        return {k: verify_kwargs[k] for k in self._RELEVANT_OPTIONS
                if k in verify_kwargs}

    def fingerprint(
        self, source: str,
        input_shapes: Optional[Dict[str, tuple]] = None,
        **verify_kwargs: object,
    ) -> str:
        return model_fingerprint(
            source, input_shapes, self._options(verify_kwargs))

    def verify(
        self, source: str,
        input_shapes: Optional[Dict[str, tuple]] = None,
        **verify_kwargs: object,
    ) -> IncrementalResult:
        """Verify *source*, reusing a cached verdict when the fingerprint hits."""
        fp = self.fingerprint(source, input_shapes, **verify_kwargs)
        root, _ = root_dependency_closure(source)
        cached = self._cache.get(fp)
        if cached is not None:
            self.hits += 1
            return IncrementalResult(
                fingerprint=fp, from_cache=True,
                safe=bool(cached["safe"]),
                num_violations=int(cached["num_violations"]),
                root=str(cached.get("root", root)),
            )
        self.misses += 1
        result = verify_model(source, input_shapes=input_shapes,
                              **verify_kwargs)
        n_viol = len(getattr(result, "violations", []) or [])
        entry = {"safe": bool(result.safe), "num_violations": n_viol,
                 "root": root}
        self._cache[fp] = entry
        return IncrementalResult(
            fingerprint=fp, from_cache=False, safe=bool(result.safe),
            num_violations=n_viol, root=root)

    def invalidate(self, source: str,
                   input_shapes: Optional[Dict[str, tuple]] = None,
                   **verify_kwargs: object) -> bool:
        """Drop the cached verdict for *source* (if present)."""
        fp = self.fingerprint(source, input_shapes, **verify_kwargs)
        return self._cache.pop(fp, None) is not None

    def save(self) -> None:
        if not self.cache_path:
            return
        with open(self.cache_path, "w") as fh:
            json.dump(self._cache, fh, indent=2, sort_keys=True)

    def __len__(self) -> int:
        return len(self._cache)
