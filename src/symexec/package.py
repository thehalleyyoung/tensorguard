"""Whole-package analysis: cross-file import resolution + project call graph
(roadmap Step 82).

:func:`analyze_source` reasons about one file at a time.  Real models, though,
are split across many: a ``model.py`` ``import``\\s an ``Encoder`` from
``layers/encoder.py`` and composes it.  Single-file analysis abstains at the
import boundary — it never sees ``Encoder``'s ``forward`` — so layer-to-layer
shape mismatches that only manifest when one module *uses* another go unfound.

This module closes that gap.  It

1.  discovers every ``.py`` file under a project root and assigns each a dotted
    module qualname (``a/b/c.py`` → ``a.b.c``; ``a/b/__init__.py`` → ``a.b``);
2.  builds, per module, a **symbol table** (top-level ``class``/``def`` names →
    their AST nodes) and an **import map** (each bound alias → the
    ``(module, name)`` it refers to), resolving ``import``, ``import … as``,
    ``from m import n [as a]`` and relative imports (``from . import m``,
    ``from .pkg import n``) against the importing module's package;
3.  :meth:`PackageIndex.resolve`\\s an ``(module, alias)`` pair through the
    import chain (re-exports included) to the defining ``class``/``def``;
4.  for each module produces an **import-augmented** :class:`ast.Module` — the
    file's own body plus a renamed copy of every directly-imported definition —
    and runs the *unchanged* analysis passes over it, so calls into imported
    symbols are followed interprocedurally; and
5.  exposes the resolved cross-file **call graph** for inspection.

Soundness is preserved throughout: only *real* project definitions are inlined,
each bound to the exact alias it was imported under, so no name is ever
mis-resolved to an unrelated same-named symbol (which could fabricate a bug).
Anything that cannot be resolved (a third-party import, a transitive symbol the
importer did not itself import) stays unresolved and the engine abstains — lost
coverage, never a false report.  The module is torch-free.
"""

from __future__ import annotations

import ast
import copy
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

from .engine import SymResult, _analyze_module
from .interpreter import Interpreter
from .config import SymConfig

__all__ = [
    "ModuleInfo",
    "PackageIndex",
    "PackageResult",
    "analyze_package",
]


# --------------------------------------------------------------------------- #
# Per-module facts                                                            #
# --------------------------------------------------------------------------- #

@dataclass
class ModuleInfo:
    """Parsed facts about one source file in the package."""

    qualname: str
    path: str
    tree: ast.Module
    is_package: bool  # True for ``__init__.py``
    source: str = ""  # raw file text (for content hashing in incremental analysis)
    #: top-level ``class``/``def`` name -> defining node
    symbols: Dict[str, ast.AST] = field(default_factory=dict)
    #: bound alias -> (target_module_qualname, target_name | None)
    #: ``None`` target_name means the alias binds a *module* (``import a.b``).
    imports: Dict[str, Tuple[str, Optional[str]]] = field(default_factory=dict)


def _qualname_for(root: str, path: str) -> str:
    rel = os.path.relpath(path, root)
    parts = rel.replace(os.sep, "/").split("/")
    if parts[-1] == "__init__.py":
        parts = parts[:-1]
    else:
        parts[-1] = parts[-1][:-3]  # strip ``.py``
    return ".".join(p for p in parts if p)


def _package_of(qualname: str, is_package: bool) -> str:
    """The package a module lives in (its qualname minus the final component,
    unless the module *is* a package, in which case it is its own package)."""
    if is_package:
        return qualname
    return qualname.rpartition(".")[0]


def _resolve_relative(package: str, level: int, module: Optional[str]) -> Optional[str]:
    """Resolve a ``from`` target given the importer's ``package`` and the
    relative ``level`` (number of leading dots).  Returns the absolute module
    qualname, or ``None`` if the relative reference escapes the project root."""
    if level == 0:
        return module
    base_parts = package.split(".") if package else []
    # level 1 == current package; each extra dot strips one more package level.
    strip = level - 1
    if strip > len(base_parts):
        return None
    base = base_parts[: len(base_parts) - strip] if strip else base_parts
    target = ".".join(base)
    if module:
        target = f"{target}.{module}" if target else module
    return target or None


# --------------------------------------------------------------------------- #
# The index                                                                   #
# --------------------------------------------------------------------------- #

class PackageIndex:
    """An index of every module under a root, with import resolution."""

    def __init__(self) -> None:
        self.modules: Dict[str, ModuleInfo] = {}

    # -- construction -------------------------------------------------------
    @classmethod
    def build(cls, root: str) -> "PackageIndex":
        idx = cls()
        for dirpath, dirnames, filenames in os.walk(root):
            # Skip hidden dirs, caches and virtualenvs.
            dirnames[:] = [
                d
                for d in dirnames
                if not d.startswith(".") and d not in ("__pycache__", "node_modules")
            ]
            for fn in sorted(filenames):
                if not fn.endswith(".py"):
                    continue
                path = os.path.join(dirpath, fn)
                try:
                    with open(path, "r", encoding="utf-8") as fh:
                        text = fh.read()
                    tree = ast.parse(text, filename=path)
                except (SyntaxError, OSError, UnicodeDecodeError):
                    continue  # unparseable file: skip (single-file analysis still flags it)
                qual = _qualname_for(root, path)
                info = ModuleInfo(
                    qualname=qual,
                    path=path,
                    tree=tree,
                    is_package=(fn == "__init__.py"),
                    source=text,
                )
                idx.modules[qual] = info
        for info in idx.modules.values():
            idx._index_symbols(info)
            idx._index_imports(info)
        return idx

    def _index_symbols(self, info: ModuleInfo) -> None:
        for node in info.tree.body:
            if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                info.symbols[node.name] = node

    def _index_imports(self, info: ModuleInfo) -> None:
        pkg = _package_of(info.qualname, info.is_package)
        for node in info.tree.body:
            if isinstance(node, ast.Import):
                for alias in node.names:
                    bound = alias.asname or alias.name.split(".")[0]
                    # ``import a.b.c`` binds ``a``; ``import a.b as c`` binds ``c``.
                    target = alias.name if alias.asname else alias.name.split(".")[0]
                    info.imports[bound] = (target, None)
            elif isinstance(node, ast.ImportFrom):
                target_mod = _resolve_relative(pkg, node.level, node.module)
                if target_mod is None:
                    continue
                for alias in node.names:
                    if alias.name == "*":
                        continue
                    bound = alias.asname or alias.name
                    info.imports[bound] = (target_mod, alias.name)

    # -- resolution ---------------------------------------------------------
    def resolve(
        self, qualname: str, name: str, _seen: Optional[Set[Tuple[str, str]]] = None
    ) -> Optional[Tuple[str, ast.AST]]:
        """Resolve ``name`` as seen in module ``qualname`` to its defining
        ``(module_qualname, node)``, following import chains and re-exports.

        Returns ``None`` for anything outside the project (third-party / stdlib
        imports), module-only imports, or unresolvable references."""
        info = self.modules.get(qualname)
        if info is None:
            return None
        if name in info.symbols:
            return (qualname, info.symbols[name])
        binding = info.imports.get(name)
        if binding is None:
            return None
        target_mod, target_name = binding
        if target_name is None:
            return None  # binds a module, not a class/def
        _seen = _seen or set()
        key = (target_mod, target_name)
        if key in _seen:
            return None  # import cycle guard
        _seen.add(key)
        return self.resolve(target_mod, target_name, _seen)

    def direct_dependency_defs(self, qualname: str) -> List[Tuple[str, ast.AST]]:
        """The ``(alias, def_node)`` pairs that :meth:`augmented_module` inlines
        for ``qualname``: every directly-imported class/def that resolves inside
        the project and is not shadowed by a local definition.

        Factored out so incremental analysis can hash exactly the foreign
        definitions a file's analysis depends on (Step 84) — the single source of
        truth for "what cross-file code does this module's result depend on"."""
        info = self.modules[qualname]
        local_names = set(info.symbols)
        out: List[Tuple[str, ast.AST]] = []
        for bound, (_target_mod, target_name) in info.imports.items():
            if target_name is None:
                continue  # binds a module, not a class/def
            if bound in local_names:
                continue  # a local def shadows the import
            resolved = self.resolve(qualname, bound)
            if resolved is None:
                continue
            _def_mod, def_node = resolved
            if not isinstance(
                def_node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)
            ):
                continue
            out.append((bound, def_node))
        return out

    def augmented_module(self, qualname: str) -> Tuple[ast.Module, Set[int]]:
        """Return ``(module, injected_ids)`` for ``qualname``: a shallow copy of
        the module whose body is extended with a renamed copy of every
        *directly-imported* class/def that resolves inside the project.

        ``injected_ids`` are the ``id()`` of those injected nodes so the engine
        can resolve calls into them without re-analysing them as local defs."""
        info = self.modules[qualname]
        body = list(info.tree.body)
        injected_ids: Set[int] = set()
        for bound, def_node in self.direct_dependency_defs(qualname):
            clone = copy.deepcopy(def_node)
            clone.name = bound  # bind under the importer's alias
            ast.fix_missing_locations(clone)
            injected_ids.add(id(clone))
            body.append(clone)
        new_mod = ast.Module(body=body, type_ignores=list(info.tree.type_ignores))
        return new_mod, injected_ids

    # -- call graph ---------------------------------------------------------
    def call_graph(self) -> Dict[str, List[str]]:
        """Resolved cross-file call/instantiation edges.

        Maps ``"module:symbol"`` (a top-level class or function that contains the
        call site) to a sorted list of ``"module:symbol"`` targets it references
        that resolve to a definition in *another* module.  Only cross-file edges
        are reported — intra-file calls are already covered by single-file
        interprocedural analysis."""
        graph: Dict[str, Set[str]] = {}
        for qual, info in self.modules.items():
            for top in info.tree.body:
                if not isinstance(
                    top, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)
                ):
                    continue
                src_key = f"{qual}:{top.name}"
                for call in ast.walk(top):
                    if not isinstance(call, ast.Call):
                        continue
                    callee = call.func
                    name = None
                    if isinstance(callee, ast.Name):
                        name = callee.id
                    elif isinstance(callee, ast.Attribute) and isinstance(
                        callee.value, ast.Name
                    ):
                        # ``mod.Thing(...)`` where ``mod`` is an imported module.
                        modbind = info.imports.get(callee.value.id)
                        if modbind is not None and modbind[1] is None:
                            tgt = self.resolve(modbind[0], callee.attr)
                            if tgt is not None and tgt[0] != qual:
                                graph.setdefault(src_key, set()).add(
                                    f"{tgt[0]}:{callee.attr}"
                                )
                        continue
                    if name is None:
                        continue
                    resolved = self.resolve(qual, name)
                    if resolved is not None and resolved[0] != qual:
                        graph.setdefault(src_key, set()).add(
                            f"{resolved[0]}:{name}"
                        )
        return {k: sorted(v) for k, v in sorted(graph.items())}


# --------------------------------------------------------------------------- #
# Driver                                                                       #
# --------------------------------------------------------------------------- #

@dataclass
class PackageResult:
    """Aggregate of a whole-package analysis."""

    root: str
    results: Dict[str, SymResult]  # keyed by file path
    index: PackageIndex

    @property
    def files_analyzed(self) -> int:
        return len(self.results)

    @property
    def functions_analyzed(self) -> int:
        return sum(r.functions_analyzed for r in self.results.values())

    def all_bugs(self) -> List:
        """Every bug across all files, de-duplicated by (path, kind, line, col,
        message) and ordered by file then position."""
        seen = set()
        out = []
        for path in sorted(self.results):
            for b in self.results[path].bugs:
                key = (path, b.kind, b.line, b.col, b.message)
                if key in seen:
                    continue
                seen.add(key)
                out.append((path, b))
        return out

    def call_graph(self) -> Dict[str, List[str]]:
        return self.index.call_graph()


def _analyze_one(
    index: PackageIndex,
    qualname: str,
    *,
    budget_ms: Optional[float] = None,
    config: "SymConfig | None" = None,
) -> SymResult:
    """Analyze a single project module over its import-augmented form.

    The atomic analysis unit for both the whole-package driver and incremental
    re-analysis (Step 84), so a re-analysed module is byte-identical to a fresh
    whole-package run of the same module.  ``config`` (Step 86) selects the
    soundness mode; the default is ``balanced``."""
    info = index.modules[qualname]
    aug_module, injected_ids = index.augmented_module(qualname)
    interp = Interpreter(aug_module, filename=info.path, config=config)
    return _analyze_module(
        aug_module,
        interp,
        filename=info.path,
        budget_ms=budget_ms,
        skip_ids=frozenset(injected_ids),
    )


def analyze_package(
    root: str,
    *,
    budget_ms: Optional[float] = None,
    config: "SymConfig | None" = None,
) -> PackageResult:
    """Analyze every module under ``root`` with cross-file import resolution.

    Each file is analysed over its import-augmented module so that calls into
    classes/functions imported from sibling files are followed
    interprocedurally.  ``budget_ms`` (optional) bounds per-file analysis as in
    :func:`analyze_source`; ``config`` (Step 86) selects the soundness mode."""
    index = PackageIndex.build(root)
    results: Dict[str, SymResult] = {}
    for qual in sorted(index.modules):
        info = index.modules[qual]
        results[info.path] = _analyze_one(
            index, qual, budget_ms=budget_ms, config=config
        )
    return PackageResult(root=root, results=results, index=index)
