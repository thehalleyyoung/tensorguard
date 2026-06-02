"""Jupyter / IPython integration: check a model the moment a cell defines it.

Step 62.  When a notebook cell defines an ``nn.Module`` subclass, TensorGuard
can verify it immediately and surface any shape bug inline, turning the notebook
into a live verification surface.

The substance lives in three pure, side-effect-free functions —
:func:`find_module_classes` (does this cell define a model?), :func:`check_cell`
(verify it), and :func:`format_cell_report` (render a concise verdict) — so the
behavior is fully unit-testable without a running kernel.  On top of that sits a
thin IPython layer: a ``post_run_cell`` event hook (registered by
:func:`load_ipython_extension`, i.e. ``%load_ext src.jupyter_integration``) that
checks every cell after it runs, plus a ``%%tensorguard`` cell magic for
explicit, shape-annotated checks.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class CellCheckOutcome:
    """Result of (maybe) checking a notebook cell."""

    module_names: List[str] = field(default_factory=list)
    checked: bool = False  # did a verification actually run?
    safe: bool = True
    bug_count: int = 0
    headline: str = ""
    detail: str = ""  # rendered diagnostics (plain text)
    error: Optional[str] = None
    result: Any = None


def _base_is_module(base: ast.expr) -> bool:
    """True if an AST base expression denotes ``nn.Module`` (or ``Module``)."""
    if isinstance(base, ast.Attribute):
        return base.attr == "Module"
    if isinstance(base, ast.Name):
        return base.id == "Module"
    return False


def find_module_classes(cell_source: str) -> List[str]:
    """Return the names of classes in *cell_source* that subclass nn.Module.

    Pure AST inspection — never imports or executes the cell.  Returns ``[]``
    for unparsable cells (a half-typed cell must not raise).
    """
    try:
        tree = ast.parse(cell_source)
    except SyntaxError:
        return []
    names: List[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            if any(_base_is_module(b) for b in node.bases):
                names.append(node.name)
    return names


def _bug_count(result: Any) -> int:
    diagnostics = getattr(result, "diagnostics", None)
    if diagnostics:
        return len(diagnostics)
    bugs = getattr(result, "bugs", None) or []
    return sum(1 for b in bugs if getattr(b, "severity", "error") == "error")


def check_cell(
    cell_source: str,
    input_shapes: Optional[Dict[str, tuple]] = None,
    **verify_kwargs: Any,
) -> CellCheckOutcome:
    """Verify the model defined in *cell_source*, if any.

    Returns an outcome with ``checked=False`` when the cell defines no
    ``nn.Module`` (the common case, so the hook stays silent for ordinary
    cells).  Verification errors are captured rather than raised so a notebook
    is never interrupted.
    """
    module_names = find_module_classes(cell_source)
    outcome = CellCheckOutcome(module_names=module_names)
    if not module_names:
        return outcome

    try:
        from src.api import verify_architecture
        result = verify_architecture(
            cell_source, input_shapes=input_shapes, **verify_kwargs
        )
    except Exception as e:
        outcome.checked = True
        outcome.safe = False
        outcome.error = str(e)
        outcome.headline = (
            f"! tensorguard could not check "
            f"{', '.join(module_names)}: {e}"
        )
        return outcome

    n = _bug_count(result)
    outcome.checked = True
    outcome.result = result
    outcome.bug_count = n
    outcome.safe = n == 0
    label = ", ".join(module_names)
    if n == 0:
        outcome.headline = f"\u2713 tensorguard: {label} verified safe"
    else:
        noun = "issue" if n == 1 else "issues"
        outcome.headline = f"\u2717 tensorguard: {label} \u2014 {n} {noun}"
        diagnostics = list(getattr(result, "diagnostics", []) or [])
        if diagnostics:
            try:
                from src.source_mapped_errors import format_plain
                outcome.detail = format_plain(diagnostics)
            except Exception:
                outcome.detail = ""
    return outcome


def format_cell_report(
    outcome: CellCheckOutcome, use_color: bool = False
) -> str:
    """Render an outcome to a concise multi-line report (empty if unchecked)."""
    if not outcome.checked:
        return ""
    if not use_color:
        head = outcome.headline
    else:
        if outcome.safe:
            head = f"\033[32m{outcome.headline}\033[0m"
        else:
            head = f"\033[31m{outcome.headline}\033[0m"
    if outcome.detail:
        return head + "\n" + outcome.detail
    return head


# ── IPython layer (thin; guarded so the pure core always imports) ───────────

def run_cell_check(
    cell_source: str,
    input_shapes: Optional[Dict[str, tuple]] = None,
    printer=print,
    use_color: bool = False,
    **verify_kwargs: Any,
) -> CellCheckOutcome:
    """Check a cell and print the report via *printer*; return the outcome."""
    outcome = check_cell(cell_source, input_shapes=input_shapes, **verify_kwargs)
    report = format_cell_report(outcome, use_color=use_color)
    if report:
        printer(report)
    return outcome


def _parse_shape_args(arg_line: str) -> Dict[str, tuple]:
    """Parse ``x=batch,10 y=1,3,8`` style shape annotations from a magic line."""
    shapes: Dict[str, tuple] = {}
    for tok in (arg_line or "").split():
        if "=" not in tok:
            continue
        name, dims_str = tok.split("=", 1)
        dims = []
        for d in dims_str.split(","):
            d = d.strip()
            if not d:
                continue
            try:
                dims.append(int(d))
            except ValueError:
                dims.append(d)
        shapes[name] = tuple(dims)
    return shapes


def load_ipython_extension(ipython) -> None:
    """Register the post-run-cell hook and the ``%%tensorguard`` magic."""

    def _after_run_cell(result) -> None:
        cell = getattr(getattr(result, "info", None), "raw_cell", None)
        if cell is None:
            cell = getattr(result, "raw_cell", None)
        if not cell:
            return
        try:
            run_cell_check(cell, use_color=True)
        except Exception:
            # An integration bug must never break the user's notebook.
            pass

    ipython.events.register("post_run_cell", _after_run_cell)
    # Stash so unload can deregister exactly this callback.
    ipython._tensorguard_hook = _after_run_cell

    try:
        from IPython.core.magic import Magics, cell_magic, magics_class

        @magics_class
        class _TensorGuardMagics(Magics):
            @cell_magic
            def tensorguard(self, line, cell):
                shapes = _parse_shape_args(line)
                run_cell_check(
                    cell, input_shapes=shapes or None, use_color=True
                )
                # Still execute the cell so the class is actually defined.
                self.shell.run_cell(cell)

        ipython.register_magics(_TensorGuardMagics)
    except Exception:
        pass


def unload_ipython_extension(ipython) -> None:
    """Deregister the post-run-cell hook."""
    hook = getattr(ipython, "_tensorguard_hook", None)
    if hook is not None:
        try:
            ipython.events.unregister("post_run_cell", hook)
        except Exception:
            pass
