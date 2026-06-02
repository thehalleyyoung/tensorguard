"""``@tensorguard.checked`` — opt-in, per-module verification at definition time.

Step 63.  Decorating an ``nn.Module`` subclass with :func:`checked` verifies its
architecture the moment the class is defined (i.e. at import), so a shape bug is
caught before the model is ever instantiated or trained.  The decorator returns
the class untouched, so decorated modules behave exactly as normal.

Usage::

    import tensorguard

    @tensorguard.checked(input_shapes={"x": ("batch", 10)})
    class Net(nn.Module):
        ...

By default a detected bug raises :class:`TensorGuardCheckError` with the full
source-mapped diagnostic; ``on_fail="warn"`` or ``"log"`` downgrade that to a
warning / printed message for gradual adoption.  If the class's source cannot be
recovered (e.g. defined in an ``exec`` with no source), the decorator abstains
silently rather than failing.
"""

from __future__ import annotations

import ast
import inspect
import textwrap
import warnings
from typing import Any, Dict, Optional


class TensorGuardCheckError(Exception):
    """Raised by ``@checked`` when a decorated module fails verification."""

    def __init__(self, message: str, result: Any = None):
        super().__init__(message)
        self.result = result


def _strip_decorators(source: str) -> str:
    """Return *source* with all decorators removed.

    ``inspect.getsource`` of a decorated class includes the ``@checked(...)``
    line, which references names (``tensorguard`` / ``checked``) that are
    irrelevant to verification and would otherwise sit in the parsed tree.
    Removing every decorator yields a clean class definition to verify.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return source
    for node in ast.walk(tree):
        if isinstance(
            node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)
        ):
            node.decorator_list = []
    try:
        return ast.unparse(tree)
    except Exception:
        return source


def _error_bugs(result: Any) -> int:
    diagnostics = getattr(result, "diagnostics", None)
    if diagnostics:
        return len(diagnostics)
    bugs = getattr(result, "bugs", None) or []
    return sum(1 for b in bugs if getattr(b, "severity", "error") == "error")


def _render(result: Any) -> str:
    diagnostics = list(getattr(result, "diagnostics", []) or [])
    if diagnostics:
        try:
            from src.source_mapped_errors import format_plain
            return format_plain(diagnostics)
        except Exception:
            pass
    return "\n".join(
        f"L{getattr(getattr(b, 'location', None), 'line', '?')}: {getattr(b, 'message', '')}"
        for b in (getattr(result, "bugs", None) or [])
    )


def checked(
    cls: Optional[type] = None,
    *,
    input_shapes: Optional[Dict[str, tuple]] = None,
    on_fail: str = "raise",
    verbose: bool = False,
    **verify_kwargs: Any,
):
    """Class decorator that verifies an ``nn.Module`` at definition time.

    May be used bare (``@checked``) or with arguments
    (``@checked(input_shapes=..., on_fail="warn")``).  The verification result
    is stored on the class as ``__tensorguard_result__`` for introspection.
    """

    def decorate(klass: type) -> type:
        try:
            raw = inspect.getsource(klass)
        except (OSError, TypeError):
            # No source available (e.g. exec without a file): abstain.
            klass.__tensorguard_result__ = None
            return klass
        source = _strip_decorators(textwrap.dedent(raw))

        try:
            from src.api import verify_architecture
            result = verify_architecture(
                source, input_shapes=input_shapes, **verify_kwargs
            )
        except Exception as e:
            # Verification machinery failed; never block the user's import.
            klass.__tensorguard_result__ = None
            if verbose:
                warnings.warn(
                    f"tensorguard: could not check {klass.__name__}: {e}"
                )
            return klass

        klass.__tensorguard_result__ = result
        n = _error_bugs(result)
        if n > 0:
            noun = "issue" if n == 1 else "issues"
            header = f"tensorguard: {klass.__name__} \u2014 {n} {noun}"
            body = f"{header}\n{_render(result)}"
            if on_fail == "raise":
                raise TensorGuardCheckError(body, result)
            elif on_fail == "warn":
                warnings.warn(body)
            elif on_fail == "log":
                print(body)
            # any other value: store result silently
        elif verbose:
            print(f"tensorguard: {klass.__name__} verified safe")
        return klass

    # Bare usage: @checked  (cls is the class itself)
    if cls is not None and isinstance(cls, type):
        return decorate(cls)
    # Parameterized usage: @checked(...)
    return decorate
