"""Step 79-80 — safe-by-construction entry points for untrusted model files.

TensorGuard's threat model (see ``SECURITY.md``) treats analysed files as
**untrusted input**.  All static analysis is performed on the *source text* via
Python's ``ast`` module — the file is never ``import``-ed, ``exec``-uted, or
``eval``-uated — so verifying a malicious file is harmless.

This module exposes the recommended entry points for untrusted input and a guard
that asserts the property holds, making the safety contract explicit and
testable:

* :func:`verify_file_safely` — read a file as text and statically verify it; the
  file's top-level code never runs.
* :func:`verify_source_safely` — same, for an in-memory source string.
* :func:`is_static_only_source` — sanity check that a source parses as Python
  (so analysis stays on the AST path) without executing anything.

The runtime graph extractors (``fx``/``dynamo``/``export``) require an *already
instantiated* ``nn.Module`` supplied by the caller; they are never invoked by
these source-level entry points, so no untrusted code path runs here.
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Any, Dict, Optional, Tuple


def is_static_only_source(source: str) -> bool:
    """True if *source* parses as Python (kept on the AST path), else False.

    Parsing with :func:`ast.parse` compiles nothing and executes nothing.
    """
    try:
        ast.parse(source)
        return True
    except SyntaxError:
        return False


def verify_source_safely(
    source: str,
    *,
    input_shapes: Optional[Dict[str, Tuple]] = None,
    soundness_mode: str = "balanced",
    filename: str = "<untrusted>",
):
    """Statically verify untrusted *source*; its top-level code never executes."""
    from src.api import verify_architecture

    return verify_architecture(
        source,
        input_shapes=input_shapes,
        soundness_mode=soundness_mode,
        filename=filename,
    )


def verify_file_safely(
    path: str,
    *,
    input_shapes: Optional[Dict[str, Tuple]] = None,
    soundness_mode: str = "balanced",
):
    """Read *path* as text and statically verify it; the file is never imported.

    Returns the ``AnalysisResult``.  ``FileNotFoundError`` is raised if the path
    does not exist; decoding errors are replaced rather than executed.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(path)
    source = p.read_text(encoding="utf-8", errors="replace")
    return verify_source_safely(
        source,
        input_shapes=input_shapes,
        soundness_mode=soundness_mode,
        filename=str(p),
    )
