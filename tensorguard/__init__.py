"""TensorGuard — the importable top-level package (Step 161).

The verifier's implementation lives in the ``src`` package (kept stable for the
console-script and ``pytest11`` entry points). This module is the *public import
surface* a ``pip install tensorguard`` user reaches for::

    from tensorguard import verify_architecture, analyze, AnalysisResult

It re-exports exactly the stability-guaranteed names from :mod:`src` (the set
pinned by ``tests/test_api_stability.py`` and documented in
``DEPRECATION_POLICY.md``), plus the Phase-7 integration entry points
(``guarded_compile``, ``make_tensorguard_backend``, ``verify_module``,
``TensorGuardViolation``) so the adoption paths in the README work against the
real package, not a private module path.

Submodules remain reachable as ``tensorguard.api``, ``tensorguard.torch`` etc.
for callers that want the full surface.
"""

from __future__ import annotations

from src import (  # noqa: F401  (re-export)
    AnalysisResult,
    Bug,
    BugCategory,
    SourceLocation,
    TensorGuardCheckError,
    __version__,
    analyze,
    analyze_directory,
    analyze_file,
    analyze_function,
    checked,
    is_static_only_source,
    quick_check,
    verify_architecture,
    verify_file_safely,
    verify_source_safely,
)
from src.torch_integration import (  # noqa: F401  (re-export)
    TensorGuardViolation,
    guarded_aot_package,
    guarded_compile,
    guarded_onnx_export,
    make_tensorguard_backend,
    verify_exported_program,
    verify_module,
)

from src.einops_verify import verify_einops  # noqa: F401  (re-export)
from src.einops_source import verify_einops_source  # noqa: F401  (re-export)
from src.distributions_verify import (  # noqa: F401  (re-export)
    verify_distribution,
    verify_log_prob,
)

# Lazily importable submodule aliases (``import tensorguard.api`` etc.).
from src import api as api  # noqa: F401
from src import torch_integration as torch  # noqa: F401

__all__ = [
    "analyze",
    "analyze_file",
    "analyze_directory",
    "analyze_function",
    "quick_check",
    "verify_architecture",
    "verify_file_safely",
    "verify_source_safely",
    "is_static_only_source",
    "AnalysisResult",
    "Bug",
    "BugCategory",
    "SourceLocation",
    "checked",
    "TensorGuardCheckError",
    "verify_module",
    "guarded_compile",
    "make_tensorguard_backend",
    "guarded_onnx_export",
    "verify_exported_program",
    "guarded_aot_package",
    "verify_einops",
    "verify_einops_source",
    "verify_distribution",
    "verify_log_prob",
    "TensorGuardViolation",
    "api",
    "torch",
    "__version__",
]
