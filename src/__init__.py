"""
GuardHarvest: Find bugs in Python code with zero annotations.

Flow-sensitive abstract interpreter that harvests programmer-written guards
(isinstance, is not None, comparisons) as implicit refinement type predicates.
"""

__version__ = "0.1.0"

from .api import (
    analyze,
    analyze_file,
    analyze_directory,
    analyze_function,
    quick_check,
    verify_architecture,
    AnalysisResult,
    Bug,
    BugCategory,
    SourceLocation,
)
from .runtime_check import checked, TensorGuardCheckError
from .safe_loader import (
    verify_file_safely,
    verify_source_safely,
    is_static_only_source,
)
from .distributions_verify import verify_distribution, verify_log_prob
from .complex_verify import verify_fft, verify_view_as_complex, verify_view_as_real
from .named_tensor_verify import verify_align_to, verify_named_tensor_source, verify_refine_names
from .vmap_verify import verify_vmap

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
    "verify_distribution",
    "verify_log_prob",
    "verify_fft",
    "verify_view_as_real",
    "verify_view_as_complex",
    "verify_refine_names",
    "verify_align_to",
    "verify_named_tensor_source",
    "verify_vmap",
    "__version__",
]
