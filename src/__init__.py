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
    "__version__",
]
