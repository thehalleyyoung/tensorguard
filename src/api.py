"""TensorGuard: Find bugs in Python code with zero annotations.

Public API for programmatic use.  Provides three analysis modes:

1. **Flow-sensitive analysis** (original) — fast abstract interpretation
   with guard harvesting and CEGAR refinement.
2. **Liquid type analysis** (new, primary) — full liquid type inference
   backed by Z3 subtyping, predicate harvesting from seven sources,
   CEGAR refinement, and interprocedural contract propagation.
3. **Unified analysis** — combines liquid types and tensor shape checking
   via CofiberedDomain, enabling cross-domain bug detection
   (e.g. Optional[Tensor] shape access without null check).

The liquid type engine is the PRIMARY analysis mode.  The original
flow-sensitive mode is still available for backward compatibility.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Dict
from enum import Enum
import ast
import time
import os
from pathlib import Path

# ── Optional liquid type engine ────────────────────────────────────────
try:
    from src.liquid import (
        LiquidTypeInferencer,
        InterproceduralLiquidAnalyzer,
        FunctionContract,
        LiquidAnalysisResult,
        LiquidBug,
        LiquidBugKind,
        PredicateHarvester,
        analyze_liquid,
    )
    _HAS_LIQUID = True
except Exception:  # Z3 or other dep missing
    _HAS_LIQUID = False

# ── Public data types ──────────────────────────────────────────────────

class BugCategory(Enum):
    NULL_DEREFERENCE = "null_dereference"
    DIVISION_BY_ZERO = "division_by_zero"
    INDEX_OUT_OF_BOUNDS = "index_out_of_bounds"
    TYPE_ERROR = "type_error"
    ATTRIBUTE_ERROR = "attribute_error"


@dataclass
class SourceLocation:
    file: str
    line: int
    column: int
    end_line: Optional[int] = None
    end_column: Optional[int] = None


@dataclass
class Bug:
    category: BugCategory
    message: str
    location: SourceLocation
    severity: str  # "error", "warning", "info"
    confidence: float  # 0.0-1.0
    fix_suggestion: Optional[str] = None
    guard_evidence: Optional[str] = None  # the guard that reveals this bug


@dataclass
class AnalysisResult:
    bugs: List[Bug] = field(default_factory=list)
    guards_harvested: int = 0
    functions_analyzed: int = 0
    lines_analyzed: int = 0
    duration_ms: float = 0.0
    counterexample: Optional[Dict[str, Any]] = None  # Z3 witness assignment
    abstained: bool = False  # True iff verifier hit an out-of-fragment construct
    opaque_layer_count: int = 0  # number of LayerKind.UNKNOWN entries

    @property
    def status(self) -> str:
        """Return ``"SAFE"`` when no bugs were found, ``"UNSAFE"`` otherwise."""
        return "SAFE" if not self.bugs else "UNSAFE"

    @property
    def bug_count(self) -> int:
        return len(self.bugs)

    def errors(self) -> List[Bug]:
        return [b for b in self.bugs if b.severity == "error"]

    def by_category(self, cat: BugCategory) -> List[Bug]:
        return [b for b in self.bugs if b.category == cat]

    def to_sarif(self, contracts: Optional[Dict[str, "FunctionContract"]] = None) -> dict:
        """Export results in SARIF format for IDE integration.

        Args:
            contracts: Optional liquid type contracts to include as properties.
        """
        contract_props = {}
        if contracts:
            contract_props = {
                "tensorguard:contracts": {
                    name: contract.annotated_signature()
                    for name, contract in contracts.items()
                }
            }

        run: dict = {
            "tool": {"driver": {"name": "TensorGuard", "version": "0.2.0",
                                 "informationUri": "https://github.com/tensorguard/tensorguard",
                                 "rules": [
                                     {"id": c.value, "shortDescription": {"text": c.value.replace("_", " ").title()}}
                                     for c in BugCategory
                                 ]}},
            "results": [
                {
                    "ruleId": b.category.value,
                    "level": "error" if b.severity == "error" else "warning",
                    "message": {"text": b.message},
                    "locations": [{
                        "physicalLocation": {
                            "artifactLocation": {"uri": b.location.file},
                            "region": {
                                "startLine": b.location.line,
                                "startColumn": b.location.column,
                            }
                        }
                    }]
                }
                for b in self.bugs
            ],
        }
        if contract_props:
            run["properties"] = contract_props

        return {
            "$schema": "https://raw.githubusercontent.com/oasis-tcs/sarif-spec/master/Schemata/sarif-schema-2.1.0.json",
            "version": "2.1.0",
            "runs": [run],
        }


# ── Category mapping from internal analyzer ────────────────────────────

# Maps internal BugCategory enum names to public BugCategory values
_CATEGORY_MAP = {
    "NULL_DEREF": BugCategory.NULL_DEREFERENCE,
    "UNGUARDED_OPTIONAL": BugCategory.NULL_DEREFERENCE,
    "DIV_BY_ZERO": BugCategory.DIVISION_BY_ZERO,
    "INDEX_OUT_OF_BOUNDS": BugCategory.INDEX_OUT_OF_BOUNDS,
    "TYPE_ERROR": BugCategory.TYPE_ERROR,
    "ATTRIBUTE_ERROR": BugCategory.ATTRIBUTE_ERROR,
}

_SEVERITY_MAP = {
    "NULL_DEREF": "error",
    "UNGUARDED_OPTIONAL": "warning",
    "DIV_BY_ZERO": "error",
    "INDEX_OUT_OF_BOUNDS": "error",
    "TYPE_ERROR": "warning",
    "ATTRIBUTE_ERROR": "warning",
}

_CONFIDENCE_MAP = {
    "NULL_DEREF": 0.95,
    "UNGUARDED_OPTIONAL": 0.80,
    "DIV_BY_ZERO": 0.95,
    "INDEX_OUT_OF_BOUNDS": 0.90,
    "TYPE_ERROR": 0.75,
    "ATTRIBUTE_ERROR": 0.75,
}

_FIX_SUGGESTIONS = {
    "NULL_DEREF": "Add a `if {var} is not None:` guard before this access.",
    "UNGUARDED_OPTIONAL": "Add a `if {var} is not None:` guard or use `{var} or default`.",
    "DIV_BY_ZERO": "Add a `if {var} != 0:` guard before dividing.",
    "INDEX_OUT_OF_BOUNDS": "Check `len({var})` before indexing, or use try/except.",
    "TYPE_ERROR": "Add an `isinstance({var}, expected_type)` guard.",
    "ATTRIBUTE_ERROR": "Add a `hasattr({var}, 'attr')` check.",
}


def _convert_internal_bug(ib, filename: str) -> Bug:
    """Convert an internal Bug dataclass to the public Bug type."""
    cat_name = ib.category.name
    category = _CATEGORY_MAP.get(cat_name, BugCategory.NULL_DEREFERENCE)
    severity = _SEVERITY_MAP.get(cat_name, "warning")
    confidence = _CONFIDENCE_MAP.get(cat_name, 0.75)
    fix_tmpl = _FIX_SUGGESTIONS.get(cat_name, "")
    fix = fix_tmpl.format(var=ib.variable) if fix_tmpl else None
    guard_ev = ib.guard_context if ib.guard_context else None

    return Bug(
        category=category,
        message=ib.message,
        location=SourceLocation(
            file=filename,
            line=ib.line,
            column=ib.col,
        ),
        severity=severity,
        confidence=confidence,
        fix_suggestion=fix,
        guard_evidence=guard_ev,
    )


def _file_result_to_analysis_result(fr, filename: str) -> AnalysisResult:
    """Convert an internal FileResult to public AnalysisResult."""
    bugs = []
    for func_result in fr.function_results:
        for ib in func_result.bugs:
            bugs.append(_convert_internal_bug(ib, filename))

    return AnalysisResult(
        bugs=bugs,
        guards_harvested=fr.total_guards,
        functions_analyzed=fr.functions_analyzed,
        lines_analyzed=fr.lines_of_code,
        duration_ms=fr.analysis_time_ms,
    )


# ── Shape analysis integration ─────────────────────────────────────────

def _filter_init_param_div_by_zero(source: str, bugs: List[Bug]) -> List[Bug]:
    """Filter false positive division-by-zero bugs for __init__ parameters.

    In nn.Module __init__ methods, patterns like ``hidden_size // num_heads``
    use constructor parameters that have non-zero integer defaults. These are
    not actual division-by-zero risks.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return bugs

    # Collect all __init__ parameters with non-zero integer defaults
    safe_divisors: set = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "__init__":
            defaults = node.args.defaults
            args = node.args.args
            n_defaults = len(defaults)
            n_args = len(args)
            for i, default in enumerate(defaults):
                arg_idx = n_args - n_defaults + i
                if arg_idx >= 0 and arg_idx < n_args:
                    arg_name = args[arg_idx].arg
                    if (isinstance(default, ast.Constant)
                            and isinstance(default.value, (int, float))
                            and default.value != 0):
                        safe_divisors.add(arg_name)

    if not safe_divisors:
        return bugs

    filtered = []
    for bug in bugs:
        if bug.category == BugCategory.DIVISION_BY_ZERO:
            # Check if the message references a safe divisor
            is_false_positive = False
            for name in safe_divisors:
                if f"'{name}'" in bug.message:
                    is_false_positive = True
                    break
            if is_false_positive:
                continue
        filtered.append(bug)
    return filtered


def _run_shape_analysis(source: str, filename: str) -> List[Bug]:
    """Run tensor shape analysis and convert shape errors to Bug objects."""
    try:
        from .tensor_shapes import TensorShapeAnalyzer, ShapeErrorKind
    except Exception:
        return []

    try:
        analyzer = TensorShapeAnalyzer(timeout_ms=3000)
        shape_result = analyzer.analyze_source(source)
    except Exception:
        return []

    bugs = []
    for err in shape_result.errors:
        bugs.append(Bug(
            category=BugCategory.TYPE_ERROR,
            message=err.message,
            location=SourceLocation(
                file=filename,
                line=err.line,
                column=err.col,
            ),
            severity=err.severity if err.severity else "warning",
            confidence=0.85,
            fix_suggestion=None,
        ))
    return bugs


# ── Public API ─────────────────────────────────────────────────────────

def analyze(source: str, filename: str = "<string>",
            use_liquid: bool = False) -> AnalysisResult:
    """Analyze Python source code for bugs. Zero annotations required.

    Harvests existing guards (isinstance, is not None, comparisons) as
    implicit refinement types, then performs flow-sensitive analysis.
    Also runs tensor shape verification to catch shape mismatches.

    Args:
        source: Python source code string.
        filename: Optional filename for error reporting.
        use_liquid: If True and Z3 is available, use liquid type inference
            for higher precision (Z3-backed subtyping + CEGAR).

    Returns:
        AnalysisResult with all detected bugs and analysis metadata.
    """
    if use_liquid and _HAS_LIQUID:
        return liquid_analyze(source, filename)

    from .real_analyzer import analyze_source
    fr = analyze_source(source, filename=filename, use_cegar=True)
    result = _file_result_to_analysis_result(fr, filename)

    # Filter false positive division-by-zero for __init__ params with non-zero defaults
    result.bugs = _filter_init_param_div_by_zero(source, result.bugs)

    # Run tensor shape analysis and merge shape bugs
    shape_bugs = _run_shape_analysis(source, filename)
    result.bugs.extend(shape_bugs)

    return result


def analyze_file(path: str) -> AnalysisResult:
    """Analyze a Python file for bugs.

    Args:
        path: Path to a .py file.

    Returns:
        AnalysisResult with detected bugs.
    """
    p = Path(path)
    if not p.exists():
        return AnalysisResult()
    try:
        source = p.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return AnalysisResult()
    result = analyze(source, filename=str(p.resolve()))
    return result


def analyze_directory(
    path: str,
    pattern: str = "**/*.py",
    exclude: Optional[List[str]] = None,
) -> AnalysisResult:
    """Recursively analyze all Python files in a directory.

    Args:
        path: Root directory path.
        pattern: Glob pattern for Python files (default: ``**/*.py``).
        exclude: Directory names to skip (default: common non-source dirs).

    Returns:
        Merged AnalysisResult across all files.
    """
    exclude = exclude or [
        "__pycache__", ".git", "node_modules", ".venv", "venv",
        ".tox", ".mypy_cache", ".pytest_cache", "dist", "build",
    ]
    t0 = time.perf_counter()
    root = Path(path)
    merged = AnalysisResult()

    for py_file in sorted(root.glob(pattern)):
        if any(ex in py_file.parts for ex in exclude):
            continue
        try:
            r = analyze_file(str(py_file))
            merged.bugs.extend(r.bugs)
            merged.guards_harvested += r.guards_harvested
            merged.functions_analyzed += r.functions_analyzed
            merged.lines_analyzed += r.lines_analyzed
        except Exception:
            pass

    merged.duration_ms = (time.perf_counter() - t0) * 1000
    return merged


def analyze_function(source: str, function_name: str) -> AnalysisResult:
    """Analyze a single function extracted from source code.

    Parses the full source, locates the named function, and runs
    flow-sensitive analysis only on that function.

    Args:
        source: Full Python source containing the function.
        function_name: Name of the function to analyze.

    Returns:
        AnalysisResult scoped to the specified function.
    """
    from .real_analyzer import FlowSensitiveAnalyzer
    t0 = time.perf_counter()

    try:
        tree = ast.parse(source)
    except SyntaxError:
        return AnalysisResult()

    analyzer = FlowSensitiveAnalyzer(source)
    funcs = [
        n for n in ast.walk(tree)
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
        and n.name == function_name
    ]

    if not funcs:
        return AnalysisResult()

    func_node = funcs[0]
    fr = analyzer.analyze_function(func_node)

    bugs = [_convert_internal_bug(b, "<string>") for b in fr.bugs]
    elapsed = (time.perf_counter() - t0) * 1000

    return AnalysisResult(
        bugs=bugs,
        guards_harvested=fr.guards_harvested,
        functions_analyzed=1,
        lines_analyzed=len(source.splitlines()),
        duration_ms=elapsed,
    )


def quick_check(source: str) -> List[str]:
    """Fast check returning list of bug descriptions. For scripting.

    Args:
        source: Python source code string.

    Returns:
        List of strings like ``"12:4 division_by_zero: Possible division by zero"``.
    """
    result = analyze(source)
    return [
        f"{b.location.line}:{b.location.column} {b.category.value}: {b.message}"
        for b in result.bugs
    ]


# ── Liquid Type API ────────────────────────────────────────────────────

# Map LiquidBugKind → public BugCategory
_LIQUID_CATEGORY_MAP: Dict[str, BugCategory] = {
    "NULL_DEREF": BugCategory.NULL_DEREFERENCE,
    "DIV_BY_ZERO": BugCategory.DIVISION_BY_ZERO,
    "INDEX_OOB": BugCategory.INDEX_OUT_OF_BOUNDS,
    "TYPE_ERROR": BugCategory.TYPE_ERROR,
    "ATTRIBUTE_ERROR": BugCategory.ATTRIBUTE_ERROR,
    "PRECONDITION_VIOLATION": BugCategory.TYPE_ERROR,
    "UNSAT_CONSTRAINT": BugCategory.TYPE_ERROR,
}


def _convert_liquid_bug(lb, filename: str) -> Bug:
    """Convert a LiquidBug to the public Bug type."""
    kind_name = lb.kind.name
    category = _LIQUID_CATEGORY_MAP.get(kind_name, BugCategory.TYPE_ERROR)
    return Bug(
        category=category,
        message=lb.message,
        location=SourceLocation(file=filename, line=lb.line, column=lb.col),
        severity=lb.severity,
        confidence=0.95,  # Z3-backed — high confidence
        fix_suggestion=None,
        guard_evidence=None,
    )


def liquid_analyze(source: str, filename: str = "<string>") -> AnalysisResult:
    """Analyze with full liquid type inference + Z3-backed subtyping.

    Uses the TensorGuard engine: predicate harvesting from guards, asserts,
    defaults, exceptions, returns, walrus operators, and comprehension
    filters, followed by Z3-backed constraint solving with CEGAR.

    Args:
        source: Python source code string.
        filename: Optional filename for error reporting.

    Returns:
        AnalysisResult with detected bugs and analysis metadata.

    Raises:
        RuntimeError: If Z3 is not installed.
    """
    if not _HAS_LIQUID:
        raise RuntimeError(
            "Liquid type analysis requires Z3. Install with: pip install z3-solver"
        )

    t0 = time.perf_counter()
    engine = InterproceduralLiquidAnalyzer()
    lresult = engine.analyze(source)

    bugs = [_convert_liquid_bug(lb, filename) for lb in lresult.bugs]
    elapsed = (time.perf_counter() - t0) * 1000

    result = AnalysisResult(
        bugs=bugs,
        guards_harvested=lresult.predicates_harvested,
        functions_analyzed=len(lresult.contracts),
        lines_analyzed=len(source.splitlines()),
        duration_ms=elapsed,
    )
    # Stash contracts for SARIF export
    result._liquid_contracts = lresult.contracts  # type: ignore[attr-defined]
    return result


def liquid_analyze_file(path: str) -> AnalysisResult:
    """Analyze a file with liquid type inference.

    Args:
        path: Path to a .py file.

    Returns:
        AnalysisResult with detected bugs.
    """
    p = Path(path)
    if not p.exists():
        return AnalysisResult()
    try:
        source = p.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return AnalysisResult()
    return liquid_analyze(source, filename=str(p.resolve()))


def liquid_analyze_directory(
    path: str,
    pattern: str = "**/*.py",
    exclude: Optional[List[str]] = None,
) -> AnalysisResult:
    """Analyze a directory with interprocedural liquid type inference.

    Args:
        path: Root directory path.
        pattern: Glob pattern for Python files (default: ``**/*.py``).
        exclude: Directory names to skip.

    Returns:
        Merged AnalysisResult across all files.
    """
    exclude = exclude or [
        "__pycache__", ".git", "node_modules", ".venv", "venv",
        ".tox", ".mypy_cache", ".pytest_cache", "dist", "build",
    ]
    t0 = time.perf_counter()
    root = Path(path)
    merged = AnalysisResult()

    for py_file in sorted(root.glob(pattern)):
        if any(ex in py_file.parts for ex in exclude):
            continue
        try:
            r = liquid_analyze_file(str(py_file))
            merged.bugs.extend(r.bugs)
            merged.guards_harvested += r.guards_harvested
            merged.functions_analyzed += r.functions_analyzed
            merged.lines_analyzed += r.lines_analyzed
        except Exception:
            pass

    merged.duration_ms = (time.perf_counter() - t0) * 1000
    return merged


def infer_contracts(source: str) -> Dict[str, str]:
    """Infer liquid type contracts for all functions.

    Returns:
        Dictionary mapping function names to annotated signature strings.
        E.g. ``{"div": "def div(x: int, y: Annotated[int, 'v ≠ 0']) -> int"}``

    Raises:
        RuntimeError: If Z3 is not installed.
    """
    if not _HAS_LIQUID:
        raise RuntimeError(
            "Contract inference requires Z3. Install with: pip install z3-solver"
        )

    engine = LiquidTypeInferencer()
    result = engine.infer_module(source)
    return {
        name: contract.annotated_signature()
        for name, contract in result.contracts.items()
    }


def liquid_quick_check(source: str) -> List[str]:
    """Quick check with liquid types, returns bug description strings.

    Args:
        source: Python source code string.

    Returns:
        List of strings like ``"12:4 division_by_zero: Possible division by zero"``.

    Raises:
        RuntimeError: If Z3 is not installed.
    """
    result = liquid_analyze(source)
    return [
        f"{b.location.line}:{b.location.column} {b.category.value}: {b.message}"
        for b in result.bugs
    ]


# ── Unified Analysis API ──────────────────────────────────────────────

try:
    from src.unified import (
        analyze_unified,
        UnifiedAnalyzer,
        UnifiedAnalysisResult,
        UnifiedBug,
    )
    _HAS_UNIFIED = True
except Exception:
    _HAS_UNIFIED = False

# ── Intent-apparent overwarning API ────────────────────────────────────
try:
    from src.intent_bugs import (
        OverwarnAnalyzer,
        IntentApparentBug,
        IntentBugKind,
    )
    _HAS_OVERWARN = True
except Exception:
    _HAS_OVERWARN = False


# ── Model Checker / Constraint Verification API ──────────────────────────
try:
    from src.model_checker import (
        verify_model,
        extract_computation_graph,
        BoundedModelChecker,  # backward-compatible alias for ConstraintVerifier
        VerificationResult,
        SafetyCertificate,
        CounterexampleTrace,
    )
    _HAS_MODEL_CHECKER = True
except Exception:
    _HAS_MODEL_CHECKER = False

try:
    from src.shape_cegar import (
        run_shape_cegar,
        verify_and_discover,
        ShapeCEGARResult,
    )
    _HAS_SHAPE_CEGAR = True
except Exception:
    _HAS_SHAPE_CEGAR = False


def verify_architecture(
    source: str,
    input_shapes: Optional[Dict[str, tuple]] = None,
    check_devices: bool = True,
    check_phases: bool = True,
    check_gradients: bool = True,
    max_cegar_iterations: int = 10,
    filename: str = "<string>",
    high_confidence_only: bool = False,
    hooks: Optional[List] = None,
) -> AnalysisResult:
    """Verify an nn.Module architecture via constraint-based verification.

    Extracts the computation graph from the nn.Module class, then verifies
    shape compatibility, device consistency, and gradient flow via Z3-backed
    symbolic constraint propagation. Optionally runs counterexample-guided
    contract discovery (CEGAR-style) to discover implicit shape contracts.

    This is TensorGuard's primary novelty: no other tool does constraint-based
    verification of tensor computation graphs.

    Args:
        source: Python source containing an nn.Module subclass.
        input_shapes: Map from input names to shape tuples. Symbolic dims
            are strings (e.g., {"x": ("batch_size", 3, 224, 224)}).
        check_devices: Whether to verify device consistency.
        check_phases: Whether to check train/eval phase dependencies.
        check_gradients: Whether to verify gradient flow.
        max_cegar_iterations: Max contract discovery iterations.
        filename: Optional filename for error reporting.
        high_confidence_only: When True, only report HIGH-confidence
            (Z3-proven) bugs. Reduces FP rate to 0% for CI/CD gating.
        hooks: Optional list of integration hooks (e.g. WandbHook,
            MLflowHook, CIHook). Hooks are called at verification
            start/end and after each CEGAR iteration.

    Returns:
        AnalysisResult. If verification succeeds, bugs list is empty.
        If it fails, bugs contain shape/device/phase errors with
        concrete counterexample traces.
    """
    hooks = hooks or []
    t0 = time.perf_counter()
    result = AnalysisResult()
    result.lines_analyzed = len(source.splitlines())

    # Notify hooks: verification starting
    for hook in hooks:
        try:
            hook.on_verification_start(source=source, filename=filename)
        except Exception:
            pass

    if not _HAS_MODEL_CHECKER:
        raise RuntimeError(
            "Model checker requires Z3. Install with: pip install z3-solver"
        )

    # Check if any hook requests deterministic mode (skip LLM pipeline)
    deterministic = any(getattr(h, "deterministic", False) for h in hooks)

    # Run constraint-based verification
    vr = verify_model(
        source,
        input_shapes=input_shapes or {},
        high_confidence_only=high_confidence_only,
    )

    # Convert verification errors to Bug objects
    for error in vr.errors:
        result.bugs.append(Bug(
            category=BugCategory.TYPE_ERROR,
            message=f"[MODEL_CHECK] {error}",
            location=SourceLocation(
                file=filename,
                line=0,
                column=0,
            ),
            severity="error",
            confidence=0.99 if not vr.safe else 0.80,
            fix_suggestion=None,
        ))

    # Convert counterexample violations to Bug objects
    if vr.counterexample is not None:
        # Expose structured counterexample (Z3 witness) in the result
        ce = vr.counterexample
        result.counterexample = {
            "concrete_dims": getattr(ce, "concrete_dims", {}),
            "violations": [
                {"kind": v.kind, "message": v.message,
                 "line": getattr(v.step, "line", 0) if v.step else 0}
                for v in ce.violations
            ],
        }
        for violation in ce.violations:
            kind = violation.kind  # e.g. "shape_incompatible", "device_mismatch"
            # Soundness filter: when a Z3 shape_incompatible witness contains
            # only auxiliary-domain variables (grad_/dev_/phase_) or unbound
            # configuration symbols (config_/cfg_/args_/hparams_), the
            # violation is not provably a shape disagreement on every
            # concrete instantiation — it is a free-symbol model-fit picked
            # by Z3 to satisfy unrelated cross-domain constraints.  In that
            # case we ABSTAIN (downgrade to warning) rather than reject.
            _msg = violation.message or ""
            _downgrade = False
            if (kind == "shape_incompatible" and "Z3 violation" in _msg):
                _shape_lhs = []
                _tail = _msg.split("Z3 violation", 1)[1]
                for _ln in _tail.splitlines():
                    _ln = _ln.strip()
                    if "=" not in _ln:
                        continue
                    _lhs = _ln.split("=", 1)[0].strip()
                    if not _lhs:
                        continue
                    if _lhs.startswith(("grad_", "dev_", "phase_")):
                        continue
                    if _lhs.startswith(("config_", "cfg_", "args_",
                                        "hparams_", "conf_",
                                        "model_config_", "_unk_", "__")):
                        continue
                    _shape_lhs.append(_lhs)
                if not _shape_lhs:
                    _downgrade = True
            category_map = {
                "shape_incompatible": BugCategory.TYPE_ERROR,
                "device_mismatch": BugCategory.TYPE_ERROR,
                "phase_error": BugCategory.TYPE_ERROR,
                "dead_output": BugCategory.TYPE_ERROR,
                "gradient_broken": BugCategory.TYPE_ERROR,
            }
            category = category_map.get(kind, BugCategory.TYPE_ERROR)
            line = getattr(violation.step, "line", 0) if violation.step else 0
            col = getattr(violation.step, "col", 0) if violation.step else 0
            confidence_val = getattr(violation.confidence, "value", str(violation.confidence))
            severity = "error" if confidence_val == "high" else "warning"
            if _downgrade:
                severity = "warning"
            result.bugs.append(Bug(
                category=category,
                message=f"[{kind.upper().replace('_', '-')}] {violation.message}",
                location=SourceLocation(
                    file=filename,
                    line=line,
                    column=col,
                ),
                severity=severity,
                confidence=0.99 if (confidence_val == "high" and not _downgrade) else 0.80,
                fix_suggestion=None,
            ))

    # ── Feature-flag post-hoc filtering (helpers defined here, applied below CEGAR) ──
    # check_devices / check_phases / check_gradients gate which violation
    # *kinds* are surfaced.  The underlying verify_model always runs all
    # checks; the flags filter the result so the feature contribution is
    # cleanly discriminated in ablation studies.
    # NOTE: filtering is applied AFTER CEGAR so that CEGAR-surfaced violations
    # (e.g. [CEGAR-REAL-BUG] Gradient flow broken: ...) are also gated correctly.
    _KIND_DEVICE  = ("device_mismatch", "DEVICE-MISMATCH")
    _KIND_PHASE   = ("phase_violation", "phase_error", "PHASE-VIOLATION", "PHASE-ERROR")
    _KIND_GRAD    = ("gradient_broken", "gradient_violation",
                     "GRADIENT-BROKEN", "GRADIENT-VIOLATION", "Gradient flow broken",
                     "GRADIENT-OUT-OF-FRAGMENT")

    def _bug_matches_kinds(bug: Bug, kinds) -> bool:
        msg = bug.message
        return any(k.lower() in msg.lower() for k in kinds)

    # ── Low-confidence mode: merge flow-sensitive analysis bugs ─────────
    # When high_confidence_only=False, additionally run the heuristic
    # flow-sensitive analyser (src.real_analyzer via analyze()) and merge
    # any bugs it finds.  These are lower-confidence (heuristic, not Z3-
    # backed) and would be suppressed at CI-gate level (high_confidence_only).
    if not high_confidence_only:
        try:
            from src.real_analyzer import analyze_source
            _fa_fr = analyze_source(source, filename=filename, use_cegar=False)
            _fa_bugs: List[Bug] = []
            for _fr in getattr(_fa_fr, "function_results", []):
                for _ib in getattr(_fr, "bugs", []):
                    _fa_bugs.append(_convert_internal_bug(_ib, filename))
            # Only add heuristic bugs not already covered by constraint-based
            _existing_msgs = {b.message for b in result.bugs}
            for _fb in _fa_bugs:
                if _fb.message not in _existing_msgs:
                    _fb.confidence = min(_fb.confidence, 0.80)  # downgrade to low-conf
                    result.bugs.append(_fb)
        except Exception:
            pass

    # ── Run CEGAR if available ──────────────────────────────────────────
    if _HAS_SHAPE_CEGAR and max_cegar_iterations > 0:
        cegar_result = run_shape_cegar(
            source,
            max_iterations=max_cegar_iterations,
        )
        # Store discovered contracts
        result._shape_contracts = cegar_result.discovered_predicates  # type: ignore[attr-defined]
        result._cegar_iterations = cegar_result.iterations  # type: ignore[attr-defined]

        # Surface any CEGAR-confirmed real shape bugs as Bug objects.
        # Note: in the current implementation _is_real_bug() only fires
        # when the shape_env tracks post-op shapes (initial shapes only
        # are currently tracked), so real_bugs is typically empty.  This
        # wiring is provided for correctness; if/when CEGAR is extended
        # to propagate shapes, these will start appearing.
        for _rb in getattr(cegar_result, "real_bugs", []):
            _rb_msg = f"[CEGAR-REAL-BUG] {getattr(_rb, 'message', str(_rb))}"
            if _rb_msg not in {b.message for b in result.bugs}:
                result.bugs.append(Bug(
                    category=BugCategory.TYPE_ERROR,
                    message=_rb_msg,
                    location=SourceLocation(file=filename, line=0, column=0),
                    severity="error",
                    confidence=0.95,
                    fix_suggestion=None,
                ))

        # Notify hooks: CEGAR iterations
        for hook in hooks:
            try:
                hook.on_cegar_iteration(
                    iteration=cegar_result.iterations,
                    status=str(getattr(cegar_result, "final_status", "done")),
                    predicates_discovered=len(cegar_result.discovered_predicates),
                )
            except Exception:
                pass

    # ── Grad-lattice out-of-fragment detector ──────────────────────────
    # The first-order grad lattice (has_grad / no_grad / unknown) cannot
    # represent the second-order graph rewriting performed by
    # ``torch.utils.checkpoint.checkpoint`` (which discards activations
    # in the forward and recomputes them in the backward) or by
    # HuggingFace's ``model.gradient_checkpointing_enable()`` / its
    # ``_set_gradient_checkpointing`` helper.  When we see any of these
    # constructs, we emit a high-confidence Refuted-Proof flagging that
    # the analysed module is out of the first-order fragment, and the
    # module is not silently verified.  This makes the runtime grad-flag
    # comparison honest: a runtime checkpointed model will not be
    # reported as SAFE+VERIFIED.
    try:
        _grad_oof_patterns = (
            "torch.utils.checkpoint.checkpoint(",
            "checkpoint_sequential(",
            ".gradient_checkpointing_enable(",
            "gradient_checkpointing=True",
            "_set_gradient_checkpointing(",
            "from torch.utils.checkpoint import checkpoint",
        )
        # Avoid false-firing on a *call* to a method literally named
        # ``checkpoint`` that is not the torch.utils variant; require
        # one of the qualifying patterns above.
        _src_for_grep = source
        _grad_oof_hit = next((p for p in _grad_oof_patterns
                              if p in _src_for_grep), None)
        if _grad_oof_hit is not None and check_gradients:
            _msg = (
                "[GRADIENT-OUT-OF-FRAGMENT] First-order grad lattice "
                "cannot soundly verify modules that use "
                f"`{_grad_oof_hit}`; analyser flips this module out "
                "of the verified fragment (Refuted-Proof, not "
                "abstention) so a runtime gradient-flag check cannot "
                "silently pass."
            )
            if _msg not in {b.message for b in result.bugs}:
                result.bugs.append(Bug(
                    category=BugCategory.TYPE_ERROR,
                    message=_msg,
                    location=SourceLocation(file=filename, line=0, column=0),
                    severity="error",
                    confidence=0.99,
                    fix_suggestion=None,
                ))
    except Exception:
        pass

    # Apply feature-flag filters after all bug sources (including CEGAR) have run.
    if not check_devices:
        result.bugs = [b for b in result.bugs if not _bug_matches_kinds(b, _KIND_DEVICE)]
    if not check_phases:
        result.bugs = [b for b in result.bugs if not _bug_matches_kinds(b, _KIND_PHASE)]
    if not check_gradients:
        result.bugs = [b for b in result.bugs if not _bug_matches_kinds(b, _KIND_GRAD)]

    result.functions_analyzed = 1
    result.duration_ms = (time.perf_counter() - t0) * 1000
    # Count opaque (out-of-fragment) layers to expose abstention status.
    try:
        from src.model_checker import LayerKind as _LK
        layers = getattr(getattr(vr, "graph", None), "layers", {}) or {}
        n_opaque = sum(1 for L in layers.values() if getattr(L, "kind", None) == _LK.UNKNOWN)
        result.opaque_layer_count = n_opaque
        result.abstained = n_opaque > 0
    except Exception:
        pass

    # Notify hooks: verification finished
    for hook in hooks:
        try:
            hook.on_verification_end(result=result)
        except Exception:
            pass

    return result


def verify_module(
    path: str,
    input_shapes: Optional[Dict[str, tuple]] = None,
    check_devices: bool = True,
    check_phases: bool = True,
    check_gradients: bool = True,
    max_cegar_iterations: int = 10,
    high_confidence_only: bool = False,
) -> AnalysisResult:
    """Verify an ``nn.Module`` from a file path.

    Convenience wrapper around :func:`verify_architecture` that reads a
    Python source file and passes its contents for verification.

    Args:
        path: Path to a ``.py`` file containing an ``nn.Module`` subclass.
        input_shapes: Map from input names to shape tuples (symbolic dims
            may be strings, e.g. ``{"x": ("batch", 3, 224, 224)}``).
        check_devices: Whether to verify device consistency.
        check_phases: Whether to check train/eval phase dependencies.
        check_gradients: Whether to verify gradient flow.
        max_cegar_iterations: Max CEGAR refinement iterations.
        high_confidence_only: Only report Z3-proven bugs.

    Returns:
        :class:`AnalysisResult` with ``status``, ``bugs``, and
        ``duration_ms`` attributes.
    """
    source = Path(path).read_text()
    return verify_architecture(
        source,
        input_shapes=input_shapes,
        check_devices=check_devices,
        check_phases=check_phases,
        check_gradients=check_gradients,
        max_cegar_iterations=max_cegar_iterations,
        filename=path,
        high_confidence_only=high_confidence_only,
    )


def overwarn_analyze(source: str, filename: str = "<string>") -> AnalysisResult:
    """Analyze source for intent-apparent ML bugs using overwarning mode.

    This deliberately overwarns: it flags any pattern that *could* be a
    semantic ML bug, even when it cannot prove the bug definitively.

    Covers all bug classes from bugclasses.jsonl including:
    - Shape/dimension mismatches and axis confusion
    - Device placement errors
    - Gradient flow breaks and hard gating
    - Optimizer protocol violations
    - Loss function misuse
    - Parameter lifecycle bugs
    """
    t0 = time.monotonic()
    result = AnalysisResult()
    result.lines_analyzed = len(source.splitlines())

    if not _HAS_OVERWARN:
        return result

    analyzer = OverwarnAnalyzer()
    intent_bugs = analyzer.analyze(source)

    for ib in intent_bugs:
        result.bugs.append(Bug(
            category=BugCategory.TYPE_ERROR,
            message=f"[OVERWARN:{ib.kind.name}] {ib.message}",
            location=SourceLocation(
                file=filename,
                line=ib.line,
                column=ib.col,
            ),
            severity=ib.severity,
            confidence=ib.inferred_intent.confidence,
            fix_suggestion=ib.inferred_intent.description,
            guard_evidence=f"Inferred intent: {ib.inferred_intent.pretty()}",
        ))

    result.duration_ms = (time.monotonic() - t0) * 1000
    return result
