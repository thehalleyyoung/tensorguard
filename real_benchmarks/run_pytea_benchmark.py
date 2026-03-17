#!/usr/bin/env python3
"""
PyTEA Benchmark Runner for TensorGuard.

Loads PyTEA test cases and synthetic model verification targets, runs
TensorGuard on each, and reports precision/recall metrics with timing.

Usage:
    cd tensorguard && PYTHONPATH=. python3 real_benchmarks/run_pytea_benchmark.py
"""

from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Ensure project root is importable
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.loaders.pytea_loader import PyTEALoader, PyTEATestCase
from src.tensor_shapes import (
    TensorShape,
    ShapeDim,
    ShapeError,
    ShapeErrorKind,
    analyze_shapes,
    compute_broadcast_shape,
    compute_matmul_shape,
    compute_reshape_shape,
)
from src.domains.dynamic_shapes import infer_neg_one_dim
from src.domains.broadcast_analysis import (
    check_broadcast_compatible,
    check_matmul_broadcast_compatible,
)
from src.smt.einsum_theory import (
    parse_einsum,
    infer_einsum_shape,
    check_einsum_compatible,
)

# Optional: model_checker for nn.Module verification
try:
    from src.model_checker import verify_model, VerificationResult
    HAS_MODEL_CHECKER = True
except ImportError:
    HAS_MODEL_CHECKER = False

# Optional: api-level analysis
try:
    from src.api import analyze as api_analyze
    HAS_API = True
except ImportError:
    HAS_API = False


# ═══════════════════════════════════════════════════════════════════════════
# Result data structures
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class TestResult:
    """Result of running TensorGuard on a single test case."""
    name: str
    category: str
    has_known_bug: bool
    tensorguard_found_bug: bool
    errors_found: List[Dict[str, Any]] = field(default_factory=list)
    verification_time_ms: float = 0.0
    # Classification
    true_positive: bool = False   # Known bug, found by TG
    false_positive: bool = False  # No known bug, TG reports bug
    true_negative: bool = False   # No known bug, TG says safe
    false_negative: bool = False  # Known bug, TG says safe
    notes: str = ""


@dataclass
class BenchmarkSummary:
    """Aggregate metrics across all test cases."""
    total_tests: int = 0
    true_positives: int = 0
    false_positives: int = 0
    true_negatives: int = 0
    false_negatives: int = 0
    total_time_ms: float = 0.0
    mean_time_ms: float = 0.0

    @property
    def precision(self) -> float:
        denom = self.true_positives + self.false_positives
        return self.true_positives / denom if denom > 0 else 1.0

    @property
    def recall(self) -> float:
        denom = self.true_positives + self.false_negatives
        return self.true_positives / denom if denom > 0 else 1.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) > 0 else 0.0


# ═══════════════════════════════════════════════════════════════════════════
# Synthetic model definitions (10 models: 5 correct + 5 buggy)
# ═══════════════════════════════════════════════════════════════════════════

SYNTHETIC_MODELS: List[Dict[str, Any]] = [
    # --- 1. Simple MLP ---
    {
        "name": "mlp_correct",
        "category": "synthetic_mlp",
        "has_bug": False,
        "source": '''
import torch.nn as nn

class SimpleMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 10)

    def forward(self, x):
        x = x.view(-1, 784)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x
''',
        "input_shapes": {"x": ("batch", 1, 28, 28)},
    },
    {
        "name": "mlp_bug_wrong_reshape",
        "category": "synthetic_mlp",
        "has_bug": True,
        "bug_desc": "Reshape to 512 but fc1 expects 784",
        "source": '''
import torch.nn as nn

class SimpleMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 10)

    def forward(self, x):
        x = x.view(-1, 512)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x
''',
        "input_shapes": {"x": ("batch", 1, 28, 28)},
    },
    # --- 2. CNN ---
    {
        "name": "cnn_correct",
        "category": "synthetic_cnn",
        "has_bug": False,
        "source": '''
import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x
''',
        "input_shapes": {"x": ("batch", 1, 28, 28)},
    },
    {
        "name": "cnn_bug_wrong_linear",
        "category": "synthetic_cnn",
        "has_bug": True,
        "bug_desc": "fc1 expects 1024 but conv output flattened is 9216",
        "source": '''
import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.fc1 = nn.Linear(1024, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x
''',
        "input_shapes": {"x": ("batch", 1, 28, 28)},
    },
    # --- 3. RNN/LSTM ---
    {
        "name": "lstm_correct",
        "category": "synthetic_rnn",
        "has_bug": False,
        "source": '''
import torch.nn as nn

class LSTMClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(10000, 128)
        self.lstm = nn.LSTM(128, 256, batch_first=True)
        self.fc = nn.Linear(256, 5)

    def forward(self, x):
        x = self.embedding(x)
        output, (h_n, c_n) = self.lstm(x)
        x = h_n[-1]
        x = self.fc(x)
        return x
''',
        "input_shapes": {"x": ("batch", "seq_len")},
    },
    {
        "name": "lstm_bug_wrong_fc",
        "category": "synthetic_rnn",
        "has_bug": True,
        "bug_desc": "fc expects 128 but LSTM hidden is 256",
        "source": '''
import torch.nn as nn

class LSTMClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(10000, 128)
        self.lstm = nn.LSTM(128, 256, batch_first=True)
        self.fc = nn.Linear(128, 5)

    def forward(self, x):
        x = self.embedding(x)
        output, (h_n, c_n) = self.lstm(x)
        x = h_n[-1]
        x = self.fc(x)
        return x
''',
        "input_shapes": {"x": ("batch", "seq_len")},
    },
    # --- 4. Transformer Encoder ---
    {
        "name": "transformer_correct",
        "category": "synthetic_transformer",
        "has_bug": False,
        "source": '''
import torch
import torch.nn as nn

class TransformerClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(30000, 512)
        encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=6)
        self.fc = nn.Linear(512, 10)

    def forward(self, x):
        x = self.embedding(x)
        x = self.encoder(x)
        x = x.mean(dim=1)
        x = self.fc(x)
        return x
''',
        "input_shapes": {"x": ("batch", "seq_len")},
    },
    {
        "name": "transformer_bug_wrong_head",
        "category": "synthetic_transformer",
        "has_bug": True,
        "bug_desc": "d_model=512 not divisible by nhead=7",
        "source": '''
import torch
import torch.nn as nn

class TransformerClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(30000, 512)
        encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=7)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=6)
        self.fc = nn.Linear(512, 10)

    def forward(self, x):
        x = self.embedding(x)
        x = self.encoder(x)
        x = x.mean(dim=1)
        x = self.fc(x)
        return x
''',
        "input_shapes": {"x": ("batch", "seq_len")},
    },
    # --- 5. Broadcast + Einsum model ---
    {
        "name": "attention_correct",
        "category": "synthetic_attention",
        "has_bug": False,
        "source": '''
import torch
import torch.nn as nn

class MultiHeadAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.W_q = nn.Linear(512, 512)
        self.W_k = nn.Linear(512, 512)
        self.W_v = nn.Linear(512, 512)
        self.W_o = nn.Linear(512, 512)

    def forward(self, x):
        q = self.W_q(x)
        k = self.W_k(x)
        v = self.W_v(x)
        q = q.view(q.size(0), q.size(1), 8, 64).transpose(1, 2)
        k = k.view(k.size(0), k.size(1), 8, 64).transpose(1, 2)
        v = v.view(v.size(0), v.size(1), 8, 64).transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-2, -1))
        attn = torch.softmax(scores, dim=-1)
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(out.size(0), -1, 512)
        return self.W_o(out)
''',
        "input_shapes": {"x": ("batch", "seq_len", 512)},
    },
    {
        "name": "attention_bug_wrong_transpose",
        "category": "synthetic_attention",
        "has_bug": True,
        "bug_desc": "Missing transpose on k, matmul inner dims mismatch",
        "source": '''
import torch
import torch.nn as nn

class MultiHeadAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.W_q = nn.Linear(512, 512)
        self.W_k = nn.Linear(512, 512)
        self.W_v = nn.Linear(512, 512)
        self.W_o = nn.Linear(512, 512)

    def forward(self, x):
        q = self.W_q(x)
        k = self.W_k(x)
        v = self.W_v(x)
        q = q.view(q.size(0), q.size(1), 8, 64).transpose(1, 2)
        k = k.view(k.size(0), k.size(1), 8, 64).transpose(1, 2)
        v = v.view(v.size(0), v.size(1), 8, 64).transpose(1, 2)
        scores = torch.matmul(q, k)
        attn = torch.softmax(scores, dim=-1)
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(out.size(0), -1, 512)
        return self.W_o(out)
''',
        "input_shapes": {"x": ("batch", "seq_len", 512)},
    },
]


# ═══════════════════════════════════════════════════════════════════════════
# Analysis runners
# ═══════════════════════════════════════════════════════════════════════════

def run_shape_analysis(source: str) -> Tuple[bool, List[Dict[str, Any]], float]:
    """Run TensorGuard shape analysis on source code.

    Returns (found_bug, errors_list, time_ms).
    """
    t0 = time.monotonic()
    result = analyze_shapes(source)
    elapsed_ms = (time.monotonic() - t0) * 1000

    errors = []
    for err in result.errors:
        errors.append({
            "kind": err.kind.name,
            "line": err.line,
            "message": err.message,
            "function": err.function,
        })

    return len(errors) > 0, errors, elapsed_ms


def run_model_verification(
    source: str, input_shapes: Optional[Dict[str, tuple]] = None,
) -> Tuple[bool, List[Dict[str, Any]], float]:
    """Run TensorGuard model verification on nn.Module source.

    Returns (found_bug, errors_list, time_ms).
    """
    if not HAS_MODEL_CHECKER:
        return False, [], 0.0

    t0 = time.monotonic()
    try:
        result = verify_model(
            source, input_shapes=input_shapes, high_confidence_only=True,
        )
        elapsed_ms = (time.monotonic() - t0) * 1000

        errors = []
        if not result.safe:
            if result.counterexample:
                for v in result.counterexample.violations:
                    errors.append({
                        "kind": v.kind if hasattr(v, "kind") else "shape_error",
                        "message": str(v),
                    })
            for e in result.errors:
                errors.append({"kind": "error", "message": str(e)})

        return len(errors) > 0, errors, elapsed_ms
    except Exception as exc:
        elapsed_ms = (time.monotonic() - t0) * 1000
        return False, [{"kind": "analysis_error", "message": str(exc)[:200]}], elapsed_ms


def run_api_analysis(source: str) -> Tuple[bool, List[Dict[str, Any]], float]:
    """Run TensorGuard API analysis (liquid types + flow analysis).

    Returns (found_bug, errors_list, time_ms).
    """
    if not HAS_API:
        return False, [], 0.0

    t0 = time.monotonic()
    try:
        result = api_analyze(source)
        elapsed_ms = (time.monotonic() - t0) * 1000

        errors = []
        for bug in result.bugs:
            errors.append({
                "kind": bug.category.value,
                "line": bug.location.line,
                "message": bug.message,
                "severity": bug.severity,
            })

        return len(errors) > 0, errors, elapsed_ms
    except Exception as exc:
        elapsed_ms = (time.monotonic() - t0) * 1000
        return False, [{"kind": "analysis_error", "message": str(exc)[:200]}], elapsed_ms


def run_dynamic_shape_checks(source: str) -> List[Dict[str, Any]]:
    """Run dynamic shape inference checks on reshape/view with -1."""
    import ast as _ast

    issues: List[Dict[str, Any]] = []
    try:
        tree = _ast.parse(source)
    except SyntaxError:
        return issues

    for node in _ast.walk(tree):
        if not isinstance(node, _ast.Call):
            continue
        # Look for .view(...) or .reshape(...) calls
        if isinstance(node.func, _ast.Attribute):
            if node.func.attr in ("view", "reshape"):
                args = []
                for arg in node.args:
                    if isinstance(arg, _ast.Constant):
                        args.append(arg.value)
                    elif isinstance(arg, _ast.UnaryOp) and isinstance(arg.op, _ast.USub):
                        if isinstance(arg.operand, _ast.Constant):
                            args.append(-arg.operand.value)

                neg_count = sum(1 for a in args if a == -1)
                if neg_count > 1:
                    issues.append({
                        "kind": "DYNAMIC_SHAPE",
                        "line": node.lineno,
                        "message": f"Multiple -1 in {node.func.attr}(): {args}",
                    })

    return issues


def run_einsum_checks(source: str) -> List[Dict[str, Any]]:
    """Check einsum operations in source code for dimension issues."""
    import ast as _ast

    issues: List[Dict[str, Any]] = []
    try:
        tree = _ast.parse(source)
    except SyntaxError:
        return issues

    for node in _ast.walk(tree):
        if not isinstance(node, _ast.Call):
            continue
        func = node.func
        func_name = ""
        if isinstance(func, _ast.Attribute):
            func_name = func.attr
        elif isinstance(func, _ast.Name):
            func_name = func.id

        if func_name == "einsum" and node.args:
            eq_node = node.args[0]
            if isinstance(eq_node, _ast.Constant) and isinstance(eq_node.value, str):
                equation = eq_node.value
                try:
                    parsed = parse_einsum(equation)
                    n_inputs = len(parsed.input_subscripts)
                    n_args = len(node.args) - 1  # First arg is equation
                    if n_args != n_inputs:
                        issues.append({
                            "kind": "EINSUM_ARITY",
                            "line": node.lineno,
                            "message": (
                                f"einsum '{equation}' expects {n_inputs} "
                                f"tensors, got {n_args}"
                            ),
                        })
                except Exception:
                    pass

    return issues


def run_broadcast_checks(source: str) -> List[Dict[str, Any]]:
    """Check for obvious broadcast incompatibilities in source."""
    import ast as _ast

    issues: List[Dict[str, Any]] = []
    # This is a lightweight heuristic check for now; the full check
    # is done by the shape analyzer and model checker
    return issues


# ═══════════════════════════════════════════════════════════════════════════
# Test case runners
# ═══════════════════════════════════════════════════════════════════════════

def run_pytea_test(case: PyTEATestCase) -> TestResult:
    """Run all available analyses on a PyTEA test case.

    For PyTEA tests we focus on shape-related errors only, since these are
    real training scripts where the API analysis would flag many non-shape
    issues (null deref in argparse, etc.) that aren't relevant.
    """
    all_errors: List[Dict[str, Any]] = []
    total_time = 0.0

    # 1. Shape analysis (always available — high signal for shape bugs)
    found_bug_shape, shape_errors, shape_time = run_shape_analysis(case.source)
    all_errors.extend(shape_errors)
    total_time += shape_time

    # 2. Model verification (for files with nn.Module classes)
    found_bug_model = False
    if case.module_classes:
        for cls_name in case.module_classes:
            input_shapes = case.input_shapes.get(cls_name)
            fb, errs, t = run_model_verification(case.source, input_shapes)
            found_bug_model = found_bug_model or fb
            all_errors.extend(errs)
            total_time += t

    # 3. Dynamic shape checks
    dyn_issues = run_dynamic_shape_checks(case.source)
    all_errors.extend(dyn_issues)

    # 4. Einsum checks
    einsum_issues = run_einsum_checks(case.source)
    all_errors.extend(einsum_issues)

    # For PyTEA tests, only count shape/model verification findings —
    # skip generic API analysis to avoid false positives from non-shape bugs
    found_bug = found_bug_shape or found_bug_model or bool(dyn_issues) or bool(einsum_issues)

    result = TestResult(
        name=case.name,
        category=case.category,
        has_known_bug=case.has_known_bug,
        tensorguard_found_bug=found_bug,
        errors_found=all_errors,
        verification_time_ms=total_time,
    )

    # Classify
    if case.has_known_bug and found_bug:
        result.true_positive = True
    elif case.has_known_bug and not found_bug:
        result.false_negative = True
        result.notes = f"Missed bug: {case.bug_description or 'unknown'}"
    elif not case.has_known_bug and found_bug:
        result.false_positive = True
    else:
        result.true_negative = True

    return result


def run_synthetic_test(model_def: Dict[str, Any]) -> TestResult:
    """Run TensorGuard on a synthetic model definition."""
    source = model_def["source"]
    has_bug = model_def["has_bug"]
    input_shapes = model_def.get("input_shapes")
    all_errors: List[Dict[str, Any]] = []
    total_time = 0.0

    # Shape analysis
    found_bug_shape, shape_errors, shape_time = run_shape_analysis(source)
    all_errors.extend(shape_errors)
    total_time += shape_time

    # Model verification
    found_bug_model, model_errors, model_time = run_model_verification(
        source, input_shapes
    )
    all_errors.extend(model_errors)
    total_time += model_time

    # API analysis
    found_bug_api, api_errors, api_time = run_api_analysis(source)
    all_errors.extend(api_errors)
    total_time += api_time

    # Dynamic shape checks
    dyn_issues = run_dynamic_shape_checks(source)
    all_errors.extend(dyn_issues)

    # Einsum checks
    einsum_issues = run_einsum_checks(source)
    all_errors.extend(einsum_issues)

    found_bug = found_bug_shape or found_bug_model or found_bug_api or bool(dyn_issues) or bool(einsum_issues)

    result = TestResult(
        name=model_def["name"],
        category=model_def["category"],
        has_known_bug=has_bug,
        tensorguard_found_bug=found_bug,
        errors_found=all_errors,
        verification_time_ms=total_time,
    )

    if has_bug and found_bug:
        result.true_positive = True
    elif has_bug and not found_bug:
        result.false_negative = True
        result.notes = f"Missed: {model_def.get('bug_desc', '')}"
    elif not has_bug and found_bug:
        result.false_positive = True
    else:
        result.true_negative = True

    return result


# ═══════════════════════════════════════════════════════════════════════════
# New-feature unit tests (run inline to validate improvements)
# ═══════════════════════════════════════════════════════════════════════════

def run_feature_tests() -> List[Dict[str, Any]]:
    """Validate dynamic shapes, einsum, and broadcasting features."""
    results: List[Dict[str, Any]] = []

    # --- Dynamic shape inference ---
    shape_4d = TensorShape.from_tuple((2, 3, 4, 5))
    out = infer_neg_one_dim(shape_4d, (-1, 60))
    results.append({
        "test": "dynamic_reshape_neg1",
        "passed": out is not None and out.dims[0].value == 2 and out.dims[1].value == 60,
        "detail": out.pretty() if out else "None",
    })

    out2 = infer_neg_one_dim(shape_4d, (2, -1))
    results.append({
        "test": "dynamic_reshape_neg1_second",
        "passed": out2 is not None and out2.dims[1].value == 60,
        "detail": out2.pretty() if out2 else "None",
    })

    out3 = infer_neg_one_dim(shape_4d, (6, -1, 5))
    results.append({
        "test": "dynamic_reshape_3d",
        "passed": out3 is not None and out3.dims[1].value == 4,
        "detail": out3.pretty() if out3 else "None",
    })

    out_bad = infer_neg_one_dim(shape_4d, (-1, -1))
    results.append({
        "test": "dynamic_reshape_multi_neg1",
        "passed": out_bad is None,
        "detail": "Correctly rejected" if out_bad is None else str(out_bad),
    })

    # --- Einsum ---
    parsed = parse_einsum("bij,bjk->bik")
    results.append({
        "test": "einsum_parse_basic",
        "passed": (
            parsed.input_subscripts == ["bij", "bjk"]
            and parsed.output_subscripts == "bik"
            and "j" in parsed.contraction_chars
        ),
        "detail": f"inputs={parsed.input_subscripts}, out={parsed.output_subscripts}",
    })

    sa = TensorShape.from_tuple((4, 8, 16))
    sb = TensorShape.from_tuple((4, 16, 32))
    out_einsum = infer_einsum_shape("bij,bjk->bik", [sa, sb])
    results.append({
        "test": "einsum_shape_inference",
        "passed": (
            out_einsum is not None
            and out_einsum.ndim == 3
            and out_einsum.dims[0].value == 4
            and out_einsum.dims[1].value == 8
            and out_einsum.dims[2].value == 32
        ),
        "detail": out_einsum.pretty() if out_einsum else "None",
    })

    err_msg = check_einsum_compatible("ij,jk->ik", [
        TensorShape.from_tuple((3, 4)),
        TensorShape.from_tuple((5, 6)),
    ])
    results.append({
        "test": "einsum_dimension_mismatch",
        "passed": err_msg is not None and "mismatch" in err_msg.lower(),
        "detail": err_msg or "None",
    })

    # --- Broadcasting ---
    s1 = TensorShape.from_tuple((4, 1, 3))
    s2 = TensorShape.from_tuple((5, 3))
    bc = check_broadcast_compatible(s1, s2)
    results.append({
        "test": "broadcast_compatible",
        "passed": (
            bc.compatible
            and bc.output_shape is not None
            and bc.output_shape.dims[0].value == 4
            and bc.output_shape.dims[1].value == 5
            and bc.output_shape.dims[2].value == 3
        ),
        "detail": bc.output_shape.pretty() if bc.output_shape else "None",
    })

    s3 = TensorShape.from_tuple((4, 3))
    s4 = TensorShape.from_tuple((5, 4))
    bc_fail = check_broadcast_compatible(s3, s4)
    results.append({
        "test": "broadcast_incompatible",
        "passed": not bc_fail.compatible,
        "detail": bc_fail.error_message or "None",
    })

    # Matmul broadcast
    s5 = TensorShape.from_tuple((2, 3, 4))
    s6 = TensorShape.from_tuple((2, 4, 5))
    mc = check_matmul_broadcast_compatible(s5, s6)
    results.append({
        "test": "matmul_broadcast_batched",
        "passed": (
            mc.compatible
            and mc.output_shape is not None
            and mc.output_shape.dims == (ShapeDim(2), ShapeDim(3), ShapeDim(5))
        ),
        "detail": mc.output_shape.pretty() if mc.output_shape else "None",
    })

    s7 = TensorShape.from_tuple((3, 4))
    s8 = TensorShape.from_tuple((5, 6))
    mc_fail = check_matmul_broadcast_compatible(s7, s8)
    results.append({
        "test": "matmul_inner_dim_mismatch",
        "passed": not mc_fail.compatible,
        "detail": mc_fail.error_message or "None",
    })

    # Z3 constraint tests
    try:
        import z3
        from src.domains.dynamic_shapes import encode_reshape_constraint_z3
        from src.smt.einsum_theory import encode_einsum_constraints_z3
        from src.domains.broadcast_analysis import encode_broadcast_constraints_z3

        # Reshape Z3 constraint
        a, b, c = z3.Ints("a b c")
        x, y = z3.Ints("x y")
        constraint = encode_reshape_constraint_z3([a, b, c], [x, y], neg_one_idx=1)
        results.append({
            "test": "z3_reshape_constraint",
            "passed": constraint is not None,
            "detail": "Z3 constraint generated",
        })

        # Einsum Z3 constraint
        i_vars = [[z3.Int(f"a{j}") for j in range(3)],
                   [z3.Int(f"b{j}") for j in range(3)]]
        o_vars = [z3.Int(f"o{j}") for j in range(3)]
        econst = encode_einsum_constraints_z3("bij,bjk->bik", i_vars, o_vars)
        results.append({
            "test": "z3_einsum_constraint",
            "passed": econst is not None,
            "detail": "Z3 einsum constraint generated",
        })

        # Broadcast Z3 constraint
        av = [z3.Int("ba0"), z3.Int("ba1")]
        bv = [z3.Int("bb0"), z3.Int("bb1")]
        ov = [z3.Int("bo0"), z3.Int("bo1")]
        bconst = encode_broadcast_constraints_z3(av, bv, ov)
        results.append({
            "test": "z3_broadcast_constraint",
            "passed": bconst is not None,
            "detail": "Z3 broadcast constraint generated",
        })

    except ImportError:
        results.append({
            "test": "z3_constraints",
            "passed": False,
            "detail": "Z3 not available",
        })

    return results


# ═══════════════════════════════════════════════════════════════════════════
# Main benchmark orchestrator
# ═══════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 72)
    print("TensorGuard — Real Benchmark Suite")
    print("=" * 72)

    all_results: List[TestResult] = []

    # --- Phase 1: Feature validation ---
    print("\n[Phase 1] Validating new features (dynamic shapes, einsum, broadcasting)")
    feature_results = run_feature_tests()
    passed = sum(1 for r in feature_results if r["passed"])
    total = len(feature_results)
    print(f"  Feature tests: {passed}/{total} passed")
    for r in feature_results:
        status = "✓" if r["passed"] else "✗"
        print(f"    {status} {r['test']}: {r['detail']}")

    # --- Phase 2: PyTEA benchmark tests ---
    print(f"\n[Phase 2] Loading PyTEA test cases")
    pytea_dir = PROJECT_ROOT / "real_benchmarks" / "data" / "pytea_tests"
    if pytea_dir.exists():
        loader = PyTEALoader(str(pytea_dir))
        cases = loader.load_all()
        print(f"  Loaded {len(cases)} PyTEA test files")
        print(f"  Categories: {sorted(set(c.category for c in cases))}")
        print(f"  Files with nn.Module: {sum(1 for c in cases if c.module_classes)}")
        print(f"  Files with known bugs: {sum(1 for c in cases if c.has_known_bug)}")

        for case in cases:
            result = run_pytea_test(case)
            all_results.append(result)
            status = "BUG" if result.tensorguard_found_bug else "OK "
            marker = ""
            if result.true_positive:
                marker = " [TP]"
            elif result.false_positive:
                marker = " [FP]"
            elif result.false_negative:
                marker = " [FN]"
            elif result.true_negative:
                marker = " [TN]"
            print(f"    {status} {result.name} ({result.verification_time_ms:.1f}ms){marker}")
    else:
        print(f"  PyTEA directory not found: {pytea_dir}")

    # --- Phase 3: Synthetic model tests ---
    print(f"\n[Phase 3] Running {len(SYNTHETIC_MODELS)} synthetic model tests")
    for model_def in SYNTHETIC_MODELS:
        result = run_synthetic_test(model_def)
        all_results.append(result)
        status = "BUG" if result.tensorguard_found_bug else "OK "
        marker = ""
        if result.true_positive:
            marker = " [TP]"
        elif result.false_positive:
            marker = " [FP]"
        elif result.false_negative:
            marker = " [FN]"
        elif result.true_negative:
            marker = " [TN]"
        expected = "bug" if model_def["has_bug"] else "safe"
        print(f"    {status} {model_def['name']} (expected: {expected}) ({result.verification_time_ms:.1f}ms){marker}")

    # --- Compute summary ---
    summary = BenchmarkSummary(
        total_tests=len(all_results),
        true_positives=sum(1 for r in all_results if r.true_positive),
        false_positives=sum(1 for r in all_results if r.false_positive),
        true_negatives=sum(1 for r in all_results if r.true_negative),
        false_negatives=sum(1 for r in all_results if r.false_negative),
        total_time_ms=sum(r.verification_time_ms for r in all_results),
    )
    summary.mean_time_ms = summary.total_time_ms / max(1, summary.total_tests)

    # --- Print summary ---
    print("\n" + "=" * 72)
    print("SUMMARY")
    print("=" * 72)
    print(f"  Total tests:      {summary.total_tests}")
    print(f"  True Positives:   {summary.true_positives}")
    print(f"  False Positives:  {summary.false_positives}")
    print(f"  True Negatives:   {summary.true_negatives}")
    print(f"  False Negatives:  {summary.false_negatives}")
    print(f"  Precision:        {summary.precision:.2%}")
    print(f"  Recall:           {summary.recall:.2%}")
    print(f"  F1 Score:         {summary.f1:.2%}")
    print(f"  Total time:       {summary.total_time_ms:.1f}ms")
    print(f"  Mean time/test:   {summary.mean_time_ms:.1f}ms")

    # Per-category breakdown
    categories = sorted(set(r.category for r in all_results))
    if len(categories) > 1:
        print(f"\n  Per-category breakdown:")
        for cat in categories:
            cat_results = [r for r in all_results if r.category == cat]
            tp = sum(1 for r in cat_results if r.true_positive)
            fp = sum(1 for r in cat_results if r.false_positive)
            tn = sum(1 for r in cat_results if r.true_negative)
            fn = sum(1 for r in cat_results if r.false_negative)
            n = len(cat_results)
            avg_ms = sum(r.verification_time_ms for r in cat_results) / max(1, n)
            print(f"    {cat:30s}  n={n:2d}  TP={tp} FP={fp} TN={tn} FN={fn}  avg={avg_ms:.1f}ms")

    # --- Comparison notes ---
    print(f"\n  Comparison baselines:")
    print(f"    PyTEA:       Catches shape errors in a subset of these benchmarks")
    print(f"                 via symbolic tracing (different approach from TensorGuard)")
    print(f"    mypy/pyright: Cannot catch tensor shape errors (no shape type support)")
    print(f"    TensorGuard: Static verification via refinement types + Z3")

    # --- Write JSON output ---
    output = {
        "summary": {
            "total_tests": summary.total_tests,
            "true_positives": summary.true_positives,
            "false_positives": summary.false_positives,
            "true_negatives": summary.true_negatives,
            "false_negatives": summary.false_negatives,
            "precision": round(summary.precision, 4),
            "recall": round(summary.recall, 4),
            "f1_score": round(summary.f1, 4),
            "total_time_ms": round(summary.total_time_ms, 2),
            "mean_time_ms": round(summary.mean_time_ms, 2),
        },
        "feature_tests": feature_results,
        "per_test_results": [
            {
                "name": r.name,
                "category": r.category,
                "has_known_bug": r.has_known_bug,
                "tensorguard_found_bug": r.tensorguard_found_bug,
                "true_positive": r.true_positive,
                "false_positive": r.false_positive,
                "true_negative": r.true_negative,
                "false_negative": r.false_negative,
                "errors_found": r.errors_found[:10],  # Cap per-test errors
                "verification_time_ms": round(r.verification_time_ms, 2),
                "notes": r.notes,
            }
            for r in all_results
        ],
        "comparison": {
            "pytea": "Symbolic execution based; catches subset of shape errors",
            "mypy_pyright": "No tensor shape support; 0 shape bugs detected",
            "tensorguard": "Refinement types + Z3; static verification with CEGAR",
        },
    }

    output_path = PROJECT_ROOT / "real_benchmarks" / "benchmark_results.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Results written to: {output_path}")
    print("=" * 72)

    return 0


if __name__ == "__main__":
    sys.exit(main())
