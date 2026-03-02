#!/usr/bin/env python3
"""
Multi-tool baseline comparison framework for TensorGuard.

Compares TensorGuard against five baseline tools — jaxtyping (beartype),
PyTEA, TorchScript, mypy + torch stubs, and Pyright — on a curated set
of tensor-shape test cases.  For each case the script records which tools
would detect the error (or correctly accept valid code) and produces a
markdown feature-comparison table plus per-test-case result matrix.

Baseline verdicts are *analytical*: they are derived from each tool's
published capabilities rather than from running the tool directly (most
baselines have incompatible runtimes or require annotation).  TensorGuard
results are obtained by calling ``verify_model`` on every test case.
"""

from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


# ═══════════════════════════════════════════════════════════════════════════════
# 1. BaselineTool dataclass
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class BaselineTool:
    name: str
    description: str
    checks_shapes: bool
    checks_devices: bool
    checks_gradients: bool
    annotation_required: bool
    supports_symbolic_dims: bool
    supports_broadcasting: bool
    supports_dynamic_shapes: bool


# ═══════════════════════════════════════════════════════════════════════════════
# 2. Registered baseline tools
# ═══════════════════════════════════════════════════════════════════════════════

BASELINE_TOOLS: Dict[str, BaselineTool] = {
    "tensorguard": BaselineTool(
        name="TensorGuard",
        description=(
            "Static SMT-backed tensor shape verifier. Checks shapes, devices, "
            "gradients, and training phase without annotations. Supports "
            "symbolic dimensions, broadcasting, and dynamic shapes via Z3."
        ),
        checks_shapes=True,
        checks_devices=True,
        checks_gradients=True,
        annotation_required=False,
        supports_symbolic_dims=True,
        supports_broadcasting=True,
        supports_dynamic_shapes=True,
    ),
    "jaxtyping": BaselineTool(
        name="jaxtyping",
        description=(
            "Runtime shape checker via beartype + jaxtyping annotations. "
            "Requires explicit type annotations on every function. Provides "
            "limited named-dim support but no static analysis, no device or "
            "gradient checking."
        ),
        checks_shapes=True,
        checks_devices=False,
        checks_gradients=False,
        annotation_required=True,
        supports_symbolic_dims=False,  # named dims only, not truly symbolic
        supports_broadcasting=False,
        supports_dynamic_shapes=False,
    ),
    "pytea": BaselineTool(
        name="PyTEA",
        description=(
            "Static symbolic shape analyser for PyTorch (KAIST, ECOOP 2023). "
            "Detects single-operation shape mismatches but has limited operator "
            "coverage (~50 ops), no SMT backend, and is an unmaintained "
            "research prototype."
        ),
        checks_shapes=True,
        checks_devices=False,
        checks_gradients=False,
        annotation_required=False,
        supports_symbolic_dims=False,
        supports_broadcasting=True,
        supports_dynamic_shapes=False,
    ),
    "torchscript": BaselineTool(
        name="TorchScript",
        description=(
            "torch.jit.script traces shapes at compile time over the "
            "scriptable Python subset. Catches some shape errors at trace "
            "time but has no symbolic dimension support."
        ),
        checks_shapes=True,
        checks_devices=False,
        checks_gradients=False,
        annotation_required=False,
        supports_symbolic_dims=False,
        supports_broadcasting=False,
        supports_dynamic_shapes=False,
    ),
    "mypy": BaselineTool(
        name="mypy + torch stubs",
        description=(
            "Static type checker with PyTorch type stubs. Catches Python "
            "type errors but performs no shape analysis. Requires type "
            "annotations."
        ),
        checks_shapes=False,
        checks_devices=False,
        checks_gradients=False,
        annotation_required=True,
        supports_symbolic_dims=False,
        supports_broadcasting=False,
        supports_dynamic_shapes=False,
    ),
    "pyright": BaselineTool(
        name="Pyright",
        description=(
            "Fast static type checker with strong type inference. Better "
            "inference than mypy but still performs no tensor shape analysis."
        ),
        checks_shapes=False,
        checks_devices=False,
        checks_gradients=False,
        annotation_required=False,
        supports_symbolic_dims=False,
        supports_broadcasting=False,
        supports_dynamic_shapes=False,
    ),
}


# ═══════════════════════════════════════════════════════════════════════════════
# 3. TestCase definition & curated test suite
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class TestCase:
    name: str
    source: str
    expected_error: bool
    description: str = ""
    input_shapes: Optional[Dict[str, tuple]] = None


TEST_CASES: List[TestCase] = [
    # ── Error cases ──────────────────────────────────────────────────────────
    TestCase(
        "matmul_mismatch",
        """\
import torch, torch.nn as nn
class M(nn.Module):
    def __init__(self):
        super().__init__()
        self.w = nn.Parameter(torch.randn(5, 6))
    def forward(self, x):
        return x @ self.w
""",
        expected_error=True,
        description="Basic matmul dimension mismatch",
        input_shapes={"x": ("batch", 4)},
    ),
    TestCase(
        "broadcast_error",
        """\
import torch, torch.nn as nn
class M(nn.Module):
    def forward(self, x, y):
        return x + y
""",
        expected_error=True,
        description="Incompatible broadcast",
        input_shapes={"x": ("batch", 3, 4), "y": ("batch", 5, 4)},
    ),
    TestCase(
        "reshape_error",
        """\
import torch, torch.nn as nn
class M(nn.Module):
    def forward(self, x):
        return x.view(3, 3)
""",
        expected_error=True,
        description="Reshape element count mismatch (6 != 9)",
        input_shapes={"x": ("batch", 6)},
    ),
    TestCase(
        "conv_channel_mismatch",
        """\
import torch, torch.nn as nn
class M(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 16, 3)
    def forward(self, x):
        return self.conv(x)
""",
        expected_error=True,
        description="Conv2d expects 3 input channels but receives 1",
        input_shapes={"x": ("batch", 1, 32, 32)},
    ),
    TestCase(
        "linear_feature_mismatch",
        """\
import torch, torch.nn as nn
class M(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 5)
    def forward(self, x):
        return self.fc(x)
""",
        expected_error=True,
        description="Linear in_features=10 but input has 20",
        input_shapes={"x": ("batch", 20)},
    ),
    TestCase(
        "concat_dim_mismatch",
        """\
import torch, torch.nn as nn
class M(nn.Module):
    def forward(self, x, y):
        return torch.cat([x, y], dim=1)
""",
        expected_error=True,
        description="cat along dim=1 with mismatched dim-0",
        input_shapes={"x": (2, 3), "y": (4, 3)},
    ),
    TestCase(
        "wrong_pool_size",
        """\
import torch, torch.nn as nn
class M(nn.Module):
    def __init__(self):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d((7, 7))
        self.fc = nn.Linear(10, 5)
    def forward(self, x):
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)
""",
        expected_error=True,
        description="Pooled feature size does not match Linear in_features",
        input_shapes={"x": ("batch", 3, 32, 32)},
    ),
    TestCase(
        "transpose_then_matmul",
        """\
import torch, torch.nn as nn
class M(nn.Module):
    def forward(self, x):
        return x @ x
""",
        expected_error=True,
        description="Square matmul fails when input is non-square",
        input_shapes={"x": ("batch", 3, 5)},
    ),

    # ── Valid cases ──────────────────────────────────────────────────────────
    TestCase(
        "valid_resnet_block",
        """\
import torch, torch.nn as nn
class ResBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(64, 64, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
    def forward(self, x):
        identity = x
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return torch.relu(out + identity)
""",
        expected_error=False,
        description="Standard residual block — shapes should be consistent",
        input_shapes={"x": ("batch", 64, 16, 16)},
    ),
    TestCase(
        "valid_transformer_block",
        """\
import torch, torch.nn as nn
class TransformerBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=512, num_heads=8)
        self.ff = nn.Sequential(nn.Linear(512, 2048), nn.ReLU(), nn.Linear(2048, 512))
        self.norm1 = nn.LayerNorm(512)
        self.norm2 = nn.LayerNorm(512)
    def forward(self, x):
        a, _ = self.attn(x, x, x)
        x = self.norm1(x + a)
        x = self.norm2(x + self.ff(x))
        return x
""",
        expected_error=False,
        description="Standard transformer block",
        input_shapes={"x": ("seq", "batch", 512)},
    ),
    TestCase(
        "symbolic_batch",
        """\
import torch, torch.nn as nn
class M(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 5)
    def forward(self, x):
        return self.fc(x)
""",
        expected_error=False,
        description="Batch dim symbolic — should verify for all batch sizes",
        input_shapes={"x": ("batch", 10)},
    ),
    TestCase(
        "valid_mlp",
        """\
import torch, torch.nn as nn
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 10)
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)
""",
        expected_error=False,
        description="Simple two-layer MLP",
        input_shapes={"x": ("batch", 784)},
    ),
    TestCase(
        "valid_sequential",
        """\
import torch, torch.nn as nn
class M(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(100, 50),
            nn.ReLU(),
            nn.Linear(50, 10),
        )
    def forward(self, x):
        return self.net(x)
""",
        expected_error=False,
        description="nn.Sequential with matching layer sizes",
        input_shapes={"x": ("batch", 100)},
    ),
    TestCase(
        "valid_conv_bn_relu",
        """\
import torch, torch.nn as nn
class M(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 16, 3, padding=1)
        self.bn = nn.BatchNorm2d(16)
    def forward(self, x):
        return torch.relu(self.bn(self.conv(x)))
""",
        expected_error=False,
        description="Conv-BN-ReLU with correct channel alignment",
        input_shapes={"x": ("batch", 3, 32, 32)},
    ),
    TestCase(
        "valid_embedding_lookup",
        """\
import torch, torch.nn as nn
class M(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(1000, 128)
        self.fc = nn.Linear(128, 10)
    def forward(self, x):
        return self.fc(self.emb(x))
""",
        expected_error=False,
        description="Embedding lookup followed by linear",
        input_shapes={"x": ("batch", "seq")},
    ),
    TestCase(
        "double_linear_mismatch",
        """\
import torch, torch.nn as nn
class M(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 20)
        self.fc2 = nn.Linear(30, 5)
    def forward(self, x):
        return self.fc2(self.fc1(x))
""",
        expected_error=True,
        description="Chained linears with mismatched hidden dim (20 != 30)",
        input_shapes={"x": ("batch", 10)},
    ),
    TestCase(
        "valid_dropout_no_shape_change",
        """\
import torch, torch.nn as nn
class M(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 5)
        self.drop = nn.Dropout(0.5)
    def forward(self, x):
        return self.drop(self.fc(x))
""",
        expected_error=False,
        description="Dropout does not change tensor shapes",
        input_shapes={"x": ("batch", 10)},
    ),
]


# ═══════════════════════════════════════════════════════════════════════════════
# 4. Analytical baseline results per test case
# ═══════════════════════════════════════════════════════════════════════════════

BASELINE_RESULTS: Dict[str, Dict[str, Dict[str, Any]]] = {
    "matmul_mismatch": {
        "tensorguard": {"detects": True, "static": True, "annotations_needed": False},
        "jaxtyping":   {"detects": True, "static": False, "annotations_needed": True},
        "pytea":       {"detects": True, "static": True, "annotations_needed": False},
        "torchscript": {"detects": True, "static": False, "annotations_needed": False},
        "mypy":        {"detects": False, "static": True, "annotations_needed": True},
        "pyright":     {"detects": False, "static": True, "annotations_needed": False},
    },
    "broadcast_error": {
        "tensorguard": {"detects": True, "static": True, "annotations_needed": False},
        "jaxtyping":   {"detects": True, "static": False, "annotations_needed": True},
        "pytea":       {"detects": True, "static": True, "annotations_needed": False},
        "torchscript": {"detects": False, "static": False, "annotations_needed": False},
        "mypy":        {"detects": False, "static": True, "annotations_needed": True},
        "pyright":     {"detects": False, "static": True, "annotations_needed": False},
    },
    "reshape_error": {
        "tensorguard": {"detects": True, "static": True, "annotations_needed": False},
        "jaxtyping":   {"detects": True, "static": False, "annotations_needed": True},
        "pytea":       {"detects": True, "static": True, "annotations_needed": False},
        "torchscript": {"detects": True, "static": False, "annotations_needed": False},
        "mypy":        {"detects": False, "static": True, "annotations_needed": True},
        "pyright":     {"detects": False, "static": True, "annotations_needed": False},
    },
    "conv_channel_mismatch": {
        "tensorguard": {"detects": True, "static": True, "annotations_needed": False},
        "jaxtyping":   {"detects": True, "static": False, "annotations_needed": True},
        "pytea":       {"detects": True, "static": True, "annotations_needed": False},
        "torchscript": {"detects": True, "static": False, "annotations_needed": False},
        "mypy":        {"detects": False, "static": True, "annotations_needed": True},
        "pyright":     {"detects": False, "static": True, "annotations_needed": False},
    },
    "linear_feature_mismatch": {
        "tensorguard": {"detects": True, "static": True, "annotations_needed": False},
        "jaxtyping":   {"detects": True, "static": False, "annotations_needed": True},
        "pytea":       {"detects": True, "static": True, "annotations_needed": False},
        "torchscript": {"detects": True, "static": False, "annotations_needed": False},
        "mypy":        {"detects": False, "static": True, "annotations_needed": True},
        "pyright":     {"detects": False, "static": True, "annotations_needed": False},
    },
    "concat_dim_mismatch": {
        "tensorguard": {"detects": True, "static": True, "annotations_needed": False},
        "jaxtyping":   {"detects": True, "static": False, "annotations_needed": True},
        "pytea":       {"detects": True, "static": True, "annotations_needed": False},
        "torchscript": {"detects": False, "static": False, "annotations_needed": False},
        "mypy":        {"detects": False, "static": True, "annotations_needed": True},
        "pyright":     {"detects": False, "static": True, "annotations_needed": False},
    },
    "wrong_pool_size": {
        "tensorguard": {"detects": True, "static": True, "annotations_needed": False},
        "jaxtyping":   {"detects": True, "static": False, "annotations_needed": True},
        "pytea":       {"detects": False, "static": True, "annotations_needed": False},
        "torchscript": {"detects": False, "static": False, "annotations_needed": False},
        "mypy":        {"detects": False, "static": True, "annotations_needed": True},
        "pyright":     {"detects": False, "static": True, "annotations_needed": False},
    },
    "transpose_then_matmul": {
        "tensorguard": {"detects": True, "static": True, "annotations_needed": False},
        "jaxtyping":   {"detects": True, "static": False, "annotations_needed": True},
        "pytea":       {"detects": True, "static": True, "annotations_needed": False},
        "torchscript": {"detects": True, "static": False, "annotations_needed": False},
        "mypy":        {"detects": False, "static": True, "annotations_needed": True},
        "pyright":     {"detects": False, "static": True, "annotations_needed": False},
    },
    "double_linear_mismatch": {
        "tensorguard": {"detects": True, "static": True, "annotations_needed": False},
        "jaxtyping":   {"detects": True, "static": False, "annotations_needed": True},
        "pytea":       {"detects": True, "static": True, "annotations_needed": False},
        "torchscript": {"detects": True, "static": False, "annotations_needed": False},
        "mypy":        {"detects": False, "static": True, "annotations_needed": True},
        "pyright":     {"detects": False, "static": True, "annotations_needed": False},
    },
    # ── Valid cases: "detects" means *correctly accepts* (no false positive)
    "valid_resnet_block": {
        "tensorguard": {"detects": False, "static": True, "annotations_needed": False},
        "jaxtyping":   {"detects": False, "static": False, "annotations_needed": True},
        "pytea":       {"detects": False, "static": True, "annotations_needed": False},
        "torchscript": {"detects": False, "static": False, "annotations_needed": False},
        "mypy":        {"detects": False, "static": True, "annotations_needed": True},
        "pyright":     {"detects": False, "static": True, "annotations_needed": False},
    },
    "valid_transformer_block": {
        "tensorguard": {"detects": False, "static": True, "annotations_needed": False},
        "jaxtyping":   {"detects": False, "static": False, "annotations_needed": True},
        "pytea":       {"detects": False, "static": True, "annotations_needed": False},
        "torchscript": {"detects": False, "static": False, "annotations_needed": False},
        "mypy":        {"detects": False, "static": True, "annotations_needed": True},
        "pyright":     {"detects": False, "static": True, "annotations_needed": False},
    },
    "symbolic_batch": {
        "tensorguard": {"detects": False, "static": True, "annotations_needed": False},
        "jaxtyping":   {"detects": False, "static": False, "annotations_needed": True},
        "pytea":       {"detects": False, "static": True, "annotations_needed": False},
        "torchscript": {"detects": False, "static": False, "annotations_needed": False},
        "mypy":        {"detects": False, "static": True, "annotations_needed": True},
        "pyright":     {"detects": False, "static": True, "annotations_needed": False},
    },
    "valid_mlp": {
        "tensorguard": {"detects": False, "static": True, "annotations_needed": False},
        "jaxtyping":   {"detects": False, "static": False, "annotations_needed": True},
        "pytea":       {"detects": False, "static": True, "annotations_needed": False},
        "torchscript": {"detects": False, "static": False, "annotations_needed": False},
        "mypy":        {"detects": False, "static": True, "annotations_needed": True},
        "pyright":     {"detects": False, "static": True, "annotations_needed": False},
    },
    "valid_sequential": {
        "tensorguard": {"detects": False, "static": True, "annotations_needed": False},
        "jaxtyping":   {"detects": False, "static": False, "annotations_needed": True},
        "pytea":       {"detects": False, "static": True, "annotations_needed": False},
        "torchscript": {"detects": False, "static": False, "annotations_needed": False},
        "mypy":        {"detects": False, "static": True, "annotations_needed": True},
        "pyright":     {"detects": False, "static": True, "annotations_needed": False},
    },
    "valid_conv_bn_relu": {
        "tensorguard": {"detects": False, "static": True, "annotations_needed": False},
        "jaxtyping":   {"detects": False, "static": False, "annotations_needed": True},
        "pytea":       {"detects": False, "static": True, "annotations_needed": False},
        "torchscript": {"detects": False, "static": False, "annotations_needed": False},
        "mypy":        {"detects": False, "static": True, "annotations_needed": True},
        "pyright":     {"detects": False, "static": True, "annotations_needed": False},
    },
    "valid_embedding_lookup": {
        "tensorguard": {"detects": False, "static": True, "annotations_needed": False},
        "jaxtyping":   {"detects": False, "static": False, "annotations_needed": True},
        "pytea":       {"detects": False, "static": True, "annotations_needed": False},
        "torchscript": {"detects": False, "static": False, "annotations_needed": False},
        "mypy":        {"detects": False, "static": True, "annotations_needed": True},
        "pyright":     {"detects": False, "static": True, "annotations_needed": False},
    },
    "valid_dropout_no_shape_change": {
        "tensorguard": {"detects": False, "static": True, "annotations_needed": False},
        "jaxtyping":   {"detects": False, "static": False, "annotations_needed": True},
        "pytea":       {"detects": False, "static": True, "annotations_needed": False},
        "torchscript": {"detects": False, "static": False, "annotations_needed": False},
        "mypy":        {"detects": False, "static": True, "annotations_needed": True},
        "pyright":     {"detects": False, "static": True, "annotations_needed": False},
    },
}


# ═══════════════════════════════════════════════════════════════════════════════
# 5. Markdown feature-comparison table
# ═══════════════════════════════════════════════════════════════════════════════

def format_comparison_table() -> str:
    """Produce a markdown feature-comparison table across all registered tools."""
    rows = [
        ("Static analysis",           ["✓", "✗", "✓", "Partial", "✓", "✓"]),
        ("Shape checking",            ["✓", "✓*", "✓", "Partial", "✗", "✗"]),
        ("No annotations needed",     ["✓", "✗", "✓", "✓", "✗", "Partial"]),
        ("Symbolic dimensions",       ["✓", "Partial", "✗", "✗", "✗", "✗"]),
        ("Broadcasting verification", ["✓", "✗", "✓", "✗", "✗", "✗"]),
        ("SMT-backed",                ["✓", "✗", "✗", "✗", "✗", "✗"]),
        ("Device checking",           ["✓", "✗", "✗", "✗", "✗", "✗"]),
        ("Gradient checking",         ["✓", "✗", "✗", "✗", "✗", "✗"]),
        ("Dynamic shape support",     ["✓", "✗", "✗", "✗", "✗", "✗"]),
        ("Operator coverage (300+)",  ["✓", "N/A", "~50", "~200", "N/A", "N/A"]),
    ]
    tool_names = ["TensorGuard", "jaxtyping", "PyTEA", "TorchScript", "mypy", "Pyright"]
    header = "| Feature                    | " + " | ".join(f"{t:<11}" for t in tool_names) + " |"
    sep    = "|" + "----------------------------|" + "|".join("-" * 13 for _ in tool_names) + "|"
    lines = [header, sep]
    for label, vals in rows:
        line = f"| {label:<27}| " + " | ".join(f"{v:<11}" for v in vals) + " |"
        lines.append(line)
    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════════
# 6. Run TensorGuard on all test cases
# ═══════════════════════════════════════════════════════════════════════════════

def run_tensorguard_evaluation() -> Dict[str, Dict[str, Any]]:
    """Run ``verify_model`` on every test case and return per-case results."""
    from src.model_checker import verify_model

    results: Dict[str, Dict[str, Any]] = {}
    for tc in TEST_CASES:
        t0 = time.perf_counter()
        try:
            result = verify_model(source=tc.source, input_shapes=tc.input_shapes)
            elapsed_ms = (time.perf_counter() - t0) * 1000
            detected = (
                len(result.errors) > 0
                if hasattr(result, "errors") and result.errors
                else not result.safe
            )
            results[tc.name] = {
                "detects": detected,
                "time_ms": round(elapsed_ms, 2),
                "static": True,
                "annotations_needed": False,
                "safe": result.safe,
                "errors": list(result.errors) if hasattr(result, "errors") else [],
            }
        except Exception as exc:
            elapsed_ms = (time.perf_counter() - t0) * 1000
            results[tc.name] = {
                "detects": False,
                "time_ms": round(elapsed_ms, 2),
                "static": True,
                "annotations_needed": False,
                "safe": None,
                "errors": [str(exc)],
                "exception": True,
            }
    return results


# ═══════════════════════════════════════════════════════════════════════════════
# 7. Aggregate & write results
# ═══════════════════════════════════════════════════════════════════════════════

def compute_metrics(tg_results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """Compute TP/FP/TN/FN and precision/recall/F1 for TensorGuard."""
    tp = fp = tn = fn = 0
    for tc in TEST_CASES:
        detected = tg_results.get(tc.name, {}).get("detects", False)
        if tc.expected_error:
            if detected:
                tp += 1
            else:
                fn += 1
        else:
            if detected:
                fp += 1
            else:
                tn += 1
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {"tp": tp, "fp": fp, "tn": tn, "fn": fn,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4)}


def main() -> None:
    print("=" * 72)
    print("TensorGuard Baseline Comparison Framework")
    print("=" * 72)

    # Feature table
    table = format_comparison_table()
    print("\n" + table + "\n")

    # Run TensorGuard
    print("Running TensorGuard on all test cases …")
    tg_results = run_tensorguard_evaluation()

    metrics = compute_metrics(tg_results)
    print(f"\nTensorGuard metrics: {metrics}")

    # Assemble output
    output = {
        "tools": {k: asdict(v) for k, v in BASELINE_TOOLS.items()},
        "test_cases": [
            {"name": tc.name, "expected_error": tc.expected_error,
             "description": tc.description}
            for tc in TEST_CASES
        ],
        "baseline_results": BASELINE_RESULTS,
        "tensorguard_actual": tg_results,
        "tensorguard_metrics": metrics,
        "feature_comparison_table": table,
    }

    out_path = Path(__file__).resolve().parent / "baseline_comparison_actual_results.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults written to {out_path}")


if __name__ == "__main__":
    main()
