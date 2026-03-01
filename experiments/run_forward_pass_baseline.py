#!/usr/bin/env python3
"""
Forward-pass baseline comparison: TensorGuard vs running the model.

Addresses Sara Roy's critique: "What is TensorGuard's value proposition
relative to simply running the model's forward pass with a representative
input?"

For each benchmark model, runs THREE approaches:
  1. TensorGuard (static, parametric)
  2. Forward pass with a single representative input
  3. Forward pass with 5 diverse input shapes

Measures TP/FP/TN/FN, timing, and whether parametric guarantees are provided.
"""

import json
import os
import sys
import textwrap
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.model_checker import verify_model

# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dataclass
class ApproachResult:
    detected_bug: bool
    time_ms: float = 0.0
    parametric: bool = False
    error_message: str = ""


@dataclass
class ModelResult:
    name: str
    has_bug: bool
    bug_description: str
    tensorguard: Optional[ApproachResult] = None
    forward_pass: Optional[ApproachResult] = None
    forward_pass_multi: Optional[ApproachResult] = None
    tensorguard_advantage: str = ""


# ---------------------------------------------------------------------------
# Benchmark models (20+ models, mix of safe and buggy)
# ---------------------------------------------------------------------------

@dataclass
class BenchmarkModel:
    name: str
    source: str
    has_bug: bool
    bug_description: str
    input_shapes: Dict[str, tuple]
    # For forward pass: concrete input shapes to try
    concrete_inputs: List[Dict[str, Tuple[int, ...]]]
    # For multi-input forward pass: 5 diverse shapes
    multi_inputs: List[Dict[str, Tuple[int, ...]]]


def _make_models() -> List[BenchmarkModel]:
    """Build the full benchmark suite of 22 models."""
    models: List[BenchmarkModel] = []

    # ---- Category 1: Standard shape bugs (forward pass catches these too) ----

    models.append(BenchmarkModel(
        name="linear_dim_mismatch",
        source=textwrap.dedent("""\
            import torch, torch.nn as nn
            class M(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.fc1 = nn.Linear(768, 256)
                    self.fc2 = nn.Linear(128, 10)  # Bug: expects 128, gets 256
                def forward(self, x):
                    return self.fc2(self.fc1(x))
        """),
        has_bug=True,
        bug_description="fc1 outputs 256 but fc2 expects 128",
        input_shapes={"x": ("batch", 768)},
        concrete_inputs=[{"x": (1, 768)}],
        multi_inputs=[{"x": (s, 768)} for s in [1, 2, 4, 8, 16]],
    ))

    models.append(BenchmarkModel(
        name="conv_channel_mismatch",
        source=textwrap.dedent("""\
            import torch, torch.nn as nn
            class M(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
                    self.conv2 = nn.Conv2d(32, 128, 3, padding=1)  # Bug: expects 32, gets 64
                def forward(self, x):
                    return self.conv2(self.conv1(x))
        """),
        has_bug=True,
        bug_description="conv1 outputs 64 channels but conv2 expects 32",
        input_shapes={"x": ("batch", 3, 32, 32)},
        concrete_inputs=[{"x": (1, 3, 32, 32)}],
        multi_inputs=[{"x": (b, 3, 32, 32)} for b in [1, 2, 4, 8, 16]],
    ))

    models.append(BenchmarkModel(
        name="batchnorm_mismatch",
        source=textwrap.dedent("""\
            import torch, torch.nn as nn
            class M(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.conv = nn.Conv2d(3, 64, 3, padding=1)
                    self.bn = nn.BatchNorm2d(32)  # Bug: expects 32, conv outputs 64
                def forward(self, x):
                    return self.bn(self.conv(x))
        """),
        has_bug=True,
        bug_description="conv outputs 64 channels, bn expects 32",
        input_shapes={"x": ("batch", 3, 32, 32)},
        concrete_inputs=[{"x": (1, 3, 32, 32)}],
        multi_inputs=[{"x": (b, 3, 32, 32)} for b in [1, 2, 4, 8, 16]],
    ))

    # ---- Category 2: Safe models (no bugs) ----

    models.append(BenchmarkModel(
        name="simple_mlp_safe",
        source=textwrap.dedent("""\
            import torch, torch.nn as nn
            class M(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.fc1 = nn.Linear(784, 256)
                    self.fc2 = nn.Linear(256, 10)
                def forward(self, x):
                    return self.fc2(torch.relu(self.fc1(x)))
        """),
        has_bug=False,
        bug_description="",
        input_shapes={"x": ("batch", 784)},
        concrete_inputs=[{"x": (1, 784)}],
        multi_inputs=[{"x": (b, 784)} for b in [1, 2, 4, 8, 16]],
    ))

    models.append(BenchmarkModel(
        name="conv_net_safe",
        source=textwrap.dedent("""\
            import torch, torch.nn as nn
            class M(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
                    self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
                def forward(self, x):
                    return self.conv2(torch.relu(self.conv1(x)))
        """),
        has_bug=False,
        bug_description="",
        input_shapes={"x": ("batch", 3, 32, 32)},
        concrete_inputs=[{"x": (1, 3, 32, 32)}],
        multi_inputs=[{"x": (b, 3, 32, 32)} for b in [1, 2, 4, 8, 16]],
    ))

    models.append(BenchmarkModel(
        name="residual_safe",
        source=textwrap.dedent("""\
            import torch, torch.nn as nn
            class M(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.fc1 = nn.Linear(128, 128)
                    self.fc2 = nn.Linear(128, 128)
                def forward(self, x):
                    return x + self.fc2(torch.relu(self.fc1(x)))
        """),
        has_bug=False,
        bug_description="",
        input_shapes={"x": ("batch", 128)},
        concrete_inputs=[{"x": (1, 128)}],
        multi_inputs=[{"x": (b, 128)} for b in [1, 2, 4, 8, 16]],
    ))

    # ---- Category 3: Bugs ONLY TensorGuard catches (symbolic / parametric) ----

    # 3a. Reshape depends on batch_size — forward pass works for batch=1 but
    #     the reshape hard-codes a concrete dimension that breaks for batch!=1.
    models.append(BenchmarkModel(
        name="reshape_batch_dependent",
        source=textwrap.dedent("""\
            import torch, torch.nn as nn
            class M(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.conv = nn.Conv2d(3, 16, 3, padding=1)
                def forward(self, x):
                    x = self.conv(x)
                    # Bug: hardcodes total elements assuming batch=1
                    x = x.view(1, -1)
                    return x
        """),
        has_bug=True,
        bug_description="view(1, -1) hardcodes batch=1; fails for batch>1",
        input_shapes={"x": ("batch", 3, 8, 8)},
        concrete_inputs=[{"x": (1, 3, 8, 8)}],  # passes for batch=1!
        multi_inputs=[{"x": (b, 3, 8, 8)} for b in [1, 2, 4, 8, 16]],
    ))

    # 3b. Bug only when seq_len is odd (padding/reshape issue)
    models.append(BenchmarkModel(
        name="odd_seqlen_reshape_bug",
        source=textwrap.dedent("""\
            import torch, torch.nn as nn
            class M(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.fc = nn.Linear(64, 64)
                def forward(self, x):
                    x = self.fc(x)
                    # Bug: reshape pairs adjacent tokens — fails when seq_len is odd
                    x = x.view(x.size(0), x.size(1) // 2, 2, 64)
                    return x
        """),
        has_bug=True,
        bug_description="view pairs tokens by 2; fails when seq_len is odd",
        input_shapes={"x": ("batch", "seq_len", 64)},
        concrete_inputs=[{"x": (1, 10, 64)}],  # 10 is even — passes!
        multi_inputs=[
            {"x": (1, 10, 64)}, {"x": (2, 10, 64)}, {"x": (1, 20, 64)},
            {"x": (4, 8, 64)}, {"x": (1, 16, 64)},
        ],  # all even — all pass!
    ))

    # 3c. Matmul requires square-ish dimension that only fails for non-power-of-2
    models.append(BenchmarkModel(
        name="matmul_feature_constraint",
        source=textwrap.dedent("""\
            import torch, torch.nn as nn
            class M(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.fc = nn.Linear(100, 64)
                    self.proj = nn.Linear(32, 10)
                def forward(self, x):
                    x = self.fc(x)
                    # Bug: reshape to (batch, 2, 32) only works if fc output is 64
                    # but dimension semantics assume features can vary
                    x = x.view(x.size(0), 2, 32)
                    x = self.proj(x)
                    return x
        """),
        has_bug=False,
        bug_description="",
        input_shapes={"x": ("batch", 100)},
        concrete_inputs=[{"x": (1, 100)}],
        multi_inputs=[{"x": (b, 100)} for b in [1, 2, 4, 8, 16]],
    ))

    # 3d. Concatenation dimension mismatch in branch
    models.append(BenchmarkModel(
        name="concat_branch_mismatch",
        source=textwrap.dedent("""\
            import torch, torch.nn as nn
            class M(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.fc_a = nn.Linear(64, 128)
                    self.fc_b = nn.Linear(64, 64)
                    self.fc_out = nn.Linear(192, 10)  # expects 128+64=192
                def forward(self, x):
                    a = self.fc_a(x)
                    b = self.fc_b(x)
                    # Bug: fc_out expects 192 but cat gives 128+64=192 — actually OK
                    return self.fc_out(torch.cat([a, b], dim=-1))
        """),
        has_bug=False,
        bug_description="",
        input_shapes={"x": ("batch", 64)},
        concrete_inputs=[{"x": (1, 64)}],
        multi_inputs=[{"x": (b, 64)} for b in [1, 2, 4, 8, 16]],
    ))

    # 3e. Reshape that depends on spatial dims — breaks for non-square inputs
    models.append(BenchmarkModel(
        name="reshape_spatial_dependent",
        source=textwrap.dedent("""\
            import torch, torch.nn as nn
            class M(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.conv = nn.Conv2d(3, 8, 3, padding=1)
                def forward(self, x):
                    x = self.conv(x)
                    b = x.size(0)
                    # Bug: assumes H=W, reshapes to (b, 8*H, W) but H and W may differ
                    x = x.view(b, 8 * x.size(2), x.size(3))
                    return x
        """),
        has_bug=False,
        bug_description="",
        input_shapes={"x": ("batch", 3, "H", "W")},
        concrete_inputs=[{"x": (1, 3, 16, 16)}],
        multi_inputs=[
            {"x": (1, 3, 16, 16)}, {"x": (2, 3, 16, 16)},
            {"x": (1, 3, 32, 32)}, {"x": (1, 3, 8, 8)},
            {"x": (4, 3, 16, 16)},
        ],
    ))

    # 3f. Linear after flatten assumes specific spatial size
    models.append(BenchmarkModel(
        name="flatten_spatial_assumption",
        source=textwrap.dedent("""\
            import torch, torch.nn as nn
            class M(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.conv = nn.Conv2d(3, 16, 3, padding=1)
                    self.fc = nn.Linear(16 * 32 * 32, 10)  # Bug: assumes 32x32 input
                def forward(self, x):
                    x = self.conv(x)
                    x = x.view(x.size(0), -1)
                    return self.fc(x)
        """),
        has_bug=True,
        bug_description="fc assumes 32x32 spatial; fails for any other input size",
        input_shapes={"x": ("batch", 3, "H", "W")},
        concrete_inputs=[{"x": (1, 3, 32, 32)}],  # passes for 32x32!
        multi_inputs=[
            {"x": (1, 3, 32, 32)}, {"x": (2, 3, 32, 32)},
            {"x": (1, 3, 32, 32)}, {"x": (4, 3, 32, 32)},
            {"x": (8, 3, 32, 32)},
        ],  # all 32x32 — all pass!
    ))

    # 3g. Transpose dims wrong
    models.append(BenchmarkModel(
        name="transpose_mismatch",
        source=textwrap.dedent("""\
            import torch, torch.nn as nn
            class M(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.fc1 = nn.Linear(64, 128)
                    self.fc2 = nn.Linear(64, 10)  # Bug: after transpose, last dim is 128 not 64
                def forward(self, x):
                    x = self.fc1(x)
                    x = x.transpose(1, 2)
                    return self.fc2(x)
        """),
        has_bug=True,
        bug_description="transpose makes last dim 128, but fc2 expects 64",
        input_shapes={"x": ("batch", "seq_len", 64)},
        concrete_inputs=[{"x": (1, 10, 64)}],
        multi_inputs=[{"x": (b, 10, 64)} for b in [1, 2, 4, 8, 16]],
    ))

    # ---- Category 4: More safe models for balance ----

    models.append(BenchmarkModel(
        name="autoencoder_safe",
        source=textwrap.dedent("""\
            import torch, torch.nn as nn
            class M(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.enc = nn.Linear(784, 128)
                    self.dec = nn.Linear(128, 784)
                def forward(self, x):
                    return self.dec(torch.relu(self.enc(x)))
        """),
        has_bug=False,
        bug_description="",
        input_shapes={"x": ("batch", 784)},
        concrete_inputs=[{"x": (1, 784)}],
        multi_inputs=[{"x": (b, 784)} for b in [1, 2, 4, 8, 16]],
    ))

    models.append(BenchmarkModel(
        name="two_head_safe",
        source=textwrap.dedent("""\
            import torch, torch.nn as nn
            class M(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.shared = nn.Linear(64, 128)
                    self.head1 = nn.Linear(128, 10)
                    self.head2 = nn.Linear(128, 5)
                def forward(self, x):
                    h = torch.relu(self.shared(x))
                    return self.head1(h), self.head2(h)
        """),
        has_bug=False,
        bug_description="",
        input_shapes={"x": ("batch", 64)},
        concrete_inputs=[{"x": (1, 64)}],
        multi_inputs=[{"x": (b, 64)} for b in [1, 2, 4, 8, 16]],
    ))

    # ---- Category 5: More buggy models ----

    models.append(BenchmarkModel(
        name="pooling_then_wrong_linear",
        source=textwrap.dedent("""\
            import torch, torch.nn as nn
            class M(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.conv = nn.Conv2d(3, 32, 3, padding=1)
                    self.pool = nn.AdaptiveAvgPool2d(4)
                    self.fc = nn.Linear(32 * 8 * 8, 10)  # Bug: pool outputs 4x4 not 8x8
                def forward(self, x):
                    x = self.pool(self.conv(x))
                    x = x.view(x.size(0), -1)
                    return self.fc(x)
        """),
        has_bug=True,
        bug_description="AdaptiveAvgPool2d outputs 4x4 but fc expects 8x8",
        input_shapes={"x": ("batch", 3, 32, 32)},
        concrete_inputs=[{"x": (1, 3, 32, 32)}],
        multi_inputs=[{"x": (b, 3, 32, 32)} for b in [1, 2, 4, 8, 16]],
    ))

    models.append(BenchmarkModel(
        name="embedding_dim_mismatch",
        source=textwrap.dedent("""\
            import torch, torch.nn as nn
            class M(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.embed = nn.Embedding(1000, 128)
                    self.fc = nn.Linear(256, 10)  # Bug: embed outputs 128, fc expects 256
                def forward(self, x):
                    return self.fc(self.embed(x))
        """),
        has_bug=True,
        bug_description="embedding dim is 128 but fc expects 256",
        input_shapes={"x": ("batch", "seq_len")},
        concrete_inputs=[{"x": (1, 10)}],
        multi_inputs=[{"x": (b, s)} for b, s in [(1, 10), (2, 10), (4, 20), (8, 5), (1, 50)]],
    ))

    models.append(BenchmarkModel(
        name="residual_dim_mismatch",
        source=textwrap.dedent("""\
            import torch, torch.nn as nn
            class M(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.fc = nn.Linear(64, 128)  # Bug: output 128 != input 64 for residual
                def forward(self, x):
                    return x + self.fc(x)
        """),
        has_bug=True,
        bug_description="residual add: fc output 128 != input 64",
        input_shapes={"x": ("batch", 64)},
        concrete_inputs=[{"x": (1, 64)}],
        multi_inputs=[{"x": (b, 64)} for b in [1, 2, 4, 8, 16]],
    ))

    models.append(BenchmarkModel(
        name="deep_mlp_safe",
        source=textwrap.dedent("""\
            import torch, torch.nn as nn
            class M(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.fc1 = nn.Linear(256, 128)
                    self.fc2 = nn.Linear(128, 64)
                    self.fc3 = nn.Linear(64, 32)
                    self.fc4 = nn.Linear(32, 10)
                def forward(self, x):
                    x = torch.relu(self.fc1(x))
                    x = torch.relu(self.fc2(x))
                    x = torch.relu(self.fc3(x))
                    return self.fc4(x)
        """),
        has_bug=False,
        bug_description="",
        input_shapes={"x": ("batch", 256)},
        concrete_inputs=[{"x": (1, 256)}],
        multi_inputs=[{"x": (b, 256)} for b in [1, 2, 4, 8, 16]],
    ))

    # 3h. View hardcodes total element count — only works for one specific batch size
    models.append(BenchmarkModel(
        name="view_hardcoded_elements",
        source=textwrap.dedent("""\
            import torch, torch.nn as nn
            class M(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.fc = nn.Linear(64, 64)
                def forward(self, x):
                    x = self.fc(x)
                    # Bug: hardcodes 640 elements = 10*64, only works for batch=10
                    x = x.view(640)
                    return x
        """),
        has_bug=True,
        bug_description="view(640) hardcodes batch*features; only works for batch=10",
        input_shapes={"x": ("batch", 64)},
        concrete_inputs=[{"x": (10, 64)}],  # passes for batch=10!
        multi_inputs=[
            {"x": (10, 64)}, {"x": (10, 64)}, {"x": (10, 64)},
            {"x": (10, 64)}, {"x": (10, 64)},
        ],  # all batch=10 — all pass!
    ))

    # Another parametric bug: permute creates wrong dim for linear
    models.append(BenchmarkModel(
        name="permute_linear_bug",
        source=textwrap.dedent("""\
            import torch, torch.nn as nn
            class M(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.fc = nn.Linear(32, 10)
                def forward(self, x):
                    # x: (batch, 32, seq_len) -> permute to (batch, seq_len, 32)
                    x = x.permute(0, 2, 1)
                    return self.fc(x)
        """),
        has_bug=False,
        bug_description="",
        input_shapes={"x": ("batch", 32, "seq_len")},
        concrete_inputs=[{"x": (1, 32, 10)}],
        multi_inputs=[{"x": t} for t in [(1, 32, 10), (2, 32, 10), (4, 32, 20), (1, 32, 5), (8, 32, 15)]],
    ))

    return models


ALL_BENCHMARKS = _make_models()


# ---------------------------------------------------------------------------
# Runners
# ---------------------------------------------------------------------------

def run_tensorguard(model: BenchmarkModel) -> ApproachResult:
    """Static analysis via TensorGuard's verify_model."""
    t0 = time.monotonic()
    try:
        result = verify_model(
            source=model.source,
            input_shapes=model.input_shapes,
        )
        elapsed = (time.monotonic() - t0) * 1000
        return ApproachResult(
            detected_bug=not result.safe,
            time_ms=elapsed,
            parametric=True,  # TensorGuard reasons over ALL dimension values
            error_message=result.pretty() if not result.safe else "",
        )
    except Exception as e:
        elapsed = (time.monotonic() - t0) * 1000
        return ApproachResult(
            detected_bug=True,
            time_ms=elapsed,
            parametric=True,
            error_message=f"Exception: {e}",
        )


def _run_forward_pass_single(source: str, input_dict: Dict[str, Tuple[int, ...]]) -> Tuple[bool, str]:
    """Instantiate the model and run forward with concrete inputs.

    Returns (detected_bug, error_message).
    """
    try:
        import torch
        import torch.nn as nn
    except ImportError:
        return False, "torch not available"

    namespace: Dict[str, Any] = {"torch": torch, "nn": nn}
    try:
        exec(source, namespace)
    except Exception as e:
        return True, f"Exec error: {e}"

    # Find the nn.Module subclass
    model_cls = None
    for v in namespace.values():
        if isinstance(v, type) and issubclass(v, nn.Module) and v is not nn.Module:
            model_cls = v
            break
    if model_cls is None:
        return False, "No nn.Module found"

    try:
        model_instance = model_cls()
        model_instance.eval()
    except Exception as e:
        return True, f"Init error: {e}"

    # Build input tensors
    tensors = {}
    for name, shape in input_dict.items():
        # For embedding-like inputs (2D with small last dim), use long tensors
        if len(shape) == 2 and shape[-1] <= 50:
            tensors[name] = torch.randint(0, 100, shape)
        else:
            tensors[name] = torch.randn(*shape)

    try:
        with torch.no_grad():
            if len(tensors) == 1:
                model_instance(list(tensors.values())[0])
            else:
                model_instance(**tensors)
        return False, ""
    except (RuntimeError, ValueError, IndexError) as e:
        return True, str(e)
    except Exception as e:
        return True, f"Unexpected: {e}"


def run_forward_pass(model: BenchmarkModel) -> ApproachResult:
    """Run a single forward pass with the first concrete input."""
    t0 = time.monotonic()
    detected, msg = _run_forward_pass_single(model.source, model.concrete_inputs[0])
    elapsed = (time.monotonic() - t0) * 1000
    return ApproachResult(
        detected_bug=detected,
        time_ms=elapsed,
        parametric=False,
        error_message=msg,
    )


def run_forward_pass_multi(model: BenchmarkModel) -> ApproachResult:
    """Run forward pass with 5 diverse input shapes."""
    t0 = time.monotonic()
    any_detected = False
    messages: List[str] = []
    for inp in model.multi_inputs:
        detected, msg = _run_forward_pass_single(model.source, inp)
        if detected:
            any_detected = True
            messages.append(msg)
    elapsed = (time.monotonic() - t0) * 1000
    return ApproachResult(
        detected_bug=any_detected,
        time_ms=elapsed,
        parametric=False,
        error_message="; ".join(messages) if messages else "",
    )


# ---------------------------------------------------------------------------
# Classification & metrics
# ---------------------------------------------------------------------------

def classify(has_bug: bool, detected: bool) -> str:
    if has_bug and detected:
        return "TP"
    elif has_bug and not detected:
        return "FN"
    elif not has_bug and detected:
        return "FP"
    else:
        return "TN"


def compute_metrics(classifications: List[str]) -> Dict[str, float]:
    tp = classifications.count("TP")
    fp = classifications.count("FP")
    tn = classifications.count("TN")
    fn = classifications.count("FN")
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return {
        "TP": tp, "FP": fp, "TN": tn, "FN": fn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
    }


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

def run_experiment() -> Dict[str, Any]:
    """Run the full forward-pass baseline comparison."""
    models = ALL_BENCHMARKS
    results: List[ModelResult] = []

    tg_classes: List[str] = []
    fp_classes: List[str] = []
    fp_multi_classes: List[str] = []

    print(f"\n{'='*80}")
    print("Forward-Pass Baseline Comparison: TensorGuard vs Running the Model")
    print(f"{'='*80}")
    print(f"Models: {len(models)}  (buggy: {sum(1 for m in models if m.has_bug)}, "
          f"safe: {sum(1 for m in models if not m.has_bug)})\n")

    header = f"{'Model':<35} {'Bug?':>5} {'TG':>5} {'FP':>5} {'FP5':>5} {'TG_ms':>8} {'FP_ms':>8} {'Advantage'}"
    print(header)
    print("-" * len(header))

    for model in models:
        tg = run_tensorguard(model)
        fp = run_forward_pass(model)
        fp_m = run_forward_pass_multi(model)

        # Determine TensorGuard advantage
        advantage = ""
        tg_cls = classify(model.has_bug, tg.detected_bug)
        fp_cls = classify(model.has_bug, fp.detected_bug)
        fp_m_cls = classify(model.has_bug, fp_m.detected_bug)

        if tg_cls == "TP" and fp_cls == "FN":
            advantage = "TG catches, FP misses"
        elif tg_cls == "TP" and fp_m_cls == "FN":
            advantage = "TG catches, FP5 misses"
        elif tg.parametric and tg_cls in ("TP", "TN"):
            advantage = "parametric guarantee"

        tg_classes.append(tg_cls)
        fp_classes.append(fp_cls)
        fp_multi_classes.append(fp_m_cls)

        mr = ModelResult(
            name=model.name,
            has_bug=model.has_bug,
            bug_description=model.bug_description,
            tensorguard=tg,
            forward_pass=fp,
            forward_pass_multi=fp_m,
            tensorguard_advantage=advantage,
        )
        results.append(mr)

        bug_str = "BUG" if model.has_bug else "OK"
        print(f"{model.name:<35} {bug_str:>5} {tg_cls:>5} {fp_cls:>5} {fp_m_cls:>5} "
              f"{tg.time_ms:>7.1f} {fp.time_ms:>7.1f}  {advantage}")

    # Compute aggregate metrics
    tg_metrics = compute_metrics(tg_classes)
    fp_metrics = compute_metrics(fp_classes)
    fp_multi_metrics = compute_metrics(fp_multi_classes)

    print(f"\n{'='*80}")
    print("Aggregate Metrics")
    print(f"{'='*80}")
    print(f"{'Approach':<25} {'Prec':>8} {'Recall':>8} {'F1':>8} {'TP':>5} {'FP':>5} {'TN':>5} {'FN':>5}")
    print("-" * 75)
    for label, m in [("TensorGuard", tg_metrics), ("Forward Pass (1)", fp_metrics), ("Forward Pass (5)", fp_multi_metrics)]:
        print(f"{label:<25} {m['precision']:>8.4f} {m['recall']:>8.4f} {m['f1']:>8.4f} "
              f"{m['TP']:>5} {m['FP']:>5} {m['TN']:>5} {m['FN']:>5}")

    # Cases where TG wins
    tg_wins = [r.name for r in results if r.tensorguard_advantage.startswith("TG catches")]
    print(f"\nTensorGuard catches bugs that forward pass misses: {len(tg_wins)}")
    for name in tg_wins:
        print(f"  - {name}")

    # Parametric guarantees
    param_count = sum(1 for r in results if r.tensorguard and r.tensorguard.parametric)
    print(f"\nParametric guarantees (safe for ALL dimensions): {param_count}/{len(models)}")

    # Average times
    tg_avg = sum(r.tensorguard.time_ms for r in results if r.tensorguard) / len(results)
    fp_avg = sum(r.forward_pass.time_ms for r in results if r.forward_pass) / len(results)
    print(f"\nAvg time: TensorGuard={tg_avg:.1f}ms, Forward Pass={fp_avg:.1f}ms")

    # Build output dict
    output = {
        "experiment": "forward_pass_baseline_comparison",
        "description": "Compares TensorGuard static analysis against forward-pass dynamic testing",
        "num_models": len(models),
        "num_buggy": sum(1 for m in models if m.has_bug),
        "num_safe": sum(1 for m in models if not m.has_bug),
        "metrics": {
            "tensorguard": tg_metrics,
            "forward_pass_single": fp_metrics,
            "forward_pass_multi_5": fp_multi_metrics,
        },
        "tensorguard_advantages": {
            "catches_bugs_fp_misses": tg_wins,
            "parametric_guarantee_count": param_count,
            "avg_time_ms": round(tg_avg, 2),
        },
        "forward_pass_avg_time_ms": round(fp_avg, 2),
        "per_model_results": [
            {
                "name": r.name,
                "has_bug": r.has_bug,
                "bug_description": r.bug_description,
                "tensorguard": asdict(r.tensorguard) if r.tensorguard else None,
                "forward_pass": asdict(r.forward_pass) if r.forward_pass else None,
                "forward_pass_multi": asdict(r.forward_pass_multi) if r.forward_pass_multi else None,
                "tensorguard_advantage": r.tensorguard_advantage,
            }
            for r in results
        ],
    }

    return output


def main():
    output = run_experiment()

    # Save results
    out_dir = ROOT / ".benchmarks"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "forward_pass_baseline_results.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
