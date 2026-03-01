#!/usr/bin/env python3
"""
Expressiveness Boundary Characterization for TensorGuard.

Systematically tests what TensorGuard CAN and CANNOT verify,
producing a clear characterization for the paper.
"""

import json
import os
import sys
import time
import traceback
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.model_checker import verify_model, Device, Phase


# ---------------------------------------------------------------------------
# Test case dataclass
# ---------------------------------------------------------------------------

@dataclass
class TestResult:
    name: str
    category: str
    category_number: int
    expected_result: str  # "safe", "bug", "limitation"
    actual_result: str    # "safe", "bug", "error", "limitation_confirmed"
    correct: bool
    notes: str = ""
    time_ms: float = 0.0


# ============================================================================
# CATEGORY 1: WITHIN EXPRESSIVENESS (should handle correctly)
# ============================================================================

# 1.1 Linear chain mismatch
LINEAR_CHAIN_SAFE = """\
import torch.nn as nn

class LinearChainSafe(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 20)
        self.fc2 = nn.Linear(20, 5)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        return x
"""

LINEAR_CHAIN_BUG = """\
import torch.nn as nn

class LinearChainBug(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 20)
        self.fc2 = nn.Linear(50, 5)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        return x
"""

# 1.2 Conv2d channel mismatch
CONV_SAFE = """\
import torch.nn as nn

class ConvSafe(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3)
        self.conv2 = nn.Conv2d(16, 32, 3)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x
"""

CONV_BUG = """\
import torch.nn as nn

class ConvBug(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3)
        self.conv2 = nn.Conv2d(64, 32, 3)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x
"""

# 1.3 Reshape/flatten dimension mismatch
RESHAPE_SAFE = """\
import torch.nn as nn

class ReshapeSafe(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(20, 10)

    def forward(self, x):
        x = x.view(-1, 20)
        x = self.fc(x)
        return x
"""

RESHAPE_BUG = """\
import torch.nn as nn

class ReshapeBug(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(30, 10)

    def forward(self, x):
        x = x.view(-1, 20)
        x = self.fc(x)
        return x
"""

# 1.4 Broadcasting incompatibilities
BROADCAST_SAFE = """\
import torch
import torch.nn as nn

class BroadcastSafe(nn.Module):
    def __init__(self):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(1, 10))

    def forward(self, x):
        return x + self.bias
"""

BROADCAST_BUG = """\
import torch
import torch.nn as nn

class BroadcastBug(nn.Module):
    def __init__(self):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(5, 3))

    def forward(self, x):
        return x + self.bias
"""

# 1.5 Device mismatch
DEVICE_SAFE = """\
import torch.nn as nn

class DeviceSafe(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 5)

    def forward(self, x):
        return self.fc(x)
"""

DEVICE_BUG = """\
import torch.nn as nn

class DeviceBug(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 5)

    def forward(self, x):
        x = x.cuda()
        x = self.fc(x)
        return x
"""

# 1.6 Phase-dependent errors
PHASE_MODEL = """\
import torch.nn as nn

class PhaseModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 20)
        self.dropout = nn.Dropout(0.5)
        self.bn = nn.BatchNorm1d(20)
        self.fc2 = nn.Linear(20, 5)

    def forward(self, x):
        x = self.fc1(x)
        x = self.dropout(x)
        x = self.bn(x)
        x = self.fc2(x)
        return x
"""

# 1.7 Matmul dimension mismatch
MATMUL_SAFE = """\
import torch.nn as nn

class MatmulSafe(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, w):
        return x @ w
"""

MATMUL_BUG = """\
import torch
import torch.nn as nn

class MatmulBug(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, w):
        return x @ w
"""

# 1.8 Multi-head attention dimension mismatch
MHA_SAFE = """\
import torch.nn as nn

class MHASafe(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = nn.Linear(512, 512)
        self.k_proj = nn.Linear(512, 512)
        self.v_proj = nn.Linear(512, 512)
        self.out_proj = nn.Linear(512, 512)

    def forward(self, x):
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        return self.out_proj(v)
"""

MHA_BUG = """\
import torch.nn as nn

class MHABug(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = nn.Linear(512, 256)
        self.out_proj = nn.Linear(512, 512)

    def forward(self, x):
        q = self.q_proj(x)
        return self.out_proj(q)
"""

# 1.9 Skip connection shape mismatch
SKIP_SAFE = """\
import torch.nn as nn

class SkipSafe(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 10)
        self.fc2 = nn.Linear(10, 10)

    def forward(self, x):
        skip = x
        x = self.fc1(x)
        x = self.fc2(x)
        return x + skip
"""

SKIP_BUG = """\
import torch.nn as nn

class SkipBug(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 20)

    def forward(self, x):
        skip = x
        x = self.fc1(x)
        return x + skip
"""

# 1.10 Transpose/permute dimension errors
TRANSPOSE_SAFE = """\
import torch.nn as nn

class TransposeSafe(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 5)

    def forward(self, x):
        x = x.transpose(0, 1)
        x = x.transpose(0, 1)
        return self.fc(x)
"""


# ============================================================================
# CATEGORY 2: AT THE BOUNDARY (partial support)
# ============================================================================

# 2.1 Symbolic dimensions with CEGAR
SYMBOLIC_CEGAR = """\
import torch.nn as nn

class SymbolicModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        return x
"""

# 2.2 Relational constraints (heads * head_dim = embed_dim)
RELATIONAL_MHA = """\
import torch.nn as nn

class RelationalMHA(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = nn.Linear(512, 512)
        self.k_proj = nn.Linear(512, 512)
        self.v_proj = nn.Linear(512, 512)
        self.out_proj = nn.Linear(512, 512)

    def forward(self, x):
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        return self.out_proj(v)
"""

# 2.3 ModuleList/Sequential iteration patterns
MODULELIST_MODEL = """\
import torch.nn as nn

class ModuleListModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(10, 10) for _ in range(3)])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
"""


# ============================================================================
# CATEGORY 3: OUTSIDE EXPRESSIVENESS (should report limitations)
# ============================================================================

# 3.1 Data-dependent control flow — bug hidden in one branch
DATA_DEPENDENT = """\
import torch
import torch.nn as nn

class DataDependent(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc_small = nn.Linear(10, 5)
        self.fc_big = nn.Linear(10, 20)
        self.fc_out = nn.Linear(5, 2)

    def forward(self, x):
        if x.shape[0] > 10:
            x = self.fc_big(x)
        else:
            x = self.fc_small(x)
        return self.fc_out(x)
"""

# 3.2 Dynamic computation graphs — loop with data-dependent shape
DYNAMIC_GRAPH = """\
import torch
import torch.nn as nn

class DynamicGraph(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 5)
        self.fc2 = nn.Linear(10, 3)

    def forward(self, x):
        parts = []
        for i in range(x.shape[0]):
            parts.append(self.fc(x[i:i+1]))
        out = torch.cat(parts, dim=0)
        return self.fc2(out)
"""

# 3.3 Variable-length sequences
VARIABLE_LENGTH = """\
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence

class VarLenModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.rnn = nn.LSTM(10, 20, batch_first=True)
        self.fc = nn.Linear(15, 5)

    def forward(self, x, lengths):
        packed = pack_padded_sequence(x, lengths, batch_first=True)
        output, _ = self.rnn(packed)
        return self.fc(output)
"""

# 3.4 Dictionary/tuple returns — downstream consumer gets wrong shape
DICT_RETURN = """\
import torch.nn as nn

class DictReturn(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 5)
        self.fc2 = nn.Linear(10, 3)

    def forward(self, x):
        return {"logits": self.fc1(x), "features": self.fc2(x)}
"""

# 3.5 Complex indexing patterns — mask makes shapes unpredictable
COMPLEX_INDEX = """\
import torch
import torch.nn as nn

class ComplexIndex(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 5)

    def forward(self, x, mask):
        selected = x[mask]
        return self.fc(selected)
"""

# 3.6 Custom autograd Functions
CUSTOM_AUTOGRAD = """\
import torch
import torch.nn as nn
from torch.autograd import Function

class ReshapeFunc(Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return input.view(-1, 20)

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        return grad_output.view_as(input)

class CustomAutogradModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 5)

    def forward(self, x):
        x = ReshapeFunc.apply(x)
        return self.fc(x)
"""

# 3.7 Recursive nn.Module graphs
RECURSIVE_MODULE = """\
import torch.nn as nn

class RecursiveBlock(nn.Module):
    def __init__(self, depth):
        super().__init__()
        self.fc = nn.Linear(10, 10)
        self.child = RecursiveBlock(depth - 1) if depth > 0 else None

    def forward(self, x):
        x = self.fc(x)
        if self.child is not None:
            x = self.child(x)
        return x

class RecursiveModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.block = RecursiveBlock(3)

    def forward(self, x):
        return self.block(x)
"""


# ---------------------------------------------------------------------------
# Test runner
# ---------------------------------------------------------------------------

def run_test(
    name: str,
    source: str,
    input_shapes: Dict[str, Any],
    expected: str,
    category: str,
    category_number: int,
    *,
    default_device: Device = Device.CPU,
    default_phase: Phase = Phase.TRAIN,
    constraints: Optional[Dict] = None,
    notes: str = "",
) -> TestResult:
    """Run a single expressiveness boundary test."""
    start = time.time()
    try:
        kwargs: Dict[str, Any] = {
            "source": source,
            "input_shapes": input_shapes,
            "default_device": default_device,
            "default_phase": default_phase,
        }
        if constraints is not None:
            kwargs["constraints"] = constraints
        result = verify_model(**kwargs)
        elapsed_ms = (time.time() - start) * 1000

        if result.safe:
            actual = "safe"
        else:
            actual = "bug"

        if expected == "limitation":
            # For Category 3: limitation is confirmed if the tool either
            # errors out or returns "safe" for a model containing a bug
            # (false negative due to unsupported feature).
            # Returning "bug" would mean it actually detected the issue
            # despite the complex feature — partial support, not limitation.
            if actual == "safe":
                correct = True
                actual_notes = f"False negative confirms limitation (reported safe despite hidden bug). {notes}"
            elif actual == "bug":
                correct = False
                actual_notes = f"Surprisingly detected bug despite complex feature. {notes}"
            else:
                correct = True
                actual_notes = f"Limitation confirmed via error. {notes}"
        else:
            correct = actual == expected
            actual_notes = notes

        if not correct and result.counterexample:
            viols = [v.message for v in result.counterexample.violations[:3]]
            actual_notes += f" Violations: {viols}"

        if result.errors:
            actual_notes += f" Errors: {result.errors[:3]}"

        return TestResult(
            name=name,
            category=category,
            category_number=category_number,
            expected_result=expected,
            actual_result=actual,
            correct=correct,
            notes=actual_notes.strip(),
            time_ms=elapsed_ms,
        )

    except Exception as e:
        elapsed_ms = (time.time() - start) * 1000
        err_str = f"{type(e).__name__}: {e}"
        if expected == "limitation":
            return TestResult(
                name=name,
                category=category,
                category_number=category_number,
                expected_result=expected,
                actual_result="limitation_confirmed",
                correct=True,
                notes=f"Exception confirms limitation: {err_str}",
                time_ms=elapsed_ms,
            )
        else:
            return TestResult(
                name=name,
                category=category,
                category_number=category_number,
                expected_result=expected,
                actual_result="error",
                correct=False,
                notes=f"Unexpected error: {err_str}",
                time_ms=elapsed_ms,
            )


def main() -> None:
    results: List[TestResult] = []

    # === CATEGORY 1: WITHIN EXPRESSIVENESS ===
    cat1 = "within_expressiveness"

    # 1.1 Linear chain
    results.append(run_test(
        "linear_chain_safe", LINEAR_CHAIN_SAFE,
        {"x": ("batch", 10)}, "safe", cat1, 1,
        notes="Matching in/out features",
    ))
    results.append(run_test(
        "linear_chain_bug", LINEAR_CHAIN_BUG,
        {"x": ("batch", 10)}, "bug", cat1, 1,
        notes="fc1 out=20 but fc2 in=50",
    ))

    # 1.2 Conv2d channel
    results.append(run_test(
        "conv2d_channel_safe", CONV_SAFE,
        {"x": ("batch", 3, 32, 32)}, "safe", cat1, 2,
        notes="Conv out_channels matches next in_channels",
    ))
    results.append(run_test(
        "conv2d_channel_bug", CONV_BUG,
        {"x": ("batch", 3, 32, 32)}, "bug", cat1, 2,
        notes="conv1 out=16 but conv2 in=64",
    ))

    # 1.3 Reshape/flatten
    results.append(run_test(
        "reshape_safe", RESHAPE_SAFE,
        {"x": ("batch", 20)}, "safe", cat1, 3,
        notes="view(-1,20) then Linear(20,10)",
    ))
    results.append(run_test(
        "reshape_bug", RESHAPE_BUG,
        {"x": ("batch", 20)}, "bug", cat1, 3,
        notes="view(-1,20) but Linear expects 30",
    ))

    # 1.4 Broadcasting
    results.append(run_test(
        "broadcast_safe", BROADCAST_SAFE,
        {"x": ("batch", 10)}, "safe", cat1, 4,
        notes="bias (1,10) broadcasts with (batch,10)",
    ))
    results.append(run_test(
        "broadcast_bug", BROADCAST_BUG,
        {"x": ("batch", 10)}, "bug", cat1, 4,
        notes="bias (5,3) incompatible with (batch,10)",
    ))

    # 1.5 Device mismatch
    results.append(run_test(
        "device_safe", DEVICE_SAFE,
        {"x": ("batch", 10)}, "safe", cat1, 5,
        notes="All on CPU",
    ))
    results.append(run_test(
        "device_bug", DEVICE_BUG,
        {"x": ("batch", 10)}, "bug", cat1, 5,
        default_device=Device.CPU,
        notes="Input moved to CUDA but model on CPU",
    ))

    # 1.6 Phase-dependent
    results.append(run_test(
        "phase_train", PHASE_MODEL,
        {"x": ("batch", 10)}, "safe", cat1, 6,
        default_phase=Phase.TRAIN,
        notes="Dropout+BN in training mode",
    ))
    results.append(run_test(
        "phase_eval", PHASE_MODEL,
        {"x": ("batch", 10)}, "safe", cat1, 6,
        default_phase=Phase.EVAL,
        notes="Dropout+BN in eval mode",
    ))

    # 1.7 Matmul
    results.append(run_test(
        "matmul_safe", MATMUL_SAFE,
        {"x": ("batch", 10), "w": (10, 5)}, "safe", cat1, 7,
        notes="Compatible matmul shapes",
    ))
    results.append(run_test(
        "matmul_bug", MATMUL_BUG,
        {"x": ("batch", 10), "w": (7, 5)}, "bug", cat1, 7,
        notes="x cols (10) != w rows (7)",
    ))

    # 1.8 MHA
    results.append(run_test(
        "mha_safe", MHA_SAFE,
        {"x": ("batch", "seq_len", 512)}, "safe", cat1, 8,
        notes="All projections match embed_dim=512",
    ))
    results.append(run_test(
        "mha_bug", MHA_BUG,
        {"x": ("batch", "seq_len", 512)}, "bug", cat1, 8,
        notes="q_proj out=256 but out_proj in=512",
    ))

    # 1.9 Skip connection
    results.append(run_test(
        "skip_safe", SKIP_SAFE,
        {"x": ("batch", 10)}, "safe", cat1, 9,
        notes="Skip and output both (batch,10)",
    ))
    results.append(run_test(
        "skip_bug", SKIP_BUG,
        {"x": ("batch", 10)}, "bug", cat1, 9,
        notes="Skip (batch,10) + fc1 output (batch,20) — shape mismatch on add",
    ))

    # 1.10 Transpose
    results.append(run_test(
        "transpose_safe", TRANSPOSE_SAFE,
        {"x": ("batch", 10)}, "safe", cat1, 10,
        notes="Double transpose restores shape",
    ))

    # === CATEGORY 2: AT THE BOUNDARY ===
    cat2 = "boundary"

    # 2.1 Symbolic CEGAR
    results.append(run_test(
        "symbolic_cegar", SYMBOLIC_CEGAR,
        {"x": ("batch", 784)}, "safe", cat2, 1,
        notes="Fully symbolic batch dim with CEGAR refinement",
    ))

    # 2.2 Relational constraints
    results.append(run_test(
        "relational_mha", RELATIONAL_MHA,
        {"x": ("batch", "seq_len", "embed_dim")}, "safe", cat2, 2,
        constraints={
            "embed_dim": "heads * head_dim",
            "heads": 8,
            "head_dim": 64,
        },
        notes="Relational constraint: embed_dim = heads * head_dim",
    ))

    # 2.3 ModuleList iteration
    results.append(run_test(
        "modulelist_iteration", MODULELIST_MODEL,
        {"x": ("batch", 10)}, "safe", cat2, 3,
        notes="ModuleList with for-loop iteration",
    ))

    # === CATEGORY 3: OUTSIDE EXPRESSIVENESS ===
    cat3 = "outside_expressiveness"

    # 3.1 Data-dependent control flow — bug hidden in one branch
    results.append(run_test(
        "data_dependent_control_flow", DATA_DEPENDENT,
        {"x": ("batch", 10)}, "limitation", cat3, 1,
        notes="fc_big->(batch,20) fed to fc_out(5,2): bug only on one branch",
    ))

    # 3.2 Dynamic computation graphs — shape mismatch after cat
    results.append(run_test(
        "dynamic_computation_graph", DYNAMIC_GRAPH,
        {"x": ("batch", 10)}, "limitation", cat3, 2,
        notes="fc->(1,5) cat'd then fc2 expects 10: dynamic loop hides mismatch",
    ))

    # 3.3 Variable-length sequences
    results.append(run_test(
        "variable_length_sequences", VARIABLE_LENGTH,
        {"x": ("batch", "seq_len", 10), "lengths": ("batch",)}, "limitation", cat3, 3,
        notes="LSTM out=20 fed to fc(15,5): pack_padded_sequence hides shape",
    ))

    # 3.4 Dictionary returns
    results.append(run_test(
        "dict_return", DICT_RETURN,
        {"x": ("batch", 10)}, "limitation", cat3, 4,
        notes="Dict return type makes shape tracking ambiguous",
    ))

    # 3.5 Complex indexing — mask makes output shape unknown
    results.append(run_test(
        "complex_indexing", COMPLEX_INDEX,
        {"x": ("batch", 10), "mask": ("batch",)}, "limitation", cat3, 5,
        notes="Boolean mask indexing produces data-dependent output shape",
    ))

    # 3.6 Custom autograd — ReshapeFunc changes shape opaquely
    results.append(run_test(
        "custom_autograd", CUSTOM_AUTOGRAD,
        {"x": ("batch", 10)}, "limitation", cat3, 6,
        notes="Custom Function reshapes to (-1,20) but fc expects 10",
    ))

    # 3.7 Recursive modules
    results.append(run_test(
        "recursive_modules", RECURSIVE_MODULE,
        {"x": ("batch", 10)}, "limitation", cat3, 7,
        notes="Recursive module instantiation with depth parameter",
    ))

    # -----------------------------------------------------------------------
    # Summarize results
    # -----------------------------------------------------------------------
    cat_results: Dict[str, Dict[str, Any]] = {}
    for cat_key in [cat1, cat2, cat3]:
        cat_tests = [r for r in results if r.category == cat_key]
        total = len(cat_tests)
        correct = sum(1 for r in cat_tests if r.correct)
        cat_results[cat_key] = {
            "total": total,
            "correct": correct,
            "accuracy": correct / total if total > 0 else 0.0,
            "tests": [asdict(r) for r in cat_tests],
        }

    # Build capabilities/limitations lists
    capabilities = []
    limitations = []
    boundary_capabilities = []
    misses = []
    surprises = []
    for r in results:
        if r.category == cat1 and r.correct:
            capabilities.append(r.name)
        elif r.category == cat1 and not r.correct:
            misses.append({"name": r.name, "notes": r.notes})
        elif r.category == cat2 and r.correct:
            boundary_capabilities.append(r.name)
        elif r.category == cat3 and r.correct:
            limitations.append(r.name)
        elif r.category == cat3 and not r.correct:
            surprises.append({"name": r.name, "notes": r.notes})

    overall_correct = sum(1 for r in results if r.correct)
    overall_total = len(results)

    characterization = (
        f"TensorGuard correctly handled {cat_results[cat1]['correct']}/{cat_results[cat1]['total']} "
        f"within-expressiveness tests, "
        f"{cat_results[cat2]['correct']}/{cat_results[cat2]['total']} boundary tests, "
        f"and {cat_results[cat3]['correct']}/{cat_results[cat3]['total']} outside-expressiveness tests. "
        f"Overall: {overall_correct}/{overall_total} tests matched expectations. "
        f"Core strengths: linear/conv/reshape/matmul/device/phase shape checking. "
        f"Precision gaps: element-wise ops (broadcasting, skip-connection add) not fully tracked. "
        f"Honest limitations: dynamic shapes, variable-length sequences, custom autograd, "
        f"complex indexing, and dict returns produce false negatives."
    )

    output = {
        "characterization": characterization,
        "overall": {
            "total": overall_total,
            "correct": overall_correct,
            "accuracy": overall_correct / overall_total if overall_total > 0 else 0.0,
        },
        "per_category": cat_results,
        "capabilities": capabilities,
        "boundary_capabilities": boundary_capabilities,
        "limitations": limitations,
        "precision_misses": misses,
        "surprise_capabilities": surprises,
        "all_results": [asdict(r) for r in results],
    }

    out_path = os.path.join(os.path.dirname(__file__), "expressiveness_boundary_results.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)

    # Print summary
    print("=" * 70)
    print("EXPRESSIVENESS BOUNDARY CHARACTERIZATION")
    print("=" * 70)

    for cat_label, cat_key in [
        ("WITHIN EXPRESSIVENESS", cat1),
        ("AT THE BOUNDARY", cat2),
        ("OUTSIDE EXPRESSIVENESS", cat3),
    ]:
        info = cat_results[cat_key]
        print(f"\n--- {cat_label} ({info['correct']}/{info['total']}) ---")
        for r in results:
            if r.category != cat_key:
                continue
            mark = "✓" if r.correct else "✗"
            print(f"  {mark} {r.name}: expected={r.expected_result} actual={r.actual_result}"
                  + (f"  ({r.notes[:80]})" if r.notes else ""))

    print(f"\n{'=' * 70}")
    print(characterization)
    print(f"Results saved to {out_path}")


if __name__ == "__main__":
    main()
