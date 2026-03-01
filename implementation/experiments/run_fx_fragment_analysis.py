"""
Experiment: Verifiable Fragment Analysis.

Tests traceability on a variety of models (simple MLP, CNN, ResNet,
Transformer, models with data-dependent control flow) and measures what
percentage of torchvision models lie within TensorGuard's verifiable fragment.

Addresses reviewer critique #5: formal syntactic characterization of the
verifiable fragment.
"""

import json
import os
import sys
import time
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    import torchvision.models as tv_models
    HAS_TORCHVISION = True
except ImportError:
    HAS_TORCHVISION = False

from src.verifiable_fragment import (
    check_traceability,
    extract_grammar,
    TraceabilityReport,
    UnsupportedCategory,
    SUPPORTED_LAYER_TYPES,
    SUPPORTED_TENSOR_METHODS,
)


# ═══════════════════════════════════════════════════════════════════════════════
# Test models: In-fragment (should be traceable)
# ═══════════════════════════════════════════════════════════════════════════════

class SimpleMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 256)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))


class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(16, 10)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        x = x.flatten(1)
        return self.fc(x)


class ResidualBlock(nn.Module):
    def __init__(self, dim=64):
        super().__init__()
        self.conv1 = nn.Conv2d(dim, dim, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(dim)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(dim, dim, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(dim)

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + identity)


class BasicAttention(nn.Module):
    def __init__(self, dim=64):
        super().__init__()
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        self.scale = dim ** -0.5

    def forward(self, x):
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = torch.softmax(attn, dim=-1)
        out = torch.matmul(attn, v)
        return self.out_proj(out)


class MultiBranch(nn.Module):
    def __init__(self):
        super().__init__()
        self.branch1 = nn.Linear(128, 64)
        self.branch2 = nn.Linear(128, 64)
        self.out = nn.Linear(128, 10)

    def forward(self, x):
        a = self.branch1(x)
        b = self.branch2(x)
        combined = torch.cat([a, b], dim=-1)
        return self.out(combined)


class SequentialModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 10),
        )

    def forward(self, x):
        return self.net(x)


class ConvEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.pool = nn.AdaptiveAvgPool2d(4)
        self.fc = nn.Linear(64 * 4 * 4, 128)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        x = x.flatten(1)
        return self.fc(x)


# ═══════════════════════════════════════════════════════════════════════════════
# Test models: Outside fragment (should NOT be traceable)
# ═══════════════════════════════════════════════════════════════════════════════

class DataDependentBranch(nn.Module):
    """if-branch on tensor value — outside V_TG."""
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(64, 64)
        self.fc2 = nn.Linear(64, 64)

    def forward(self, x):
        x = self.fc1(x)
        if x.sum() > 0:
            x = self.fc2(x)
        return x


class DataDependentLoop(nn.Module):
    """Loop count depends on tensor — outside V_TG."""
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(64, 64)

    def forward(self, x):
        n = int(x.abs().sum().item()) % 5 + 1
        for _ in range(n):
            x = self.fc(x)
        return x


class TensorToScalar(nn.Module):
    """.item() call — outside V_TG."""
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(32, 32)

    def forward(self, x):
        scale = x.max().item()
        return self.fc(x) * scale


class DataDependentEarlyReturn(nn.Module):
    """Early return on tensor condition — outside V_TG."""
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(32, 32)

    def forward(self, x):
        if x.numel() == 0:
            return x
        return self.fc(x)


class CustomAutogradModel(nn.Module):
    """Custom autograd function — outside V_TG."""
    class ScaleGrad(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x, scale):
            ctx.save_for_backward(torch.tensor(scale))
            return x

        @staticmethod
        def backward(ctx, grad):
            scale, = ctx.saved_tensors
            return grad * scale, None

    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(32, 32)

    def forward(self, x):
        x = self.ScaleGrad.apply(x, 0.1)
        return self.fc(x)


class DynamicAssertModel(nn.Module):
    """Assert in forward — outside V_TG."""
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(64, 64)

    def forward(self, x):
        assert x.shape[-1] == 64
        return self.fc(x)


# ═══════════════════════════════════════════════════════════════════════════════
# Model registry
# ═══════════════════════════════════════════════════════════════════════════════

def build_model_registry():
    """Build the full test registry: (name, category, factory, input_shapes, expect_in_fragment)."""
    registry = []

    # In-fragment models
    registry.extend([
        ("SimpleMLP", "in_fragment", lambda: SimpleMLP(),
         {"x": ("batch", 784)}, True),
        ("SimpleCNN", "in_fragment", lambda: SimpleCNN(),
         {"x": ("batch", 3, 32, 32)}, True),
        ("ResidualBlock", "in_fragment", lambda: ResidualBlock(64),
         {"x": ("batch", 64, 8, 8)}, True),
        ("BasicAttention", "in_fragment", lambda: BasicAttention(64),
         {"x": ("batch", 8, 64)}, True),
        ("MultiBranch", "in_fragment", lambda: MultiBranch(),
         {"x": ("batch", 128)}, True),
        ("SequentialModel", "in_fragment", lambda: SequentialModel(),
         {"x": ("batch", 256)}, True),
        ("ConvEncoder", "in_fragment", lambda: ConvEncoder(),
         {"x": ("batch", 3, 224, 224)}, True),
    ])

    # Outside-fragment models
    registry.extend([
        ("DataDependentBranch", "outside_fragment", lambda: DataDependentBranch(),
         {"x": ("batch", 64)}, False),
        ("DataDependentLoop", "outside_fragment", lambda: DataDependentLoop(),
         {"x": ("batch", 64)}, False),
        ("TensorToScalar", "outside_fragment", lambda: TensorToScalar(),
         {"x": ("batch", 32)}, False),
        ("DataDependentEarlyReturn", "outside_fragment", lambda: DataDependentEarlyReturn(),
         {"x": ("batch", 32)}, False),
        ("CustomAutogradModel", "outside_fragment", lambda: CustomAutogradModel(),
         {"x": ("batch", 32)}, None),  # may or may not trace; fx handles some
        ("DynamicAssertModel", "outside_fragment", lambda: DynamicAssertModel(),
         {"x": ("batch", 64)}, False),
    ])

    # Torchvision models (all expected to be in fragment)
    if HAS_TORCHVISION:
        tv_list = [
            ("tv/ResNet-18", lambda: tv_models.resnet18(weights=None),
             {"x": ("batch", 3, 224, 224)}),
            ("tv/ResNet-50", lambda: tv_models.resnet50(weights=None),
             {"x": ("batch", 3, 224, 224)}),
            ("tv/VGG-11", lambda: tv_models.vgg11(weights=None),
             {"x": ("batch", 3, 224, 224)}),
            ("tv/VGG-16", lambda: tv_models.vgg16(weights=None),
             {"x": ("batch", 3, 224, 224)}),
            ("tv/MobileNetV2", lambda: tv_models.mobilenet_v2(weights=None),
             {"x": ("batch", 3, 224, 224)}),
            ("tv/DenseNet-121", lambda: tv_models.densenet121(weights=None),
             {"x": ("batch", 3, 224, 224)}),
            ("tv/SqueezeNet-1.0", lambda: tv_models.squeezenet1_0(weights=None),
             {"x": ("batch", 3, 224, 224)}),
            ("tv/ShuffleNetV2", lambda: tv_models.shufflenet_v2_x1_0(weights=None),
             {"x": ("batch", 3, 224, 224)}),
            ("tv/EfficientNet-B0", lambda: tv_models.efficientnet_b0(weights=None),
             {"x": ("batch", 3, 224, 224)}),
            ("tv/RegNet-Y-400MF", lambda: tv_models.regnet_y_400mf(weights=None),
             {"x": ("batch", 3, 224, 224)}),
            ("tv/MNASNet-1.0", lambda: tv_models.mnasnet1_0(weights=None),
             {"x": ("batch", 3, 224, 224)}),
            ("tv/GoogLeNet", lambda: tv_models.googlenet(weights=None),
             {"x": ("batch", 3, 224, 224)}),
            ("tv/AlexNet", lambda: tv_models.alexnet(weights=None),
             {"x": ("batch", 3, 224, 224)}),
            ("tv/WideResNet-50-2", lambda: tv_models.wide_resnet50_2(weights=None),
             {"x": ("batch", 3, 224, 224)}),
            ("tv/ResNeXt-50", lambda: tv_models.resnext50_32x4d(weights=None),
             {"x": ("batch", 3, 224, 224)}),
            ("tv/ConvNeXt-Tiny", lambda: tv_models.convnext_tiny(weights=None),
             {"x": ("batch", 3, 224, 224)}),
        ]
        for name, factory, shapes in tv_list:
            registry.append((name, "torchvision", factory, shapes, True))

    return registry


# ═══════════════════════════════════════════════════════════════════════════════
# Main experiment
# ═══════════════════════════════════════════════════════════════════════════════

def run_experiment():
    """Run the verifiable fragment analysis experiment."""
    if not HAS_TORCH:
        print("ERROR: PyTorch is required.")
        sys.exit(1)

    registry = build_model_registry()
    results = []
    summary = {
        "total": 0,
        "in_fragment": 0,
        "outside_fragment": 0,
        "correct_predictions": 0,
        "torchvision_total": 0,
        "torchvision_in_fragment": 0,
        "custom_in_fragment_total": 0,
        "custom_in_fragment_correct": 0,
        "custom_outside_total": 0,
        "custom_outside_correct": 0,
        "categories": {},
        "unsupported_category_counts": {},
    }

    print("Verifiable Fragment Analysis Experiment")
    print(f"PyTorch version: {torch.__version__}")
    print(f"Torchvision available: {HAS_TORCHVISION}")
    print(f"Supported layer types: {len(SUPPORTED_LAYER_TYPES)}")
    print(f"Supported tensor methods: {len(SUPPORTED_TENSOR_METHODS)}")
    print(f"Total models to analyze: {len(registry)}")
    print("=" * 70)

    for name, category, factory, input_shapes, expect_in_fragment in registry:
        summary["total"] += 1
        print(f"\n[{category}] {name}...", end=" ", flush=True)

        try:
            model = factory()
        except Exception as e:
            print(f"INSTANTIATION ERROR: {e}")
            results.append({
                "name": name, "category": category,
                "status": "instantiation_error", "error": str(e),
            })
            continue

        t0 = time.monotonic()
        report = check_traceability(model)
        elapsed = (time.monotonic() - t0) * 1000

        in_frag = report.in_verifiable_fragment
        correct = (expect_in_fragment is None) or (in_frag == expect_in_fragment)

        if correct:
            summary["correct_predictions"] += 1

        if in_frag:
            summary["in_fragment"] += 1
        else:
            summary["outside_fragment"] += 1

        # Track by category
        if category not in summary["categories"]:
            summary["categories"][category] = {
                "total": 0, "in_fragment": 0, "outside_fragment": 0,
            }
        summary["categories"][category]["total"] += 1
        if in_frag:
            summary["categories"][category]["in_fragment"] += 1
        else:
            summary["categories"][category]["outside_fragment"] += 1

        # Track torchvision separately
        if category == "torchvision":
            summary["torchvision_total"] += 1
            if in_frag:
                summary["torchvision_in_fragment"] += 1
        elif category == "in_fragment":
            summary["custom_in_fragment_total"] += 1
            if correct:
                summary["custom_in_fragment_correct"] += 1
        elif category == "outside_fragment":
            summary["custom_outside_total"] += 1
            if correct:
                summary["custom_outside_correct"] += 1

        # Track unsupported construct categories
        for uc in report.unsupported_constructs:
            cat_name = uc.category.name
            summary["unsupported_category_counts"][cat_name] = (
                summary["unsupported_category_counts"].get(cat_name, 0) + 1
            )

        status = "IN_FRAGMENT" if in_frag else "OUTSIDE"
        correctness = "CORRECT" if correct else "WRONG"
        print(f"{status} ({correctness}) [{elapsed:.1f}ms]")

        if not in_frag:
            for uc in report.blocking_issues[:2]:
                print(f"    [{uc.category.name}] {uc.description[:80]}")

        # Extract grammar for in-fragment models
        grammar_info = None
        if in_frag:
            grammar = extract_grammar(model)
            if grammar:
                grammar_info = {
                    "num_layer_decls": len(grammar.layers),
                    "num_forward_stmts": len(grammar.forward_stmts),
                    "layer_types": list({ld.layer_type for ld in grammar.layers}),
                }

        rec = {
            "name": name,
            "category": category,
            "expected_in_fragment": expect_in_fragment,
            "actual_in_fragment": in_frag,
            "correct": correct,
            "fx_traceable": report.fx_traceable,
            "fx_trace_error": report.fx_trace_error,
            "num_parameters": report.num_parameters,
            "num_submodules": report.num_submodules,
            "num_supported_layers": len(report.supported_layers),
            "num_unsupported_layers": len(report.unsupported_layers),
            "unsupported_constructs": [
                {"category": uc.category.name, "description": uc.description[:200],
                 "severity": uc.severity}
                for uc in report.unsupported_constructs
            ],
            "analysis_time_ms": round(elapsed, 1),
        }
        if grammar_info:
            rec["grammar"] = grammar_info
        results.append(rec)

    # ─── Summary ──────────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"Total models analyzed: {summary['total']}")
    print(f"In verifiable fragment: {summary['in_fragment']}/{summary['total']} "
          f"({100*summary['in_fragment']/max(summary['total'],1):.1f}%)")
    print(f"Outside fragment: {summary['outside_fragment']}/{summary['total']}")
    print(f"Correct predictions: {summary['correct_predictions']}/{summary['total']}")

    if summary["torchvision_total"] > 0:
        tv_pct = 100 * summary["torchvision_in_fragment"] / summary["torchvision_total"]
        print(f"\nTorchvision coverage: {summary['torchvision_in_fragment']}"
              f"/{summary['torchvision_total']} ({tv_pct:.1f}%)")

    print(f"\nPer-category breakdown:")
    for cat, stats in summary["categories"].items():
        pct = 100 * stats["in_fragment"] / max(stats["total"], 1)
        print(f"  {cat}: {stats['in_fragment']}/{stats['total']} in fragment ({pct:.0f}%)")

    if summary["unsupported_category_counts"]:
        print(f"\nUnsupported construct taxonomy:")
        for cat, count in sorted(summary["unsupported_category_counts"].items(),
                                  key=lambda x: -x[1]):
            print(f"  {cat}: {count}")

    # Compute formal fragment coverage metric
    total_coverage = summary["in_fragment"] / max(summary["total"], 1)
    tv_coverage = (summary["torchvision_in_fragment"] /
                   max(summary["torchvision_total"], 1))

    summary["fragment_coverage"] = round(total_coverage, 4)
    summary["torchvision_coverage"] = round(tv_coverage, 4)
    summary["prediction_accuracy"] = round(
        summary["correct_predictions"] / max(summary["total"], 1), 4)
    summary["results"] = results

    # Grammar specification metadata
    summary["grammar_spec"] = {
        "supported_layer_types": sorted(SUPPORTED_LAYER_TYPES),
        "supported_tensor_methods": sorted(SUPPORTED_TENSOR_METHODS),
        "num_supported_layer_types": len(SUPPORTED_LAYER_TYPES),
        "num_supported_tensor_methods": len(SUPPORTED_TENSOR_METHODS),
        "excluded_constructs": [
            "Data-dependent control flow (if/while on tensor values)",
            "Data-dependent iteration (for with tensor-derived range)",
            "Tensor-to-scalar conversions (.item(), .tolist(), .numpy())",
            "Custom autograd functions (torch.autograd.Function)",
            "Dynamic assertions (assert with tensor expressions)",
            "torch.jit.script modules",
            "Opaque external library calls",
        ],
    }

    out_path = os.path.join(os.path.dirname(__file__),
                            "fx_fragment_analysis_results.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nResults saved to {out_path}")

    return summary


if __name__ == "__main__":
    run_experiment()
