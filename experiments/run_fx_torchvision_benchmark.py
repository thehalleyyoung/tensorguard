"""
Benchmark: verify_module on torchvision models via torch.fx.

Demonstrates TensorGuard's ability to verify *arbitrary* PyTorch nn.Module
instances — not just source code — by tracing through torch.fx.
"""

import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import torch.nn as nn

try:
    import torchvision.models as tv_models
    HAS_TORCHVISION = True
except ImportError:
    HAS_TORCHVISION = False

from src.fx_extractor import verify_module, trace_stats


def build_model_registry():
    """Build registry of models to test, mixing torchvision and custom."""
    models = []

    # --- Torchvision models ---
    if HAS_TORCHVISION:
        tv_list = [
            ("ResNet-18", lambda: tv_models.resnet18(weights=None),
             {"x": ("batch", 3, 224, 224)}, True),
            ("ResNet-50", lambda: tv_models.resnet50(weights=None),
             {"x": ("batch", 3, 224, 224)}, True),
            ("VGG-11", lambda: tv_models.vgg11(weights=None),
             {"x": ("batch", 3, 224, 224)}, True),
            ("VGG-16", lambda: tv_models.vgg16(weights=None),
             {"x": ("batch", 3, 224, 224)}, True),
            ("MobileNetV2", lambda: tv_models.mobilenet_v2(weights=None),
             {"x": ("batch", 3, 224, 224)}, True),
            ("DenseNet-121", lambda: tv_models.densenet121(weights=None),
             {"x": ("batch", 3, 224, 224)}, True),
            ("SqueezeNet-1.0", lambda: tv_models.squeezenet1_0(weights=None),
             {"x": ("batch", 3, 224, 224)}, True),
            ("ShuffleNetV2-x1.0", lambda: tv_models.shufflenet_v2_x1_0(weights=None),
             {"x": ("batch", 3, 224, 224)}, True),
            ("EfficientNet-B0", lambda: tv_models.efficientnet_b0(weights=None),
             {"x": ("batch", 3, 224, 224)}, True),
            ("RegNet-Y-400MF", lambda: tv_models.regnet_y_400mf(weights=None),
             {"x": ("batch", 3, 224, 224)}, True),
            ("MNASNet-1.0", lambda: tv_models.mnasnet1_0(weights=None),
             {"x": ("batch", 3, 224, 224)}, True),
            ("GoogLeNet", lambda: tv_models.googlenet(weights=None),
             {"x": ("batch", 3, 224, 224)}, True),
            ("AlexNet", lambda: tv_models.alexnet(weights=None),
             {"x": ("batch", 3, 224, 224)}, True),
            ("WideResNet-50-2", lambda: tv_models.wide_resnet50_2(weights=None),
             {"x": ("batch", 3, 224, 224)}, True),
            ("ResNeXt-50", lambda: tv_models.resnext50_32x4d(weights=None),
             {"x": ("batch", 3, 224, 224)}, True),
            ("ConvNeXt-Tiny", lambda: tv_models.convnext_tiny(weights=None),
             {"x": ("batch", 3, 224, 224)}, True),
        ]
        for name, factory, shapes, expected_safe in tv_list:
            models.append((name, factory, shapes, expected_safe, "torchvision"))

    # --- Custom buggy models ---
    class BuggyResidual(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
            self.conv2 = nn.Conv2d(16, 32, 3, padding=1)  # 32 != 16

        def forward(self, x):
            identity = x
            out = self.conv1(x)
            out = self.conv2(out)
            return out + identity  # BUG: 32 channels + 3 channels

    models.append((
        "BuggyResidual-ChannelMismatch", lambda: BuggyResidual(),
        {"x": ("batch", 3, 32, 32)}, False, "custom-buggy"
    ))

    class BuggyDeepMLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleList([
                nn.Linear(256, 256) for _ in range(9)
            ])
            self.layers.append(nn.Linear(128, 10))  # BUG at layer 10

        def forward(self, x):
            for layer in self.layers:
                x = layer(x)
            return x

    models.append((
        "BuggyDeepMLP-Layer10", lambda: BuggyDeepMLP(),
        {"x": ("batch", 256)}, False, "custom-buggy"
    ))

    class BuggyBroadcastConcat(nn.Module):
        def __init__(self):
            super().__init__()
            self.branch_a = nn.Linear(512, 64)
            self.branch_b = nn.Linear(512, 128)
            self.out = nn.Linear(192, 10)

        def forward(self, x):
            a = self.branch_a(x)
            b = self.branch_b(x)
            return self.out(a + b)  # BUG: 64 + 128 broadcast mismatch

    models.append((
        "BuggyBroadcastConcat", lambda: BuggyBroadcastConcat(),
        {"x": ("batch", 512)}, False, "custom-buggy"
    ))

    # Custom safe models
    class SafeEncoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(3, 64, 7, stride=2, padding=3)
            self.bn1 = nn.BatchNorm2d(64)
            self.relu = nn.ReLU()
            self.pool = nn.AdaptiveAvgPool2d(4)
            self.fc = nn.Linear(64 * 4 * 4, 128)

        def forward(self, x):
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.pool(x)
            x = x.flatten(1)
            x = self.fc(x)
            return x

    models.append((
        "SafeEncoder", lambda: SafeEncoder(),
        {"x": ("batch", 3, 224, 224)}, True, "custom-safe"
    ))

    class SafeMultiBranch(nn.Module):
        def __init__(self):
            super().__init__()
            self.branch1 = nn.Linear(256, 64)
            self.branch2 = nn.Linear(256, 64)
            self.out = nn.Linear(128, 10)

        def forward(self, x):
            a = self.branch1(x)
            b = self.branch2(x)
            combined = torch.cat([a, b], dim=-1)
            return self.out(combined)

    models.append((
        "SafeMultiBranch", lambda: SafeMultiBranch(),
        {"x": ("batch", 256)}, True, "custom-safe"
    ))

    return models


def run_benchmark():
    models = build_model_registry()
    results = []
    summary = {
        "total": 0,
        "traceable": 0,
        "correct_verdict": 0,
        "safe_correct": 0,
        "unsafe_correct": 0,
        "trace_failures": 0,
        "wrong_verdicts": 0,
        "torchvision_models": 0,
        "torchvision_correct": 0,
    }

    for name, factory, input_shapes, expected_safe, source in models:
        summary["total"] += 1
        if source == "torchvision":
            summary["torchvision_models"] += 1

        print(f"\n{'='*60}")
        print(f"Model: {name} (source: {source})")
        print(f"Expected: {'SAFE' if expected_safe else 'UNSAFE'}")

        try:
            model = factory()
        except Exception as e:
            print(f"  SKIP: Failed to instantiate: {e}")
            results.append({
                "name": name, "source": source,
                "status": "instantiation_error", "error": str(e)
            })
            continue

        # Trace stats
        stats = trace_stats(model)
        if not stats.traceable:
            print(f"  NOT TRACEABLE: {stats.trace_error}")
            summary["trace_failures"] += 1
            results.append({
                "name": name, "source": source,
                "status": "trace_error", "error": stats.trace_error
            })
            continue

        summary["traceable"] += 1
        print(f"  Traced: {stats.num_layers} layers, {stats.num_steps} steps")

        # Verify
        t0 = time.monotonic()
        result = verify_module(model, input_shapes=input_shapes)
        elapsed = (time.monotonic() - t0) * 1000

        actual_safe = result.safe
        correct = actual_safe == expected_safe

        if correct:
            summary["correct_verdict"] += 1
            if expected_safe:
                summary["safe_correct"] += 1
            else:
                summary["unsafe_correct"] += 1
            if source == "torchvision":
                summary["torchvision_correct"] += 1
        else:
            summary["wrong_verdicts"] += 1

        status = "CORRECT" if correct else "WRONG"
        verdict = "SAFE" if actual_safe else "UNSAFE"
        print(f"  Verdict: {verdict} ({status}) [{elapsed:.1f}ms]")

        if not actual_safe and result.counterexample:
            viols = result.counterexample.violations[:2]
            for v in viols:
                print(f"    Violation: {v.message[:100]}")

        if result.errors:
            print(f"  Errors: {result.errors[:2]}")

        rec = {
            "name": name,
            "source": source,
            "expected_safe": expected_safe,
            "actual_safe": actual_safe,
            "correct": correct,
            "time_ms": round(elapsed, 1),
            "num_layers": stats.num_layers,
            "num_steps": stats.num_steps,
            "layer_kinds": stats.layer_kinds,
        }
        if not actual_safe and result.counterexample:
            rec["violations"] = [
                v.message[:200] for v in result.counterexample.violations[:3]
            ]
        if actual_safe and result.certificate:
            rec["certificate"] = True
        results.append(rec)

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"Total models: {summary['total']}")
    print(f"Traceable: {summary['traceable']}/{summary['total']}")
    print(f"Correct verdicts: {summary['correct_verdict']}/{summary['traceable']}")
    print(f"  Safe correct: {summary['safe_correct']}")
    print(f"  Unsafe correct: {summary['unsafe_correct']}")
    print(f"  Wrong verdicts: {summary['wrong_verdicts']}")
    if summary['torchvision_models'] > 0:
        print(f"Torchvision: {summary['torchvision_correct']}/{summary['torchvision_models']} correct")

    accuracy = summary["correct_verdict"] / max(summary["traceable"], 1)
    summary["accuracy"] = round(accuracy, 4)
    summary["results"] = results

    out_path = os.path.join(os.path.dirname(__file__),
                            "fx_torchvision_benchmark_results.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nResults saved to {out_path}")

    return summary


if __name__ == "__main__":
    run_benchmark()
