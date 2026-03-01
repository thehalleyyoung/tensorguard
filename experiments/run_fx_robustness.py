"""
Systematic torch.fx robustness characterization experiment.

Tests torch.fx tracing on a variety of model patterns beyond torchvision
to characterize failure modes and success rates across categories:
  a) Simple models (MLP, CNN, basic attention)
  b) Data-dependent control flow
  c) In-place operations
  d) Dynamic shapes
  e) Skip connections and residual patterns
  f) Common third-party patterns (dropout, layer norm, etc.)
"""

import json
import os
import sys
import time
import traceback
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

from src.fx_extractor import verify_module, trace_stats, fx_trace_to_graph


# ═══════════════════════════════════════════════════════════════════════════════
# Result data structures
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class ModelTestResult:
    name: str
    category: str
    can_trace: bool
    trace_time_ms: float
    verification_result: Optional[str] = None  # "safe", "unsafe", "error", None
    error_message: Optional[str] = None
    failure_mode: Optional[str] = None  # taxonomy classification
    num_layers: int = 0
    num_steps: int = 0


# ═══════════════════════════════════════════════════════════════════════════════
# Category A: Simple models (should always work)
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


class BasicAttention(nn.Module):
    """Simple scaled dot-product attention without data-dependent branching."""
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


class TwoLayerMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 10),
        )

    def forward(self, x):
        return self.net(x)


# ═══════════════════════════════════════════════════════════════════════════════
# Category B: Data-dependent control flow
# ═══════════════════════════════════════════════════════════════════════════════

class DataDependentBranch(nn.Module):
    """If-branch based on tensor values — untraceable by symbolic trace."""
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
    """Loop count depends on input — untraceable."""
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(64, 64)

    def forward(self, x):
        n = int(x.abs().sum().item()) % 5 + 1
        for _ in range(n):
            x = self.fc(x)
        return x


class DataDependentAssert(nn.Module):
    """Assert based on data — untraceable."""
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(32, 32)

    def forward(self, x):
        assert x.shape[0] > 0, "batch must be positive"
        if x.max() > 100:
            x = x / x.max()
        return self.fc(x)


class DataDependentEarlyReturn(nn.Module):
    """Early return based on data — untraceable."""
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(32, 32)

    def forward(self, x):
        if x.numel() == 0:
            return x
        return self.fc(x)


# ═══════════════════════════════════════════════════════════════════════════════
# Category C: In-place operations
# ═══════════════════════════════════════════════════════════════════════════════

class InPlaceReLU(nn.Module):
    """Uses inplace ReLU — may or may not trace."""
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(64, 64)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x):
        x = self.fc1(x)
        self.relu(x)
        return self.fc2(x)


class InPlaceAdd(nn.Module):
    """Uses += (in-place add)."""
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(64, 64)
        self.fc2 = nn.Linear(64, 64)

    def forward(self, x):
        out = self.fc1(x)
        out += self.fc2(x)
        return out


class InPlaceMul(nn.Module):
    """Uses *= (in-place multiply)."""
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(32, 32)

    def forward(self, x):
        x = self.fc(x)
        x *= 0.5
        return x


# ═══════════════════════════════════════════════════════════════════════════════
# Category D: Dynamic shapes
# ═══════════════════════════════════════════════════════════════════════════════

class DynamicReshape(nn.Module):
    """Uses x.size() in reshape — may cause issues."""
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(64, 64)

    def forward(self, x):
        b = x.size(0)
        x = self.fc(x)
        return x.view(b, -1)


class DynamicSlice(nn.Module):
    """Slicing with dynamic indices."""
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(64, 64)

    def forward(self, x):
        half = x.shape[-1] // 2
        left = x[..., :half]
        right = x[..., half:]
        return self.fc(left + right)


class AdaptiveModel(nn.Module):
    """Model that works with different spatial sizes."""
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 16, 3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(16, 10)

    def forward(self, x):
        x = self.conv(x)
        x = self.pool(x)
        x = x.flatten(1)
        return self.fc(x)


# ═══════════════════════════════════════════════════════════════════════════════
# Category E: Skip connections and residual patterns
# ═══════════════════════════════════════════════════════════════════════════════

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


class DenseBlock(nn.Module):
    """DenseNet-style: concatenate skip connections."""
    def __init__(self, in_ch=32, growth=16):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, growth, 3, padding=1)
        self.conv2 = nn.Conv2d(in_ch + growth, growth, 3, padding=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        out1 = self.relu(self.conv1(x))
        cat1 = torch.cat([x, out1], dim=1)
        out2 = self.relu(self.conv2(cat1))
        return torch.cat([cat1, out2], dim=1)


class UNetSkip(nn.Module):
    """U-Net style encoder-decoder with skip connection."""
    def __init__(self):
        super().__init__()
        self.enc = nn.Conv2d(3, 16, 3, padding=1)
        self.dec = nn.Conv2d(16, 16, 3, padding=1)
        self.final = nn.Conv2d(32, 3, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        enc_out = self.relu(self.enc(x))
        dec_out = self.relu(self.dec(enc_out))
        combined = torch.cat([enc_out, dec_out], dim=1)
        return self.final(combined)


class BottleneckResidual(nn.Module):
    """Bottleneck residual with 1x1 projection."""
    def __init__(self, in_ch=64, mid_ch=16):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, mid_ch, 1)
        self.conv2 = nn.Conv2d(mid_ch, mid_ch, 3, padding=1)
        self.conv3 = nn.Conv2d(mid_ch, in_ch, 1)
        self.bn1 = nn.BatchNorm2d(mid_ch)
        self.bn2 = nn.BatchNorm2d(mid_ch)
        self.bn3 = nn.BatchNorm2d(in_ch)
        self.relu = nn.ReLU()

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        return self.relu(out + identity)


# ═══════════════════════════════════════════════════════════════════════════════
# Category F: Common third-party patterns
# ═══════════════════════════════════════════════════════════════════════════════

class LayerNormModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(128, 128)
        self.ln = nn.LayerNorm(128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        return self.fc2(self.ln(self.fc1(x)))


class DropoutModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(128, 64)
        self.drop = nn.Dropout(0.5)
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x):
        return self.fc2(self.drop(F.relu(self.fc1(x))))


class GroupNormModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 32, 3, padding=1)
        self.gn = nn.GroupNorm(8, 32)
        self.relu = nn.ReLU()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(32, 10)

    def forward(self, x):
        x = self.relu(self.gn(self.conv(x)))
        x = self.pool(x).flatten(1)
        return self.fc(x)


class EmbeddingModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(1000, 64)
        self.fc = nn.Linear(64, 10)

    def forward(self, x):
        return self.fc(self.emb(x).mean(dim=1))


class MultiHeadAttnModule(nn.Module):
    """Uses nn.MultiheadAttention — known to sometimes fail fx tracing."""
    def __init__(self):
        super().__init__()
        self.attn = nn.MultiheadAttention(64, 4, batch_first=True)
        self.fc = nn.Linear(64, 10)

    def forward(self, x):
        out, _ = self.attn(x, x, x)
        return self.fc(out.mean(dim=1))


class TransformerBlock(nn.Module):
    """Single transformer encoder layer."""
    def __init__(self):
        super().__init__()
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=64, nhead=4, dim_feedforward=128, batch_first=True
        )
        self.fc = nn.Linear(64, 10)

    def forward(self, x):
        x = self.encoder_layer(x)
        return self.fc(x.mean(dim=1))


# ═══════════════════════════════════════════════════════════════════════════════
# Model registry
# ═══════════════════════════════════════════════════════════════════════════════

def build_model_registry():
    """Build the full test registry: (name, category, factory, input_shapes)."""
    registry = []

    # Category A: Simple models
    registry.extend([
        ("SimpleMLP", "simple", lambda: SimpleMLP(),
         {"x": ("batch", 784)}),
        ("SimpleCNN", "simple", lambda: SimpleCNN(),
         {"x": ("batch", 3, 32, 32)}),
        ("BasicAttention", "simple", lambda: BasicAttention(64),
         {"x": ("batch", 8, 64)}),
        ("TwoLayerMLP", "simple", lambda: TwoLayerMLP(),
         {"x": ("batch", 128)}),
    ])

    # Category B: Data-dependent control flow
    registry.extend([
        ("DataDependentBranch", "data_dependent_control_flow",
         lambda: DataDependentBranch(), {"x": ("batch", 64)}),
        ("DataDependentLoop", "data_dependent_control_flow",
         lambda: DataDependentLoop(), {"x": ("batch", 64)}),
        ("DataDependentAssert", "data_dependent_control_flow",
         lambda: DataDependentAssert(), {"x": ("batch", 32)}),
        ("DataDependentEarlyReturn", "data_dependent_control_flow",
         lambda: DataDependentEarlyReturn(), {"x": ("batch", 32)}),
    ])

    # Category C: In-place operations
    registry.extend([
        ("InPlaceReLU", "inplace_operations",
         lambda: InPlaceReLU(), {"x": ("batch", 64)}),
        ("InPlaceAdd", "inplace_operations",
         lambda: InPlaceAdd(), {"x": ("batch", 64)}),
        ("InPlaceMul", "inplace_operations",
         lambda: InPlaceMul(), {"x": ("batch", 32)}),
    ])

    # Category D: Dynamic shapes
    registry.extend([
        ("DynamicReshape", "dynamic_shapes",
         lambda: DynamicReshape(), {"x": ("batch", 64)}),
        ("DynamicSlice", "dynamic_shapes",
         lambda: DynamicSlice(), {"x": ("batch", 64)}),
        ("AdaptiveModel", "dynamic_shapes",
         lambda: AdaptiveModel(), {"x": ("batch", 3, 32, 32)}),
    ])

    # Category E: Skip connections and residuals
    registry.extend([
        ("ResidualBlock", "skip_connections",
         lambda: ResidualBlock(64), {"x": ("batch", 64, 8, 8)}),
        ("DenseBlock", "skip_connections",
         lambda: DenseBlock(32, 16), {"x": ("batch", 32, 8, 8)}),
        ("UNetSkip", "skip_connections",
         lambda: UNetSkip(), {"x": ("batch", 3, 32, 32)}),
        ("BottleneckResidual", "skip_connections",
         lambda: BottleneckResidual(64, 16), {"x": ("batch", 64, 8, 8)}),
    ])

    # Category F: Common third-party patterns
    registry.extend([
        ("LayerNormModel", "third_party_patterns",
         lambda: LayerNormModel(), {"x": ("batch", 128)}),
        ("DropoutModel", "third_party_patterns",
         lambda: DropoutModel(), {"x": ("batch", 128)}),
        ("GroupNormModel", "third_party_patterns",
         lambda: GroupNormModel(), {"x": ("batch", 3, 16, 16)}),
        ("EmbeddingModel", "third_party_patterns",
         lambda: EmbeddingModel(), {"x": ("batch", 10)}),
        ("MultiHeadAttnModule", "third_party_patterns",
         lambda: MultiHeadAttnModule(), {"x": ("batch", 8, 64)}),
        ("TransformerBlock", "third_party_patterns",
         lambda: TransformerBlock(), {"x": ("batch", 8, 64)}),
    ])

    return registry


# ═══════════════════════════════════════════════════════════════════════════════
# Failure mode taxonomy
# ═══════════════════════════════════════════════════════════════════════════════

def classify_failure(error_message: str) -> str:
    """Classify a trace/verification error into a failure mode taxonomy."""
    if not error_message:
        return "unknown"
    err = error_message.lower()
    if "control flow" in err or "conditional" in err or "if" in err:
        return "data_dependent_control_flow"
    if "item" in err or ".item()" in err:
        return "data_dependent_item_call"
    if "is not defined" in err or "tracer" in err or "proxy" in err:
        return "symbolic_trace_proxy_error"
    if "inplace" in err or "in-place" in err or "in_place" in err:
        return "inplace_operation"
    if "dynamic" in err or "shape" in err:
        return "dynamic_shape_error"
    if "autograd" in err or "custom" in err:
        return "custom_autograd"
    if "not supported" in err or "unsupported" in err:
        return "unsupported_operation"
    if "assert" in err:
        return "assertion_in_forward"
    if "conversion" in err or "fx graph" in err:
        return "graph_conversion_error"
    return "other"


# ═══════════════════════════════════════════════════════════════════════════════
# Main experiment runner
# ═══════════════════════════════════════════════════════════════════════════════

def run_experiment():
    """Run the full fx robustness characterization experiment."""
    if not HAS_TORCH:
        print("ERROR: PyTorch is required to run this experiment.")
        sys.exit(1)

    registry = build_model_registry()
    results: List[ModelTestResult] = []

    print(f"torch.fx Robustness Characterization Experiment")
    print(f"PyTorch version: {torch.__version__}")
    print(f"Total models to test: {len(registry)}")
    print("=" * 70)

    for name, category, factory, input_shapes in registry:
        print(f"\n[{category}] {name}...", end=" ")

        try:
            model = factory()
        except Exception as e:
            result = ModelTestResult(
                name=name, category=category,
                can_trace=False, trace_time_ms=0.0,
                error_message=f"Instantiation failed: {e}",
                failure_mode="instantiation_error",
            )
            results.append(result)
            print(f"INSTANTIATION ERROR: {e}")
            continue

        # Attempt tracing
        t0 = time.monotonic()
        try:
            model.eval()
            traced = torch.fx.symbolic_trace(model)
            trace_time = (time.monotonic() - t0) * 1000
            can_trace = True
            trace_error = None
        except Exception as e:
            trace_time = (time.monotonic() - t0) * 1000
            can_trace = False
            trace_error = str(e)

        if not can_trace:
            failure_mode = classify_failure(trace_error)
            result = ModelTestResult(
                name=name, category=category,
                can_trace=False, trace_time_ms=round(trace_time, 2),
                verification_result="error",
                error_message=f"Trace failed: {trace_error}",
                failure_mode=failure_mode,
            )
            results.append(result)
            print(f"TRACE FAILED ({failure_mode}): {trace_error[:80]}")
            continue

        # Get trace stats
        try:
            graph = fx_trace_to_graph(traced, class_name=name)
            num_layers = len(graph.layers)
            num_steps = graph.num_steps
        except Exception as e:
            result = ModelTestResult(
                name=name, category=category,
                can_trace=True, trace_time_ms=round(trace_time, 2),
                verification_result="error",
                error_message=f"Graph conversion failed: {e}",
                failure_mode="graph_conversion_error",
                num_layers=0, num_steps=0,
            )
            results.append(result)
            print(f"CONVERSION ERROR: {e}")
            continue

        # Attempt verification
        try:
            vresult = verify_module(model, input_shapes=input_shapes)
            if vresult.safe:
                verification_result = "safe"
            elif vresult.errors and "tracing failed" in str(vresult.errors):
                verification_result = "error"
            else:
                verification_result = "unsafe"
            verr = None
        except Exception as e:
            verification_result = "error"
            verr = str(e)

        result = ModelTestResult(
            name=name, category=category,
            can_trace=can_trace, trace_time_ms=round(trace_time, 2),
            verification_result=verification_result,
            error_message=verr,
            failure_mode=classify_failure(verr) if verr else None,
            num_layers=num_layers, num_steps=num_steps,
        )
        results.append(result)
        print(f"OK (trace={trace_time:.1f}ms, layers={num_layers}, "
              f"steps={num_steps}, result={verification_result})")

    # ─── Build summary ────────────────────────────────────────────────────────
    categories = {}
    failure_modes = {}
    total = len(results)
    total_traceable = sum(1 for r in results if r.can_trace)

    for r in results:
        cat = r.category
        if cat not in categories:
            categories[cat] = {"total": 0, "traceable": 0, "safe": 0,
                               "unsafe": 0, "error": 0}
        categories[cat]["total"] += 1
        if r.can_trace:
            categories[cat]["traceable"] += 1
        if r.verification_result == "safe":
            categories[cat]["safe"] += 1
        elif r.verification_result == "unsafe":
            categories[cat]["unsafe"] += 1
        elif r.verification_result == "error":
            categories[cat]["error"] += 1

        if r.failure_mode:
            failure_modes[r.failure_mode] = failure_modes.get(r.failure_mode, 0) + 1

    # Per-category success rate
    category_success_rates = {}
    for cat, stats in categories.items():
        rate = stats["traceable"] / stats["total"] if stats["total"] > 0 else 0
        category_success_rates[cat] = round(rate, 4)

    summary = {
        "total_models_tested": total,
        "total_traceable": total_traceable,
        "overall_trace_success_rate": round(total_traceable / total, 4) if total > 0 else 0,
        "category_breakdown": categories,
        "category_success_rates": category_success_rates,
        "failure_modes_taxonomy": failure_modes,
        "pytorch_version": torch.__version__,
    }

    output = {
        "summary": summary,
        "results": [asdict(r) for r in results],
    }

    # Print summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Total models tested: {total}")
    print(f"Traceable: {total_traceable}/{total} "
          f"({summary['overall_trace_success_rate']*100:.1f}%)")
    print()
    print("Per-category trace success rates:")
    for cat, rate in category_success_rates.items():
        stats = categories[cat]
        print(f"  {cat}: {stats['traceable']}/{stats['total']} "
              f"({rate*100:.1f}%)")
    print()
    if failure_modes:
        print("Failure modes taxonomy:")
        for mode, count in sorted(failure_modes.items(), key=lambda x: -x[1]):
            print(f"  {mode}: {count}")

    # Save results
    results_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(results_dir, exist_ok=True)
    out_path = os.path.join(results_dir, "fx_robustness_results.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {out_path}")

    return output


if __name__ == "__main__":
    run_experiment()
