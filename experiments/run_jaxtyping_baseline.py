#!/usr/bin/env python3
"""
jaxtyping + beartype baseline comparison for TensorGuard.

Compares TensorGuard (static, annotation-free SMT verification) against
jaxtyping + beartype (runtime, annotation-required shape checking) on 20
representative nn.Module benchmarks (10 buggy, 10 safe).

Key findings:
  - jaxtyping requires explicit shape annotations on every tensor parameter
    and return value; TensorGuard works on unannotated code.
  - jaxtyping is purely runtime: it needs concrete inputs and only catches
    bugs on executed code paths.  TensorGuard is static and catches bugs
    before any execution.
  - On standard shape bugs both tools may detect them, but jaxtyping needs
    annotated code + test inputs — which is what "runtime testing" already
    provides even without jaxtyping.
"""

import json
import os
import sys
import time
import traceback
from pathlib import Path

import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.model_checker import verify_model

OUTPUT_FILE = SCRIPT_DIR / "jaxtyping_baseline_results.json"

# ---------------------------------------------------------------------------
# Benchmark definitions — 20 models (10 buggy, 10 safe)
# ---------------------------------------------------------------------------
# Each entry: (name, source_code, input_shapes_dict, input_tensor_shape, is_buggy, category)

BENCHMARKS = [
    # -----------------------------------------------------------------------
    # 1. Linear dimension mismatch — BUG
    # -----------------------------------------------------------------------
    (
        "linear_dim_mismatch_bug",
        """\
import torch
import torch.nn as nn

class LinearDimBug(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(128, 10)  # expects 128 but fc1 outputs 256

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        return x
""",
        {"x": ("batch", 784)},
        (4, 784),
        True,
        "linear_dim_mismatch",
    ),
    # -----------------------------------------------------------------------
    # 2. Linear dimension — SAFE
    # -----------------------------------------------------------------------
    (
        "linear_dim_safe",
        """\
import torch
import torch.nn as nn

class LinearDimSafe(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        return x
""",
        {"x": ("batch", 784)},
        (4, 784),
        False,
        "linear_dim_mismatch",
    ),
    # -----------------------------------------------------------------------
    # 3. Conv2d channel mismatch — BUG
    # -----------------------------------------------------------------------
    (
        "conv_channel_mismatch_bug",
        """\
import torch
import torch.nn as nn

class ConvChannelBug(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 128, 3, padding=1)  # expects 32 but conv1 outputs 64

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x
""",
        {"x": ("batch", 3, 32, 32)},
        (4, 3, 32, 32),
        True,
        "conv_channel_mismatch",
    ),
    # -----------------------------------------------------------------------
    # 4. Conv2d channel — SAFE
    # -----------------------------------------------------------------------
    (
        "conv_channel_safe",
        """\
import torch
import torch.nn as nn

class ConvChannelSafe(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, 3, padding=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x
""",
        {"x": ("batch", 3, 32, 32)},
        (4, 3, 32, 32),
        False,
        "conv_channel_mismatch",
    ),
    # -----------------------------------------------------------------------
    # 5. Reshape / flatten bug — BUG
    # -----------------------------------------------------------------------
    (
        "reshape_flatten_bug",
        """\
import torch
import torch.nn as nn

class ReshapeBug(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 16, 3, padding=1)
        self.fc = nn.Linear(8192, 10)  # wrong: 16*32*32=16384, not 8192

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
""",
        {"x": ("batch", 3, 32, 32)},
        (4, 3, 32, 32),
        True,
        "reshape_flatten",
    ),
    # -----------------------------------------------------------------------
    # 6. Reshape / flatten — SAFE
    # -----------------------------------------------------------------------
    (
        "reshape_flatten_safe",
        """\
import torch
import torch.nn as nn

class ReshapeSafe(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 16, 3, padding=1)
        self.fc = nn.Linear(16384, 10)  # correct: 16*32*32=16384

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
""",
        {"x": ("batch", 3, 32, 32)},
        (4, 3, 32, 32),
        False,
        "reshape_flatten",
    ),
    # -----------------------------------------------------------------------
    # 7. Multi-head attention embed_dim % num_heads — BUG
    # -----------------------------------------------------------------------
    (
        "mha_divisibility_bug",
        """\
import torch
import torch.nn as nn

class MHABug(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=256, num_heads=7)  # 256 % 7 != 0

    def forward(self, x):
        x, _ = self.attn(x, x, x)
        return x
""",
        {"x": ("batch", 10, 256)},
        (10, 4, 256),
        True,
        "mha_divisibility",
    ),
    # -----------------------------------------------------------------------
    # 8. Multi-head attention — SAFE
    # -----------------------------------------------------------------------
    (
        "mha_divisibility_safe",
        """\
import torch
import torch.nn as nn

class MHASafe(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=256, num_heads=8)  # 256 % 8 == 0

    def forward(self, x):
        x, _ = self.attn(x, x, x)
        return x
""",
        {"x": ("batch", 10, 256)},
        (10, 4, 256),
        False,
        "mha_divisibility",
    ),
    # -----------------------------------------------------------------------
    # 9. Deep chain bug — BUG
    # -----------------------------------------------------------------------
    (
        "deep_chain_bug",
        """\
import torch
import torch.nn as nn

class DeepChainBug(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(512, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(32, 10)  # expects 32 but fc3 outputs 64

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)
        return x
""",
        {"x": ("batch", 512)},
        (4, 512),
        True,
        "deep_chain",
    ),
    # -----------------------------------------------------------------------
    # 10. Deep chain — SAFE
    # -----------------------------------------------------------------------
    (
        "deep_chain_safe",
        """\
import torch
import torch.nn as nn

class DeepChainSafe(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(512, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 10)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)
        return x
""",
        {"x": ("batch", 512)},
        (4, 512),
        False,
        "deep_chain",
    ),
    # -----------------------------------------------------------------------
    # 11. Conv → Linear flatten mismatch — BUG
    # -----------------------------------------------------------------------
    (
        "conv_linear_flatten_bug",
        """\
import torch
import torch.nn as nn

class ConvLinearBug(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3)  # 28->26
        self.conv2 = nn.Conv2d(32, 64, 3)  # 26->24
        self.fc = nn.Linear(64 * 12 * 12, 10)  # wrong: should be 64*24*24

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
""",
        {"x": ("batch", 1, 28, 28)},
        (4, 1, 28, 28),
        True,
        "conv_linear_flatten",
    ),
    # -----------------------------------------------------------------------
    # 12. Conv → Linear flatten — SAFE
    # -----------------------------------------------------------------------
    (
        "conv_linear_flatten_safe",
        """\
import torch
import torch.nn as nn

class ConvLinearSafe(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3)  # 28->26
        self.conv2 = nn.Conv2d(32, 64, 3)  # 26->24
        self.fc = nn.Linear(64 * 24 * 24, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
""",
        {"x": ("batch", 1, 28, 28)},
        (4, 1, 28, 28),
        False,
        "conv_linear_flatten",
    ),
    # -----------------------------------------------------------------------
    # 13. Transposed conv channel bug — BUG
    # -----------------------------------------------------------------------
    (
        "transposed_conv_bug",
        """\
import torch
import torch.nn as nn

class TransConvBug(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 64, 3, padding=1)
        self.deconv = nn.ConvTranspose2d(32, 3, 3, padding=1)  # expects 32 but gets 64

    def forward(self, x):
        x = self.conv(x)
        x = self.deconv(x)
        return x
""",
        {"x": ("batch", 3, 32, 32)},
        (4, 3, 32, 32),
        True,
        "transposed_conv",
    ),
    # -----------------------------------------------------------------------
    # 14. Transposed conv channel — SAFE
    # -----------------------------------------------------------------------
    (
        "transposed_conv_safe",
        """\
import torch
import torch.nn as nn

class TransConvSafe(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 64, 3, padding=1)
        self.deconv = nn.ConvTranspose2d(64, 3, 3, padding=1)

    def forward(self, x):
        x = self.conv(x)
        x = self.deconv(x)
        return x
""",
        {"x": ("batch", 3, 32, 32)},
        (4, 3, 32, 32),
        False,
        "transposed_conv",
    ),
    # -----------------------------------------------------------------------
    # 15. Linear after pooling bug — BUG
    # -----------------------------------------------------------------------
    (
        "pool_linear_bug",
        """\
import torch
import torch.nn as nn

class PoolLinearBug(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 32, 3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d((4, 4))
        self.fc = nn.Linear(32 * 2 * 2, 10)  # wrong: pool outputs 4x4 not 2x2

    def forward(self, x):
        x = self.conv(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
""",
        {"x": ("batch", 3, 32, 32)},
        (4, 3, 32, 32),
        True,
        "pool_linear",
    ),
    # -----------------------------------------------------------------------
    # 16. Linear after pooling — SAFE
    # -----------------------------------------------------------------------
    (
        "pool_linear_safe",
        """\
import torch
import torch.nn as nn

class PoolLinearSafe(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 32, 3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d((4, 4))
        self.fc = nn.Linear(32 * 4 * 4, 10)

    def forward(self, x):
        x = self.conv(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
""",
        {"x": ("batch", 3, 32, 32)},
        (4, 3, 32, 32),
        False,
        "pool_linear",
    ),
    # -----------------------------------------------------------------------
    # 17. Residual add shape mismatch — BUG
    # -----------------------------------------------------------------------
    (
        "residual_shape_bug",
        """\
import torch
import torch.nn as nn

class ResidualBug(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(256, 128)  # changes dim from 256 to 128
        self.fc2 = nn.Linear(128, 128)

    def forward(self, x):
        residual = x  # shape (batch, 256)
        x = self.fc1(x)  # shape (batch, 128)
        x = self.fc2(x)  # shape (batch, 128)
        x = x + residual  # BUG: (batch,128) + (batch,256)
        return x
""",
        {"x": ("batch", 256)},
        (4, 256),
        True,
        "residual_shape",
    ),
    # -----------------------------------------------------------------------
    # 18. Residual add shape — SAFE
    # -----------------------------------------------------------------------
    (
        "residual_shape_safe",
        """\
import torch
import torch.nn as nn

class ResidualSafe(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(256, 256)
        self.fc2 = nn.Linear(256, 256)

    def forward(self, x):
        residual = x
        x = self.fc1(x)
        x = self.fc2(x)
        x = x + residual
        return x
""",
        {"x": ("batch", 256)},
        (4, 256),
        False,
        "residual_shape",
    ),
    # -----------------------------------------------------------------------
    # 19. BatchNorm channel mismatch — BUG
    # -----------------------------------------------------------------------
    (
        "batchnorm_channel_bug",
        """\
import torch
import torch.nn as nn

class BNBug(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 64, 3, padding=1)
        self.bn = nn.BatchNorm2d(32)  # expects 32 channels but conv outputs 64

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x
""",
        {"x": ("batch", 3, 32, 32)},
        (4, 3, 32, 32),
        True,
        "batchnorm_channel",
    ),
    # -----------------------------------------------------------------------
    # 20. BatchNorm channel — SAFE
    # -----------------------------------------------------------------------
    (
        "batchnorm_channel_safe",
        """\
import torch
import torch.nn as nn

class BNSafe(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 64, 3, padding=1)
        self.bn = nn.BatchNorm2d(64)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x
""",
        {"x": ("batch", 3, 32, 32)},
        (4, 3, 32, 32),
        False,
        "batchnorm_channel",
    ),
]


# ---------------------------------------------------------------------------
# Runners
# ---------------------------------------------------------------------------

def run_tensorguard(source: str, input_shapes: dict) -> dict:
    """Run TensorGuard static verification."""
    t0 = time.monotonic()
    try:
        result = verify_model(source, input_shapes)
        elapsed = (time.monotonic() - t0) * 1000
        detected = not result.safe
        errors = list(result.errors) if result.errors else []
        if result.counterexample:
            errors.append(str(result.counterexample))
        return {
            "detected_bug": detected,
            "time_ms": round(elapsed, 2),
            "errors": errors,
        }
    except Exception as e:
        elapsed = (time.monotonic() - t0) * 1000
        return {
            "detected_bug": False,
            "time_ms": round(elapsed, 2),
            "errors": [f"TensorGuard exception: {e}"],
        }


def run_jaxtyping_runtime(source: str, input_shape: tuple) -> dict:
    """Run the model with a concrete tensor to detect runtime shape errors.

    This is what jaxtyping + beartype enables: runtime shape checking when
    annotated code is executed with concrete inputs.  Without jaxtyping
    annotations, PyTorch itself raises RuntimeError on shape mismatches,
    so this baseline captures what runtime testing achieves.
    """
    t0 = time.monotonic()
    try:
        # Execute the source to get the model class
        namespace = {}
        exec(source, namespace)

        # Find the nn.Module subclass defined in the source
        model_cls = None
        for obj in namespace.values():
            if isinstance(obj, type) and issubclass(obj, nn.Module) and obj is not nn.Module:
                model_cls = obj
                break

        if model_cls is None:
            elapsed = (time.monotonic() - t0) * 1000
            return {
                "detected_bug": False,
                "time_ms": round(elapsed, 2),
                "errors": ["No nn.Module subclass found"],
            }

        # Instantiate and run forward with a random tensor
        model = model_cls()
        model.eval()
        x = torch.randn(*input_shape)
        with torch.no_grad():
            model.forward(x)

        elapsed = (time.monotonic() - t0) * 1000
        return {
            "detected_bug": False,
            "time_ms": round(elapsed, 2),
            "errors": [],
        }
    except RuntimeError as e:
        elapsed = (time.monotonic() - t0) * 1000
        return {
            "detected_bug": True,
            "time_ms": round(elapsed, 2),
            "errors": [f"RuntimeError: {e}"],
        }
    except Exception as e:
        elapsed = (time.monotonic() - t0) * 1000
        return {
            "detected_bug": True,
            "time_ms": round(elapsed, 2),
            "errors": [f"{type(e).__name__}: {e}"],
        }


def check_jaxtyping_static(source: str) -> dict:
    """Check whether jaxtyping could detect the bug statically (it cannot).

    jaxtyping is purely a runtime tool — it decorates functions with
    @jaxtyped and uses beartype to check shapes at call time.  There is
    no static analysis component.
    """
    return {
        "detected_bug": False,
        "note": "jaxtyping has no static analysis; requires runtime execution with annotations",
    }


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(results: list, key: str) -> dict:
    tp = fp = tn = fn = 0
    for r in results:
        is_buggy = r["is_buggy"]
        detected = r[key]["detected_bug"]
        if is_buggy and detected:
            tp += 1
        elif is_buggy and not detected:
            fn += 1
        elif not is_buggy and detected:
            fp += 1
        else:
            tn += 1

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 70)
    print("jaxtyping + beartype baseline comparison")
    print("=" * 70)
    print(f"Benchmarks: {len(BENCHMARKS)} ({sum(1 for b in BENCHMARKS if b[4])} buggy, "
          f"{sum(1 for b in BENCHMARKS if not b[4])} safe)\n")

    results = []

    for name, source, input_shapes, tensor_shape, is_buggy, category in BENCHMARKS:
        print(f"  [{category}] {name} ({'BUG' if is_buggy else 'SAFE'}) ... ", end="", flush=True)

        tg = run_tensorguard(source, input_shapes)
        jrt = run_jaxtyping_runtime(source, tensor_shape)
        jst = check_jaxtyping_static(source)

        # Verdict summary
        tg_v = "BUG" if tg["detected_bug"] else "safe"
        jrt_v = "BUG" if jrt["detected_bug"] else "safe"
        print(f"TG={tg_v}  jaxtyping-runtime={jrt_v}")

        results.append({
            "name": name,
            "category": category,
            "is_buggy": is_buggy,
            "tensorguard": tg,
            "jaxtyping_runtime": jrt,
            "jaxtyping_static": jst,
        })

    # Compute aggregate metrics
    tg_metrics = compute_metrics(results, "tensorguard")
    jrt_metrics = compute_metrics(results, "jaxtyping_runtime")

    analysis = (
        "TensorGuard is a static verifier that works on unannotated nn.Module code, "
        "catching shape bugs before any execution via Z3 SMT solving. "
        "jaxtyping + beartype requires (1) explicit shape annotations on every "
        "tensor parameter and return value, and (2) concrete test inputs that "
        "exercise the buggy code path. Without annotations, jaxtyping provides "
        "no benefit over plain runtime testing — PyTorch itself raises RuntimeError "
        "on shape mismatches. The 'jaxtyping_runtime' column here measures runtime "
        "testing (which is what annotated jaxtyping enables), showing that while "
        "runtime testing can catch bugs on executed paths, it fundamentally cannot "
        "provide the ahead-of-time guarantees that static verification offers. "
        "TensorGuard requires zero developer annotation effort and catches bugs "
        "before deployment."
    )

    output = {
        "benchmarks": results,
        "tensorguard": tg_metrics,
        "jaxtyping_runtime": jrt_metrics,
        "analysis": analysis,
    }

    with open(OUTPUT_FILE, "w") as f:
        json.dump(output, f, indent=2, default=str)

    # Print summary
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)

    print(f"\n{'Metric':<12} {'TensorGuard':>14} {'jaxtyping-runtime':>18}")
    print("-" * 46)
    for m in ["tp", "fp", "tn", "fn", "precision", "recall", "f1"]:
        print(f"{m:<12} {tg_metrics[m]:>14} {jrt_metrics[m]:>18}")

    print(f"\nResults saved to {OUTPUT_FILE}")

    print("\n" + "=" * 70)
    print("KEY INSIGHT")
    print("=" * 70)
    print("• TensorGuard: static, annotation-free, catches bugs BEFORE runtime")
    print("• jaxtyping:   runtime-only, requires annotations + test inputs")
    print("• jaxtyping without annotations ≡ plain runtime testing (PyTorch")
    print("  already raises RuntimeError on shape mismatches)")
    print("• jaxtyping cannot catch bugs on untested code paths")
    print("=" * 70)


if __name__ == "__main__":
    main()
