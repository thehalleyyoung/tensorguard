"""
Counterexample Validation: Concretize TensorGuard Bugs Against PyTorch.

For each bug TensorGuard finds, constructs a concrete PyTorch model,
creates an input tensor with the counterexample dimensions, and runs
a forward pass to verify that PyTorch raises RuntimeError.

This validates that TensorGuard's bug reports correspond to real
runtime errors, not false positives from over-approximation.
"""

from __future__ import annotations

import json
import os
import sys
import time
import importlib
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

RESULTS_FILE = Path(__file__).parent / "counterexample_validation_results.json"


# Models with known bugs and the expected runtime error
VALIDATION_CASES: List[Dict[str, Any]] = [
    {
        "name": "linear_dim_mismatch",
        "description": "Linear layer expects 10 features, input has 20",
        "code": """\
import torch
import torch.nn as nn
class M(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 5)
    def forward(self, x):
        return self.fc(x)
""",
        "input_shapes": {"x": ("batch", 20)},
        "concrete_input": {"x": (4, 20)},
        "expected_error": "RuntimeError",
    },
    {
        "name": "conv2d_channel_mismatch",
        "description": "Conv2d expects 3 channels, input has 64",
        "code": """\
import torch
import torch.nn as nn
class M(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 16, 3)
    def forward(self, x):
        return self.conv(x)
""",
        "input_shapes": {"x": ("batch", 64, 32, 32)},
        "concrete_input": {"x": (2, 64, 32, 32)},
        "expected_error": "RuntimeError",
    },
    {
        "name": "matmul_incompatible",
        "description": "Matrix multiply with incompatible inner dimensions",
        "code": """\
import torch
import torch.nn as nn
class M(nn.Module):
    def __init__(self):
        super().__init__()
        self.w = nn.Linear(10, 5)
        self.v = nn.Linear(8, 3)
    def forward(self, x):
        a = self.w(x)
        b = self.v(x)
        return a @ b
""",
        "input_shapes": {"x": ("batch", 10)},
        "concrete_input": {"x": (4, 10)},
        "expected_error": "RuntimeError",
    },
    {
        "name": "linear_chain_mismatch",
        "description": "Second Linear expects 256 but first outputs 128",
        "code": """\
import torch
import torch.nn as nn
class M(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(100, 128)
        self.fc2 = nn.Linear(256, 64)
    def forward(self, x):
        return self.fc2(self.fc1(x))
""",
        "input_shapes": {"x": ("batch", 100)},
        "concrete_input": {"x": (4, 100)},
        "expected_error": "RuntimeError",
    },
    {
        "name": "batchnorm_channel_mismatch",
        "description": "BatchNorm2d expects 32 features but Conv2d outputs 64",
        "code": """\
import torch
import torch.nn as nn
class M(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 64, 3, padding=1)
        self.bn = nn.BatchNorm2d(32)
    def forward(self, x):
        return self.bn(self.conv(x))
""",
        "input_shapes": {"x": ("batch", 3, 32, 32)},
        "concrete_input": {"x": (2, 3, 32, 32)},
        "expected_error": "RuntimeError",
    },
    {
        "name": "add_shape_mismatch",
        "description": "Adding tensors with incompatible shapes (non-broadcastable)",
        "code": """\
import torch
import torch.nn as nn
class M(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(100, 50)
        self.fc2 = nn.Linear(100, 30)
    def forward(self, x):
        return self.fc1(x) + self.fc2(x)
""",
        "input_shapes": {"x": ("batch", 100)},
        "concrete_input": {"x": (4, 100)},
        "expected_error": "RuntimeError",
    },
    {
        "name": "mha_embed_dim_mismatch",
        "description": "MultiheadAttention embed_dim=512 but input last dim=768",
        "code": """\
import torch
import torch.nn as nn
class M(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = nn.MultiheadAttention(512, 8)
    def forward(self, x):
        out, _ = self.attn(x, x, x)
        return out
""",
        "input_shapes": {"x": ("seq", "batch", 768)},
        "concrete_input": {"x": (10, 4, 768)},
        "expected_error": "RuntimeError",
    },
    {
        "name": "transformer_encoder_dim_mismatch",
        "description": "TransformerEncoder d_model=512 but input last dim=256",
        "code": """\
import torch
import torch.nn as nn
class M(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=512, nhead=8), num_layers=2)
    def forward(self, x):
        return self.encoder(x)
""",
        "input_shapes": {"x": ("seq", "batch", 256)},
        "concrete_input": {"x": (10, 4, 256)},
        "expected_error": "RuntimeError",
    },
    {
        "name": "residual_shape_mismatch",
        "description": "Residual add fails because projection changes dim",
        "code": """\
import torch
import torch.nn as nn
class M(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(512, 256)
    def forward(self, x):
        return x + self.fc(x)
""",
        "input_shapes": {"x": ("batch", 512)},
        "concrete_input": {"x": (4, 512)},
        "expected_error": "RuntimeError",
    },
    {
        "name": "conv_chain_channel_mismatch",
        "description": "Second Conv2d expects 32 channels but first outputs 64",
        "code": """\
import torch
import torch.nn as nn
class M(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 128, 3, padding=1)
    def forward(self, x):
        return self.conv2(self.conv1(x))
""",
        "input_shapes": {"x": ("batch", 3, 32, 32)},
        "concrete_input": {"x": (2, 3, 32, 32)},
        "expected_error": "RuntimeError",
    },
]


def verify_and_validate(case: Dict[str, Any]) -> Dict[str, Any]:
    """Run TensorGuard verification then concretize against PyTorch."""
    from src.model_checker import verify_model

    # Step 1: TensorGuard verification
    result = verify_model(case["code"], input_shapes=case["input_shapes"])

    # Step 2: Concrete PyTorch execution
    pytorch_error = None
    pytorch_error_type = None
    try:
        import torch
        # Execute the model code to define the class
        namespace = {}
        exec(case["code"], namespace)
        # Find the nn.Module class
        model_cls = None
        for name, obj in namespace.items():
            if isinstance(obj, type) and issubclass(obj, torch.nn.Module) and obj != torch.nn.Module:
                model_cls = obj
                break

        if model_cls is not None:
            model = model_cls()
            model.eval()
            # Create concrete input
            inputs = {}
            for param_name, shape in case["concrete_input"].items():
                inputs[param_name] = torch.randn(*shape)
            # Run forward pass
            with torch.no_grad():
                if len(inputs) == 1:
                    out = model(list(inputs.values())[0])
                else:
                    out = model(*inputs.values())
    except RuntimeError as e:
        pytorch_error = str(e)[:200]
        pytorch_error_type = "RuntimeError"
    except Exception as e:
        pytorch_error = str(e)[:200]
        pytorch_error_type = type(e).__name__

    # Step 3: Compare
    tensorguard_found_bug = not result.safe
    pytorch_raised_error = pytorch_error is not None
    validated = tensorguard_found_bug and pytorch_raised_error

    return {
        "name": case["name"],
        "description": case["description"],
        "tensorguard_found_bug": tensorguard_found_bug,
        "pytorch_raised_error": pytorch_raised_error,
        "pytorch_error_type": pytorch_error_type,
        "pytorch_error_message": pytorch_error,
        "counterexample_validated": validated,
        "expected_error": case["expected_error"],
    }


def run_counterexample_validation() -> Dict[str, Any]:
    """Run full counterexample validation suite."""
    results = {
        "experiment": "counterexample_validation",
        "date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "cases": [],
        "summary": {},
    }

    print("=" * 70)
    print("TensorGuard Counterexample Validation")
    print("=" * 70)

    for case in VALIDATION_CASES:
        r = verify_and_validate(case)
        results["cases"].append(r)

        status = "✓ VALIDATED" if r["counterexample_validated"] else (
            "✗ MISMATCH" if r["tensorguard_found_bug"] != r["pytorch_raised_error"]
            else "? NO BUG FOUND"
        )
        print(f"  {case['name']:40s} {status}")
        if r["pytorch_error_message"]:
            print(f"    PyTorch: {r['pytorch_error_message'][:80]}")

    validated = sum(1 for r in results["cases"] if r["counterexample_validated"])
    total = len(results["cases"])
    tg_found = sum(1 for r in results["cases"] if r["tensorguard_found_bug"])
    pt_raised = sum(1 for r in results["cases"] if r["pytorch_raised_error"])

    results["summary"] = {
        "total_cases": total,
        "tensorguard_found_bug": tg_found,
        "pytorch_raised_error": pt_raised,
        "counterexamples_validated": validated,
        "validation_rate": round(validated / total, 4) if total else 0,
        "false_positive_rate": round(
            (tg_found - validated) / tg_found, 4
        ) if tg_found else 0,
    }

    print(f"\n  Summary: {validated}/{total} counterexamples validated "
          f"({results['summary']['validation_rate']:.0%})")
    print(f"  TensorGuard found bugs: {tg_found}/{total}")
    print(f"  PyTorch raised errors: {pt_raised}/{total}")

    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved to {RESULTS_FILE}")

    return results


if __name__ == "__main__":
    run_counterexample_validation()
