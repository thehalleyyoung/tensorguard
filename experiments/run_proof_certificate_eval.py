#!/usr/bin/env python3
"""
Proof Certificate Evaluation for TensorGuard.

Runs proof extraction on benchmark models and reports:
  - Extraction success rate
  - Average proof steps
  - Average extraction time
  - Local verification rate

Saves results to .benchmarks/proof_certificate_results.json.
"""
from __future__ import annotations

import json
import os
import sys
import time
import traceback
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

try:
    import z3
    HAS_Z3 = True
except ImportError:
    HAS_Z3 = False

from src.model_checker import verify_model, VerificationResult

EXPERIMENTS_DIR = Path(__file__).resolve().parent
BENCHMARKS_DIR = EXPERIMENTS_DIR / ".benchmarks"
BENCHMARKS_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_FILE = BENCHMARKS_DIR / "proof_certificate_results.json"

# ═══════════════════════════════════════════════════════════════════════════════
# Benchmark model source code
# ═══════════════════════════════════════════════════════════════════════════════

BENCHMARK_MODELS: List[Dict[str, str]] = [
    {
        "name": "SimpleMLP",
        "code": """
import torch
import torch.nn as nn

class SimpleMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        return x
""",
    },
    {
        "name": "TwoLayerMLP",
        "code": """
import torch
import torch.nn as nn

class TwoLayerMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x
""",
    },
    {
        "name": "SingleLinear",
        "code": """
import torch.nn as nn

class SingleLinear(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 5)

    def forward(self, x):
        return self.fc(x)
""",
    },
    {
        "name": "DropoutMLP",
        "code": """
import torch
import torch.nn as nn

class DropoutMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(100, 50)
        self.drop = nn.Dropout(0.5)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.drop(x)
        x = self.fc2(x)
        return x
""",
    },
    {
        "name": "BatchNormMLP",
        "code": """
import torch
import torch.nn as nn

class BatchNormMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(64, 32)
        self.bn = nn.BatchNorm1d(32)
        self.fc2 = nn.Linear(32, 10)

    def forward(self, x):
        x = self.fc1(x)
        x = self.bn(x)
        x = torch.relu(x)
        x = self.fc2(x)
        return x
""",
    },
    {
        "name": "WideMLP",
        "code": """
import torch
import torch.nn as nn

class WideMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(512, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 10)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x
""",
    },
    {
        "name": "DeepMLP",
        "code": """
import torch
import torch.nn as nn

class DeepMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 10)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)
        return x
""",
    },
    {
        "name": "Autoencoder",
        "code": """
import torch
import torch.nn as nn

class Autoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Linear(784, 64)
        self.decoder = nn.Linear(64, 784)

    def forward(self, x):
        z = torch.relu(self.encoder(x))
        x_hat = torch.sigmoid(self.decoder(z))
        return x_hat
""",
    },
    {
        "name": "SimpleCNN",
        "code": """
import torch
import torch.nn as nn

class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 3)
        self.fc1 = nn.Linear(16, 10)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = self.fc1(x)
        return x
""",
    },
    {
        "name": "ResidualBlock",
        "code": """
import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(64, 64)
        self.fc2 = nn.Linear(64, 64)

    def forward(self, x):
        residual = x
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        x = x + residual
        return torch.relu(x)
""",
    },
    {
        "name": "SmallClassifier",
        "code": """
import torch
import torch.nn as nn

class SmallClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(20, 3)

    def forward(self, x):
        return torch.softmax(self.fc(x), dim=-1)
""",
    },
    {
        "name": "DoubleLinear",
        "code": """
import torch
import torch.nn as nn

class DoubleLinear(nn.Module):
    def __init__(self):
        super().__init__()
        self.a = nn.Linear(32, 16)
        self.b = nn.Linear(16, 8)

    def forward(self, x):
        return self.b(torch.relu(self.a(x)))
""",
    },
    {
        "name": "SigmoidNet",
        "code": """
import torch
import torch.nn as nn

class SigmoidNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(50, 25)
        self.fc2 = nn.Linear(25, 1)

    def forward(self, x):
        x = torch.sigmoid(self.fc1(x))
        x = self.fc2(x)
        return x
""",
    },
    {
        "name": "TanhMLP",
        "code": """
import torch
import torch.nn as nn

class TanhMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(40, 20)
        self.fc2 = nn.Linear(20, 5)

    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        x = self.fc2(x)
        return x
""",
    },
    {
        "name": "IdentityPassthrough",
        "code": """
import torch.nn as nn

class IdentityPassthrough(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 10)

    def forward(self, x):
        return self.fc(x)
""",
    },
]


@dataclass
class BenchmarkResult:
    name: str
    safe: bool = False
    has_proof_certificate: bool = False
    proof_steps: int = 0
    theory_lemma_count: int = 0
    max_depth: int = 0
    extraction_time_ms: float = 0.0
    locally_verified: bool = False
    theories_used: List[str] = None  # type: ignore[assignment]
    error: Optional[str] = None

    def __post_init__(self):
        if self.theories_used is None:
            self.theories_used = []


def run_benchmark(model_info: Dict[str, str]) -> BenchmarkResult:
    name = model_info["name"]
    code = model_info["code"]
    result = BenchmarkResult(name=name)

    try:
        vr: VerificationResult = verify_model(code)
        result.safe = vr.safe

        pc = getattr(vr, "proof_certificate", None)
        if pc is None and vr.certificate is not None:
            pc = getattr(vr.certificate, "proof_certificate", None)

        if pc is not None:
            result.has_proof_certificate = True
            stats = pc.summary_stats()
            result.proof_steps = stats["step_count"]
            result.theory_lemma_count = stats["theory_lemma_count"]
            result.max_depth = stats["max_depth"]
            result.extraction_time_ms = pc.extraction_time_ms
            result.locally_verified = pc.verify_locally()
            result.theories_used = pc.theories_used
    except Exception as e:
        result.error = f"{type(e).__name__}: {e}"

    return result


def main():
    print("=" * 70)
    print("TensorGuard Proof Certificate Evaluation")
    print("=" * 70)

    if not HAS_Z3:
        print("ERROR: z3 not installed.  pip install z3-solver")
        sys.exit(1)

    results: List[Dict[str, Any]] = []
    total = len(BENCHMARK_MODELS)
    extracted = 0
    verified = 0
    total_steps = 0
    total_time = 0.0

    for i, model_info in enumerate(BENCHMARK_MODELS, 1):
        name = model_info["name"]
        print(f"\n[{i}/{total}] {name} ... ", end="", flush=True)
        br = run_benchmark(model_info)
        results.append(asdict(br))

        if br.error:
            print(f"ERROR: {br.error}")
        elif br.has_proof_certificate:
            extracted += 1
            total_steps += br.proof_steps
            total_time += br.extraction_time_ms
            if br.locally_verified:
                verified += 1
            print(
                f"✓ proof: {br.proof_steps} steps, "
                f"depth={br.max_depth}, "
                f"{br.extraction_time_ms:.1f}ms, "
                f"verified={br.locally_verified}"
            )
        else:
            print(f"safe={br.safe}, no proof certificate (expected for SAT-based verification)")

    # Summary
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    print(f"  Total benchmarks:       {total}")
    print(f"  Extraction success:     {extracted}/{total} ({100*extracted/total:.0f}%)")
    avg_steps = total_steps / extracted if extracted else 0
    avg_time = total_time / extracted if extracted else 0
    print(f"  Avg proof steps:        {avg_steps:.1f}")
    print(f"  Avg extraction time:    {avg_time:.1f}ms")
    print(f"  Local verification:     {verified}/{extracted} ({100*verified/extracted:.0f}% of extracted)" if extracted else "  Local verification:     N/A")

    summary = {
        "total_benchmarks": total,
        "extraction_success": extracted,
        "extraction_rate": extracted / total if total else 0,
        "avg_proof_steps": avg_steps,
        "avg_extraction_time_ms": avg_time,
        "verified_count": verified,
        "verification_rate": verified / extracted if extracted else 0,
        "results": results,
    }

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nResults saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
