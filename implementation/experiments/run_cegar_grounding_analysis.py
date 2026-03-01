"""
CEGAR Convergence Grounding Analysis.

Addresses MAJOR reviewer critique #5: the CEGAR convergence bound (N
iterations) is abstract. Reviewers want to know the relationship between
N and the actual predicate template.

This script:
  1. Enumerates the predicate template kinds used by the CEGAR loop.
  2. For each benchmark, tracks: CEGAR iterations, predicates discovered
     per iteration, total universe size.
  3. Computes the empirical universe size and relates it to the
     theoretical N bound (L × D × K).
  4. Partitions benchmarks into decidable vs undecidable fragments.

Outputs:
  - experiments/results/cegar_grounding_results.json
"""

from __future__ import annotations

import json
import math
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.shape_cegar import (
    run_shape_cegar,
    ShapeCEGARResult,
    CEGARStatus,
    CEGARVerdict,
    PredicateKind,
    ShapePredicate,
    compute_predicate_universe_bound,
)
from src.cegar_cpa import (
    PredicateLattice,
    NUM_PREDICATE_KINDS,
)
from src.model_checker import Device, Phase

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_FILE = RESULTS_DIR / "cegar_grounding_results.json"

# ---------------------------------------------------------------------------
# Benchmark suite for CEGAR grounding analysis
# ---------------------------------------------------------------------------

CEGAR_BENCHMARKS = [
    {
        "name": "single_linear",
        "description": "Single linear layer — minimal CEGAR case",
        "num_layers": 1,
        "max_dims": 2,
        "code": '''
import torch.nn as nn
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(768, 10)
    def forward(self, x):
        return self.fc(x)
''',
        "input_shapes": {"x": ("batch", "features")},
        "has_bug": False,
    },
    {
        "name": "two_layer_mlp",
        "description": "Two-layer MLP",
        "num_layers": 2,
        "max_dims": 2,
        "code": '''
import torch.nn as nn
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 10)
    def forward(self, x):
        return self.fc2(self.fc1(x))
''',
        "input_shapes": {"x": ("batch", 784)},
        "has_bug": False,
    },
    {
        "name": "three_layer_mlp",
        "description": "Three-layer MLP with ReLU",
        "num_layers": 3,
        "max_dims": 2,
        "code": '''
import torch.nn as nn
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 10)
        self.relu = nn.ReLU()
    def forward(self, x):
        h = self.relu(self.fc1(x))
        h = self.relu(self.fc2(h))
        return self.fc3(h)
''',
        "input_shapes": {"x": ("batch", 128)},
        "has_bug": False,
    },
    {
        "name": "deep_mlp_5layer",
        "description": "Five-layer deep MLP",
        "num_layers": 5,
        "max_dims": 2,
        "code": '''
import torch.nn as nn
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(512, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 32)
        self.fc5 = nn.Linear(32, 10)
    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)
        return self.fc5(x)
''',
        "input_shapes": {"x": ("batch", 512)},
        "has_bug": False,
    },
    {
        "name": "conv_net",
        "description": "Simple CNN with Conv2d + Linear",
        "num_layers": 3,
        "max_dims": 4,
        "code": '''
import torch.nn as nn
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3)
        self.conv2 = nn.Conv2d(16, 32, 3)
        self.fc = nn.Linear(32, 10)
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return self.fc(x.view(x.size(0), -1))
''',
        "input_shapes": {"x": ("batch", 3, 32, 32)},
        "has_bug": False,
    },
    {
        "name": "linear_dim_mismatch_bug",
        "description": "Buggy: Linear expects 768 but gets 512",
        "num_layers": 1,
        "max_dims": 2,
        "code": '''
import torch.nn as nn
class BuggyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(768, 10)
    def forward(self, x):
        return self.fc(x)
''',
        "input_shapes": {"x": ("batch", 512)},
        "has_bug": True,
    },
    {
        "name": "chain_mismatch_bug",
        "description": "Buggy: fc2 expects 128 but fc1 outputs 256",
        "num_layers": 2,
        "max_dims": 2,
        "code": '''
import torch.nn as nn
class BuggyChain(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(100, 256)
        self.fc2 = nn.Linear(128, 10)
    def forward(self, x):
        return self.fc2(self.fc1(x))
''',
        "input_shapes": {"x": ("batch", 100)},
        "has_bug": True,
    },
    {
        "name": "embedding_model",
        "description": "Embedding + Linear",
        "num_layers": 2,
        "max_dims": 3,
        "code": '''
import torch.nn as nn
class EmbModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(10000, 128)
        self.fc = nn.Linear(128, 10)
    def forward(self, x):
        e = self.embed(x)
        return self.fc(e)
''',
        "input_shapes": {"x": ("batch", "seq_len")},
        "has_bug": False,
    },
    {
        "name": "residual_block",
        "description": "Residual block with skip connection",
        "num_layers": 2,
        "max_dims": 2,
        "code": '''
import torch
import torch.nn as nn
class ResBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(256, 256)
        self.fc2 = nn.Linear(256, 256)
    def forward(self, x):
        h = self.fc2(self.fc1(x))
        return h + x
''',
        "input_shapes": {"x": ("batch", 256)},
        "has_bug": False,
    },
    {
        "name": "batchnorm_mlp",
        "description": "MLP with BatchNorm layers",
        "num_layers": 4,
        "max_dims": 2,
        "code": '''
import torch.nn as nn
class BNNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(100, 64)
        self.bn1 = nn.BatchNorm1d(64)
        self.fc2 = nn.Linear(64, 32)
        self.bn2 = nn.BatchNorm1d(32)
        self.fc3 = nn.Linear(32, 10)
    def forward(self, x):
        x = self.bn1(self.fc1(x))
        x = self.bn2(self.fc2(x))
        return self.fc3(x)
''',
        "input_shapes": {"x": ("batch", 100)},
        "has_bug": False,
    },
    {
        "name": "symbolic_input",
        "description": "Fully symbolic input dimensions",
        "num_layers": 2,
        "max_dims": 2,
        "code": '''
import torch.nn as nn
class SymNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(64, 32)
        self.fc2 = nn.Linear(32, 10)
    def forward(self, x):
        return self.fc2(self.fc1(x))
''',
        "input_shapes": {"x": ("batch", "d_in")},
        "has_bug": False,
    },
    {
        "name": "wide_network",
        "description": "Wide network with many parameters per layer",
        "num_layers": 2,
        "max_dims": 2,
        "code": '''
import torch.nn as nn
class WideNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(2048, 4096)
        self.fc2 = nn.Linear(4096, 1000)
    def forward(self, x):
        return self.fc2(self.fc1(x))
''',
        "input_shapes": {"x": ("batch", 2048)},
        "has_bug": False,
    },
    {
        "name": "autoencoder",
        "description": "Autoencoder with encoder + decoder",
        "num_layers": 4,
        "max_dims": 2,
        "code": '''
import torch.nn as nn
class AE(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc1 = nn.Linear(784, 256)
        self.enc2 = nn.Linear(256, 64)
        self.dec1 = nn.Linear(64, 256)
        self.dec2 = nn.Linear(256, 784)
    def forward(self, x):
        z = self.enc2(self.enc1(x))
        return self.dec2(self.dec1(z))
''',
        "input_shapes": {"x": ("batch", 784)},
        "has_bug": False,
    },
    {
        "name": "multi_head_simple",
        "description": "Simple multi-head architecture",
        "num_layers": 3,
        "max_dims": 2,
        "code": '''
import torch
import torch.nn as nn
class MultiHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.shared = nn.Linear(100, 64)
        self.head1 = nn.Linear(64, 10)
        self.head2 = nn.Linear(64, 5)
    def forward(self, x):
        h = self.shared(x)
        return self.head1(h), self.head2(h)
''',
        "input_shapes": {"x": ("batch", 100)},
        "has_bug": False,
    },
    {
        "name": "deep_chain_7layer",
        "description": "Seven-layer deep chain — tests convergence bound",
        "num_layers": 7,
        "max_dims": 2,
        "code": '''
import torch.nn as nn
class DeepChain(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 64)
        self.fc5 = nn.Linear(64, 32)
        self.fc6 = nn.Linear(32, 32)
        self.fc7 = nn.Linear(32, 10)
    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)
        x = self.fc5(x)
        x = self.fc6(x)
        return self.fc7(x)
''',
        "input_shapes": {"x": ("batch", 256)},
        "has_bug": False,
    },
]


# ---------------------------------------------------------------------------
# Analysis functions
# ---------------------------------------------------------------------------

def analyze_benchmark(bench: Dict[str, Any]) -> Dict[str, Any]:
    """Run CEGAR on a benchmark and collect convergence grounding data."""
    name = bench["name"]
    source = bench["code"]
    input_shapes = bench["input_shapes"]

    entry: Dict[str, Any] = {
        "benchmark": name,
        "description": bench.get("description", ""),
        "has_bug": bench.get("has_bug", False),
        "num_layers_annotated": bench.get("num_layers", 0),
        "max_dims_annotated": bench.get("max_dims", 0),
    }

    # Compute theoretical bound
    L = bench.get("num_layers", 1)
    D = bench.get("max_dims", 2)
    K = NUM_PREDICATE_KINDS  # 7
    theoretical_bound = L * D * K
    entry["theoretical_bound"] = {
        "N": theoretical_bound,
        "L_layers": L,
        "D_max_dims": D,
        "K_predicate_kinds": K,
        "formula": f"|P_prog| = {L} × {D} × {K} = {theoretical_bound}",
    }

    # Also use the compute_predicate_universe_bound utility
    bound_info = compute_predicate_universe_bound({
        "num_layers": L,
        "max_dims_per_layer": D,
    })
    entry["bound_from_utility"] = bound_info

    # Run CEGAR
    t0 = time.monotonic()
    try:
        result = run_shape_cegar(
            source,
            input_shapes=input_shapes,
            max_iterations=20,  # generous budget to see if bound is tight
        )
    except Exception as e:
        entry["error"] = str(e)[:200]
        entry["empirical"] = None
        return entry
    elapsed_ms = (time.monotonic() - t0) * 1000

    # Collect empirical data
    empirical: Dict[str, Any] = {
        "iterations": result.iterations,
        "final_status": result.final_status.name,
        "verdict": result.verdict.name,
        "total_predicates_discovered": len(result.discovered_predicates),
        "total_time_ms": round(elapsed_ms, 2),
    }

    # Per-iteration breakdown
    per_iteration: List[Dict[str, Any]] = []
    for rec in result.iteration_log:
        iter_entry = {
            "iteration": rec.iteration,
            "num_violations": rec.num_violations,
            "num_spurious": rec.num_spurious,
            "num_real": rec.num_real,
            "predicates_added": len(rec.predicates_added),
            "predicate_kinds": [p.kind.name for p in rec.predicates_added],
            "time_ms": round(rec.time_ms, 2),
        }
        per_iteration.append(iter_entry)
    empirical["per_iteration"] = per_iteration

    # Predicate kind distribution
    kind_counts: Dict[str, int] = {}
    for p in result.discovered_predicates:
        k = p.kind.name
        kind_counts[k] = kind_counts.get(k, 0) + 1
    empirical["predicate_kind_distribution"] = kind_counts

    # Predicate details
    empirical["discovered_predicates"] = [
        {
            "kind": p.kind.name,
            "tensor": p.tensor,
            "axis": p.axis,
            "value": p.value,
            "pretty": p.pretty(),
        }
        for p in result.discovered_predicates
    ]

    # Universe utilization
    empirical_universe_size = len(result.discovered_predicates)
    utilization = empirical_universe_size / max(theoretical_bound, 1)
    empirical["universe_utilization"] = {
        "empirical_universe_size": empirical_universe_size,
        "theoretical_bound": theoretical_bound,
        "utilization_ratio": round(utilization, 4),
        "converged_within_bound": result.iterations <= theoretical_bound,
        "slack": theoretical_bound - result.iterations,
    }

    # Decidability classification
    if result.verdict == CEGARVerdict.SAFE:
        decidability = "decidable_safe"
    elif result.verdict == CEGARVerdict.UNSAFE:
        decidability = "decidable_unsafe"
    elif result.verdict == CEGARVerdict.TIMEOUT:
        decidability = "undecidable_timeout"
    elif result.verdict == CEGARVerdict.UNKNOWN:
        decidability = "undecidable_unknown"
    else:
        decidability = "unknown"
    empirical["decidability_fragment"] = decidability

    entry["empirical"] = empirical
    return entry


def run_grounding_analysis() -> Dict[str, Any]:
    print("=" * 72)
    print("  CEGAR Convergence Grounding Analysis")
    print("=" * 72)
    print()

    # Enumerate predicate template kinds
    predicate_kinds = [
        {
            "name": pk.name,
            "value": pk.value,
            "description": _predicate_kind_description(pk),
        }
        for pk in PredicateKind
    ]
    print(f"  Predicate template kinds: {len(predicate_kinds)}")
    for pk in predicate_kinds:
        print(f"    {pk['name']:20s} — {pk['description']}")
    print()

    # Run analysis on all benchmarks
    results: List[Dict[str, Any]] = []
    for i, bench in enumerate(CEGAR_BENCHMARKS):
        name = bench["name"]
        print(f"  [{i+1:2d}/{len(CEGAR_BENCHMARKS)}] {name:30s} ", end="", flush=True)
        entry = analyze_benchmark(bench)
        results.append(entry)

        emp = entry.get("empirical")
        if emp:
            bound = entry["theoretical_bound"]["N"]
            iters = emp["iterations"]
            n_preds = emp["total_predicates_discovered"]
            verdict = emp["verdict"]
            print(f"iters={iters:2d}  preds={n_preds:2d}  "
                  f"bound={bound:3d}  verdict={verdict}")
        else:
            print(f"ERROR: {entry.get('error', 'unknown')[:50]}")

    # Aggregate statistics
    decidable_safe = [r for r in results if r.get("empirical", {}).get("decidability_fragment") == "decidable_safe"]
    decidable_unsafe = [r for r in results if r.get("empirical", {}).get("decidability_fragment") == "decidable_unsafe"]
    undecidable = [r for r in results if r.get("empirical", {}).get("decidability_fragment", "").startswith("undecidable")]
    errored = [r for r in results if r.get("empirical") is None]

    # Convergence statistics
    iterations_list = [r["empirical"]["iterations"] for r in results if r.get("empirical")]
    bounds_list = [r["theoretical_bound"]["N"] for r in results if r.get("empirical")]
    utilizations = [
        r["empirical"]["universe_utilization"]["utilization_ratio"]
        for r in results if r.get("empirical")
    ]

    all_within_bound = all(
        r["empirical"]["universe_utilization"]["converged_within_bound"]
        for r in results if r.get("empirical")
    )

    print(f"\n  {'═' * 60}")
    print(f"  SUMMARY")
    print(f"  {'═' * 60}")
    print(f"  Total benchmarks:    {len(results)}")
    print(f"  Decidable (SAFE):    {len(decidable_safe)}")
    print(f"  Decidable (UNSAFE):  {len(decidable_unsafe)}")
    print(f"  Undecidable:         {len(undecidable)}")
    print(f"  Errors:              {len(errored)}")
    print(f"  All within bound:    {all_within_bound}")
    if iterations_list:
        print(f"  Mean iterations:     {sum(iterations_list)/len(iterations_list):.1f}")
        print(f"  Max iterations:      {max(iterations_list)}")
        print(f"  Mean bound:          {sum(bounds_list)/len(bounds_list):.1f}")
        print(f"  Mean utilization:    {sum(utilizations)/len(utilizations):.3f}")

    timestamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    output = {
        "experiment": "cegar_convergence_grounding_analysis",
        "timestamp": timestamp,
        "description": (
            "Grounds the abstract CEGAR convergence bound (N iterations) "
            "in concrete predicate templates, addressing MAJOR reviewer "
            "critique #5. Enumerates predicate kinds, tracks per-benchmark "
            "convergence, and relates empirical iterations to the "
            "theoretical bound N = L × D × K."
        ),
        "predicate_template_kinds": predicate_kinds,
        "num_predicate_kinds": NUM_PREDICATE_KINDS,
        "convergence_theory": {
            "bound_formula": "N = L × D × K",
            "L": "number of layers in nn.Module",
            "D": "maximum shape dimensions per layer",
            "K": f"|PredicateKind| = {NUM_PREDICATE_KINDS}",
            "explanation": (
                "The predicate universe P_prog contains at most L × D × K "
                "candidate predicates. The CEGAR loop adds ≥1 new predicate "
                "per iteration (monotone growth), so it terminates in at "
                "most |P_prog| = L × D × K iterations. This is formalized "
                "as Kleene's fixed-point theorem on the powerset lattice "
                "in cegar_cpa.py and mechanized in lean/TheoryCombination.lean."
            ),
            "mechanized_proof": "cegar_terminates in lean/TheoryCombination.lean",
        },
        "aggregate_statistics": {
            "total_benchmarks": len(results),
            "decidable_safe": len(decidable_safe),
            "decidable_unsafe": len(decidable_unsafe),
            "undecidable": len(undecidable),
            "errors": len(errored),
            "all_converged_within_bound": all_within_bound,
            "mean_iterations": round(sum(iterations_list) / max(len(iterations_list), 1), 2),
            "max_iterations": max(iterations_list) if iterations_list else 0,
            "mean_theoretical_bound": round(sum(bounds_list) / max(len(bounds_list), 1), 2),
            "mean_utilization_ratio": round(sum(utilizations) / max(len(utilizations), 1), 4),
        },
        "decidability_partition": {
            "decidable_safe": [r["benchmark"] for r in decidable_safe],
            "decidable_unsafe": [r["benchmark"] for r in decidable_unsafe],
            "undecidable": [r["benchmark"] for r in undecidable],
        },
        "per_benchmark_results": results,
    }

    return output


def _predicate_kind_description(pk: PredicateKind) -> str:
    descriptions = {
        PredicateKind.DIM_EQ: "tensor.shape[axis] == value",
        PredicateKind.DIM_GT: "tensor.shape[axis] > value",
        PredicateKind.DIM_GE: "tensor.shape[axis] >= value",
        PredicateKind.DIM_DIVISIBLE: "tensor.shape[axis] % divisor == 0",
        PredicateKind.DIM_MATCH: "tensor_a.shape[axis_a] == tensor_b.shape[axis_b]",
        PredicateKind.NDIM_EQ: "len(tensor.shape) == value",
        PredicateKind.SHAPE_EQ: "tensor.shape == (d0, d1, ...)",
    }
    return descriptions.get(pk, "unknown")


def main():
    output = run_grounding_analysis()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_FILE, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n  Results saved → {RESULTS_FILE}")


if __name__ == "__main__":
    main()
