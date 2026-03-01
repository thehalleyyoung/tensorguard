"""
Guard Harvesting Contribution Analysis.

Quantifies the contribution of each predicate source in TensorGuard's
contract discovery pipeline:

  (a) Guard harvesting: predicates from isinstance, assert, conditional checks
  (b) Layer parameters: predicates from Conv2d/Linear/etc. constructor args
  (c) Unsat-core CEGAR: predicates discovered via counterexample-guided refinement

For each benchmark with symbolic inputs, runs CEGAR and classifies every
discovered predicate by its origin.
"""

from __future__ import annotations

import json
import os
import sys
import time
from collections import defaultdict
from typing import Any, Dict, List, Tuple

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.model_checker import extract_computation_graph, LayerKind, OpKind
from src.shape_cegar import (
    PredicateKind,
    ShapePredicate,
    run_shape_cegar,
)
from src.guard_extractor import extract_guards

# Benchmarks that exercise symbolic inputs (where CEGAR discovers predicates)
SYMBOLIC_BENCHMARKS: List[Dict[str, Any]] = [
    {
        "name": "mlp_2layer",
        "code": """\
import torch.nn as nn
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(768, 256)
        self.fc2 = nn.Linear(256, 10)
    def forward(self, x):
        return self.fc2(self.fc1(x))
""",
        "input_shapes": {"x": ("batch", "features")},
    },
    {
        "name": "mlp_3layer",
        "code": """\
import torch.nn as nn
class MLP3(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(512, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 10)
    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        return self.fc3(x)
""",
        "input_shapes": {"x": ("batch", "d")},
    },
    {
        "name": "conv_block",
        "code": """\
import torch.nn as nn
class ConvBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 64, 3, padding=1)
        self.bn = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))
""",
        "input_shapes": {"x": ("batch", "channels", "h", "w")},
    },
    {
        "name": "encoder_block",
        "code": """\
import torch.nn as nn
class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln = nn.LayerNorm(512)
        self.attn = nn.MultiheadAttention(512, 8, batch_first=True)
        self.fc1 = nn.Linear(512, 2048)
        self.fc2 = nn.Linear(2048, 512)
    def forward(self, x):
        a = self.ln(x)
        a, _ = self.attn(a, a, a)
        x = x + a
        return x + self.fc2(self.fc1(self.ln(x)))
""",
        "input_shapes": {"x": ("batch", "seq", "d_model")},
    },
    {
        "name": "residual_mlp",
        "code": """\
import torch.nn as nn
class ResidualMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(256, 256)
        self.fc2 = nn.Linear(256, 256)
        self.relu = nn.ReLU()
    def forward(self, x):
        return x + self.relu(self.fc2(self.relu(self.fc1(x))))
""",
        "input_shapes": {"x": ("batch", "d")},
    },
    {
        "name": "double_conv",
        "code": """\
import torch.nn as nn
class DoubleConv(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(128)
    def forward(self, x):
        x = self.bn1(self.conv1(x))
        return self.bn2(self.conv2(x))
""",
        "input_shapes": {"x": ("batch", "c", "h", "w")},
    },
    {
        "name": "linear_chain_5",
        "code": """\
import torch.nn as nn
class Chain5(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 64)
        self.fc5 = nn.Linear(64, 10)
    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)
        return self.fc5(x)
""",
        "input_shapes": {"x": ("batch", "d")},
    },
    {
        "name": "embedding_classifier",
        "code": """\
import torch.nn as nn
class EmbeddingClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(10000, 128)
        self.fc = nn.Linear(128, 10)
    def forward(self, x):
        return self.fc(self.embed(x))
""",
        "input_shapes": {"x": ("batch", "seq")},
    },
    {
        "name": "conv_flatten_fc",
        "code": """\
import torch.nn as nn
class ConvFC(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 32, 3, padding=1)
        self.fc = nn.Linear(32, 10)
    def forward(self, x):
        x = self.conv(x)
        return self.fc(x)
""",
        "input_shapes": {"x": ("batch", "c", "h", "w")},
    },
    {
        "name": "norm_variations",
        "code": """\
import torch.nn as nn
class NormTest(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln = nn.LayerNorm(256)
        self.fc = nn.Linear(256, 256)
    def forward(self, x):
        return self.fc(self.ln(x))
""",
        "input_shapes": {"x": ("batch", "seq", "d")},
    },
    {
        "name": "projection_pair",
        "code": """\
import torch.nn as nn
class ProjectionPair(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj_q = nn.Linear(512, 64)
        self.proj_k = nn.Linear(512, 64)
    def forward(self, x):
        q = self.proj_q(x)
        k = self.proj_k(x)
        return q + k
""",
        "input_shapes": {"x": ("batch", "seq", "d")},
    },
    {
        "name": "deep_mlp_10",
        "code": """\
import torch.nn as nn
class DeepMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(128, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 128)
        self.fc4 = nn.Linear(128, 128)
        self.fc5 = nn.Linear(128, 128)
        self.fc6 = nn.Linear(128, 128)
        self.fc7 = nn.Linear(128, 128)
        self.fc8 = nn.Linear(128, 128)
        self.fc9 = nn.Linear(128, 128)
        self.fc10 = nn.Linear(128, 10)
    def forward(self, x):
        for fc in [self.fc1, self.fc2, self.fc3, self.fc4, self.fc5,
                    self.fc6, self.fc7, self.fc8, self.fc9]:
            x = fc(x)
        return self.fc10(x)
""",
        "input_shapes": {"x": ("batch", "d")},
    },
    {
        "name": "guarded_model",
        "code": """\
import torch.nn as nn
class GuardedModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, 10)
    def forward(self, x):
        assert x.shape[-1] == 256
        x = self.fc1(x)
        return self.fc2(x)
""",
        "input_shapes": {"x": ("batch", "d")},
    },
    {
        "name": "isinstance_guard",
        "code": """\
import torch
import torch.nn as nn
class IsinstGuard(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(64, 10)
    def forward(self, x):
        if isinstance(x, torch.Tensor):
            return self.fc(x)
        return x
""",
        "input_shapes": {"x": ("batch", "d")},
    },
    {
        "name": "multi_guard",
        "code": """\
import torch
import torch.nn as nn
class MultiGuard(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(512, 256)
        self.fc2 = nn.Linear(256, 10)
    def forward(self, x):
        assert len(x.shape) == 2
        assert x.shape[-1] == 512
        x = self.fc1(x)
        return self.fc2(x)
""",
        "input_shapes": {"x": ("batch", "d")},
    },
]


def classify_predicate_source(
    pred: ShapePredicate,
    graph,
    guard_preds: List[Dict],
) -> str:
    """Classify a discovered predicate's origin.

    Returns one of:
      - 'guard_harvest': predicate matches an extracted guard
      - 'layer_param': predicate encodes a layer parameter constraint
      - 'cegar_unsat_core': predicate from CEGAR counterexample refinement
    """
    # Check if it matches a guard-harvested predicate
    for gp in guard_preds:
        if pred.kind == PredicateKind.DIM_EQ and gp.get("kind") == "assertion":
            if pred.tensor and pred.value is not None:
                if str(pred.value) in str(gp.get("expression", "")):
                    return "guard_harvest"
        if gp.get("kind") in ("isinstance", "type_check"):
            return "guard_harvest"

    # Check if the predicate corresponds to a layer parameter
    for layer_name, layer in graph.layers.items():
        if pred.kind == PredicateKind.DIM_EQ and pred.value is not None:
            # Linear layer: in_features
            if layer.kind == LayerKind.LINEAR and layer.in_features is not None:
                if pred.value == layer.in_features:
                    return "layer_param"
            # Conv2d: in_channels
            if layer.kind == LayerKind.CONV2D and layer.in_channels is not None:
                if pred.value == layer.in_channels:
                    return "layer_param"
            # LayerNorm: normalized_shape
            if layer.kind == LayerKind.LAYERNORM:
                norm_shape = getattr(layer, "normalized_shape", None)
                if norm_shape and pred.value in (
                    norm_shape if isinstance(norm_shape, (list, tuple))
                    else [norm_shape]
                ):
                    return "layer_param"
            # Embedding: embedding_dim
            if layer.kind == LayerKind.EMBEDDING:
                edim = getattr(layer, "embedding_dim", None)
                if edim is not None and pred.value == edim:
                    return "layer_param"
            # MHA: embed_dim
            if layer.kind == LayerKind.MULTIHEAD_ATTENTION:
                edim = getattr(layer, "embed_dim", None)
                if edim is not None and pred.value == edim:
                    return "layer_param"
            # BatchNorm: num_features
            if layer.kind in (LayerKind.BATCHNORM1D, LayerKind.BATCHNORM2D):
                nf = getattr(layer, "num_features", None)
                if nf is not None and pred.value == nf:
                    return "layer_param"

        # DIM_MATCH typically from layer param compatibility
        if pred.kind == PredicateKind.DIM_MATCH:
            return "layer_param"

        # NDIM_EQ often from layer param requirements
        if pred.kind == PredicateKind.NDIM_EQ:
            return "layer_param"

        # DIM_DIVISIBLE from MHA num_heads
        if pred.kind == PredicateKind.DIM_DIVISIBLE:
            if layer.kind == LayerKind.MULTIHEAD_ATTENTION:
                nh = getattr(layer, "num_heads", None)
                if nh is not None and pred.divisor == nh:
                    return "layer_param"

    # Default: from CEGAR unsat core refinement
    return "cegar_unsat_core"


def extract_guard_predicates(code: str) -> List[Dict]:
    """Extract guard predicates from source code."""
    try:
        guards = extract_guards(code)
        return [
            {
                "kind": g.pattern.name.lower() if hasattr(g, "pattern") else "unknown",
                "expression": str(g),
            }
            for g in guards
        ]
    except Exception:
        return []


def run_analysis() -> Dict[str, Any]:
    """Run guard harvesting analysis on all symbolic benchmarks."""
    results = {
        "benchmarks": [],
        "aggregate": {
            "total_predicates": 0,
            "guard_harvest": 0,
            "layer_param": 0,
            "cegar_unsat_core": 0,
        },
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }

    for bench in SYMBOLIC_BENCHMARKS:
        print(f"  Analyzing: {bench['name']}...", end=" ", flush=True)

        try:
            graph = extract_computation_graph(bench["code"])
        except Exception as e:
            print(f"SKIP (parse error: {e})")
            continue

        # Extract guards from code
        guard_preds = extract_guard_predicates(bench["code"])

        # Run CEGAR to discover predicates
        try:
            cegar_result = run_shape_cegar(
                bench["code"],
                input_shapes=bench["input_shapes"],
                max_iterations=10,
            )
        except Exception as e:
            print(f"SKIP (CEGAR error: {e})")
            continue

        # Classify each discovered predicate
        source_counts = defaultdict(int)
        pred_details = []

        for pred in cegar_result.discovered_predicates:
            source = classify_predicate_source(pred, graph, guard_preds)
            source_counts[source] += 1
            pred_details.append({
                "predicate": pred.pretty(),
                "kind": pred.kind.name,
                "source": source,
            })

        total = sum(source_counts.values())
        bench_result = {
            "name": bench["name"],
            "total_predicates": total,
            "guard_harvest": source_counts.get("guard_harvest", 0),
            "layer_param": source_counts.get("layer_param", 0),
            "cegar_unsat_core": source_counts.get("cegar_unsat_core", 0),
            "num_guards_in_source": len(guard_preds),
            "cegar_iterations": cegar_result.iterations,
            "cegar_status": cegar_result.final_status.name,
            "predicates": pred_details,
        }
        results["benchmarks"].append(bench_result)

        # Update aggregate
        results["aggregate"]["total_predicates"] += total
        results["aggregate"]["guard_harvest"] += source_counts.get("guard_harvest", 0)
        results["aggregate"]["layer_param"] += source_counts.get("layer_param", 0)
        results["aggregate"]["cegar_unsat_core"] += source_counts.get("cegar_unsat_core", 0)

        print(f"OK ({total} preds: "
              f"guard={source_counts.get('guard_harvest', 0)}, "
              f"layer={source_counts.get('layer_param', 0)}, "
              f"cegar={source_counts.get('cegar_unsat_core', 0)})")

    # Compute percentages
    total = results["aggregate"]["total_predicates"]
    if total > 0:
        results["aggregate"]["guard_harvest_pct"] = round(
            100 * results["aggregate"]["guard_harvest"] / total, 1
        )
        results["aggregate"]["layer_param_pct"] = round(
            100 * results["aggregate"]["layer_param"] / total, 1
        )
        results["aggregate"]["cegar_unsat_core_pct"] = round(
            100 * results["aggregate"]["cegar_unsat_core"] / total, 1
        )
    else:
        results["aggregate"]["guard_harvest_pct"] = 0
        results["aggregate"]["layer_param_pct"] = 0
        results["aggregate"]["cegar_unsat_core_pct"] = 0

    return results


if __name__ == "__main__":
    print("=" * 60)
    print("Guard Harvesting Contribution Analysis")
    print("=" * 60)
    results = run_analysis()

    out_path = os.path.join(
        os.path.dirname(__file__),
        "guard_harvest_results.json",
    )
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 60)
    print("AGGREGATE RESULTS")
    print("=" * 60)
    agg = results["aggregate"]
    print(f"Total predicates discovered: {agg['total_predicates']}")
    print(f"  Guard harvesting:    {agg['guard_harvest']:3d} ({agg['guard_harvest_pct']:.1f}%)")
    print(f"  Layer parameters:    {agg['layer_param']:3d} ({agg['layer_param_pct']:.1f}%)")
    print(f"  CEGAR unsat cores:   {agg['cegar_unsat_core']:3d} ({agg['cegar_unsat_core_pct']:.1f}%)")
    print(f"\nResults saved to: {out_path}")
