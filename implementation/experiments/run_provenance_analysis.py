"""
Predicate Provenance Analysis for TensorGuard.

Runs TensorGuard on a representative set of benchmarks from Suite B and Suite D,
tracks the provenance of each discovered predicate, and computes aggregate
statistics showing what percentage of constraints come from each source:

  - explicit_guard:    from user-written if/assert statements
  - api_stub:          from PyTorch API stub models (layer definitions in __init__)
  - pattern_matching:  from AST pattern matching (e.g., torch.cat, reshape patterns)
  - cegar_discovered:  from CEGAR loop counterexample-guided refinement
"""

from __future__ import annotations

import json
import os
import sys
import time
from collections import defaultdict
from typing import Any, Dict, List

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.model_checker import extract_computation_graph, LayerKind, OpKind
from src.shape_cegar import (
    PredicateKind,
    ShapePredicate,
    run_shape_cegar,
)
from src.guard_extractor import extract_guards, PredicateKind as GuardPredKind

RESULTS_PATH = os.path.join(os.path.dirname(__file__), "provenance_analysis_results.json")

# ── Representative benchmarks (12 from Suite B, 12 from Suite D) ────────

PROVENANCE_BENCHMARKS: List[Dict[str, Any]] = [
    # === Suite B: Expanded Evaluation ===
    {
        "name": "B_mlp_block",
        "suite": "B",
        "code": """\
import torch.nn as nn
class MLPBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(768, 3072)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(3072, 768)
    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))
""",
        "input_shapes": {"x": ("batch", "seq", 768)},
    },
    {
        "name": "B_prenorm_residual",
        "suite": "B",
        "code": """\
import torch.nn as nn
class PreNormResidual(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm = nn.LayerNorm(768)
        self.fc = nn.Linear(768, 768)
    def forward(self, x):
        return x + self.fc(self.norm(x))
""",
        "input_shapes": {"x": ("batch", "seq", 768)},
    },
    {
        "name": "B_conv_bn_relu",
        "suite": "B",
        "code": """\
import torch.nn as nn
class ConvBNReLU(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 64, 3, padding=1)
        self.bn = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))
""",
        "input_shapes": {"x": ("batch", 3, "h", "w")},
    },
    {
        "name": "B_double_conv",
        "suite": "B",
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
        "input_shapes": {"x": ("batch", 3, "h", "w")},
    },
    {
        "name": "B_linear_chain_5",
        "suite": "B",
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
        "name": "B_embedding_classifier",
        "suite": "B",
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
        "name": "B_residual_mlp",
        "suite": "B",
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
        "name": "B_projection_pair",
        "suite": "B",
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
        "name": "B_deep_mlp_10",
        "suite": "B",
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
        "name": "B_guarded_model",
        "suite": "B",
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
        "name": "B_isinstance_guard",
        "suite": "B",
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
        "name": "B_multi_assert",
        "suite": "B",
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
    # === Suite D: External Benchmark Suite ===
    {
        "name": "D_simple_classifier",
        "suite": "D",
        "code": """\
import torch.nn as nn
class SimpleClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 10)
        self.relu = nn.ReLU()
    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))
""",
        "input_shapes": {"x": ("batch", "features")},
    },
    {
        "name": "D_autoencoder",
        "suite": "D",
        "code": """\
import torch.nn as nn
class Autoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Linear(784, 64)
        self.decoder = nn.Linear(64, 784)
    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)
""",
        "input_shapes": {"x": ("batch", "d")},
    },
    {
        "name": "D_resnet_block",
        "suite": "D",
        "code": """\
import torch.nn as nn
class ResBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(64, 64, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + residual)
""",
        "input_shapes": {"x": ("batch", 64, "h", "w")},
    },
    {
        "name": "D_transformer_ffn",
        "suite": "D",
        "code": """\
import torch.nn as nn
class TransformerFFN(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(512, 2048)
        self.linear2 = nn.Linear(2048, 512)
        self.norm = nn.LayerNorm(512)
        self.relu = nn.ReLU()
    def forward(self, x):
        return self.norm(x + self.linear2(self.relu(self.linear1(x))))
""",
        "input_shapes": {"x": ("batch", "seq", "d")},
    },
    {
        "name": "D_vgg_block",
        "suite": "D",
        "code": """\
import torch.nn as nn
class VGGBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
        self.conv2 = nn.Conv2d(64, 64, 3, padding=1)
        self.relu = nn.ReLU()
    def forward(self, x):
        x = self.relu(self.conv1(x))
        return self.relu(self.conv2(x))
""",
        "input_shapes": {"x": ("batch", 3, "h", "w")},
    },
    {
        "name": "D_dim_mismatch_bug",
        "suite": "D",
        "code": """\
import torch.nn as nn
class DimBug(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(100, 50)
        self.fc2 = nn.Linear(60, 10)
    def forward(self, x):
        x = self.fc1(x)
        return self.fc2(x)
""",
        "input_shapes": {"x": ("batch", "d")},
        "has_bug": True,
    },
    {
        "name": "D_norm_ffn_stack",
        "suite": "D",
        "code": """\
import torch.nn as nn
class NormFFN(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm = nn.LayerNorm(256)
        self.fc1 = nn.Linear(256, 1024)
        self.fc2 = nn.Linear(1024, 256)
        self.relu = nn.ReLU()
    def forward(self, x):
        x = self.norm(x)
        return x + self.fc2(self.relu(self.fc1(x)))
""",
        "input_shapes": {"x": ("batch", "seq", "d")},
    },
    {
        "name": "D_conv_channel_bug",
        "suite": "D",
        "code": """\
import torch.nn as nn
class ChannelBug(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(64, 64, 3, padding=1)
    def forward(self, x):
        x = self.conv1(x)
        return self.conv2(x)
""",
        "input_shapes": {"x": ("batch", 3, "h", "w")},
        "has_bug": True,
    },
    {
        "name": "D_bottleneck",
        "suite": "D",
        "code": """\
import torch.nn as nn
class Bottleneck(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(256, 64, 1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 256, 1)
        self.bn3 = nn.BatchNorm2d(256)
        self.relu = nn.ReLU()
    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        return self.relu(out + identity)
""",
        "input_shapes": {"x": ("batch", 256, "h", "w")},
    },
    {
        "name": "D_guarded_classifier",
        "suite": "D",
        "code": """\
import torch.nn as nn
class GuardedClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(128, 10)
    def forward(self, x):
        if x is None:
            return None
        assert x.shape[-1] == 128
        return self.fc(x)
""",
        "input_shapes": {"x": ("batch", "d")},
    },
    # === Additional benchmarks for broader provenance coverage ===
    {
        "name": "B_encoder_block",
        "suite": "B",
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
        "name": "D_three_layer_conv",
        "suite": "D",
        "code": """\
import torch.nn as nn
class ThreeConv(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.relu = nn.ReLU()
    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        return self.relu(self.conv3(x))
""",
        "input_shapes": {"x": ("batch", "c", "h", "w")},
    },
    {
        "name": "D_wide_mlp",
        "suite": "D",
        "code": """\
import torch.nn as nn
class WideMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(2048, 4096)
        self.fc2 = nn.Linear(4096, 2048)
        self.fc3 = nn.Linear(2048, 10)
    def forward(self, x):
        assert x.shape[-1] == 2048
        x = self.fc1(x)
        x = self.fc2(x)
        return self.fc3(x)
""",
        "input_shapes": {"x": ("batch", "d")},
    },
]


def classify_predicate_provenance(
    pred: ShapePredicate,
    graph,
    guard_preds: List[Dict],
) -> str:
    """Classify a discovered predicate's provenance into four categories.

    Returns one of: explicit_guard, api_stub, pattern_matching, cegar_discovered.
    """
    # 1. If the predicate already has provenance set from synthesis, use it
    if hasattr(pred, "provenance") and pred.provenance == "api_stub":
        return "api_stub"

    # 2. Check if it matches a guard-harvested predicate (explicit_guard)
    for gp in guard_preds:
        if pred.kind == PredicateKind.DIM_EQ and gp.get("kind") in ("assertion", "comparison"):
            if pred.tensor and pred.value is not None:
                if str(pred.value) in str(gp.get("expression", "")):
                    return "explicit_guard"
        if gp.get("kind") in ("isinstance", "type_check", "identity"):
            if pred.tensor and gp.get("variable") == pred.tensor:
                return "explicit_guard"

    # 3. Check for pattern_matching (predicates from reshape/cat/view op shapes)
    for step in graph.steps:
        if step.op in (OpKind.CAT, OpKind.RESHAPE):
            if pred.tensor in step.inputs:
                return "pattern_matching"
        if step.op == OpKind.MATMUL and pred.kind == PredicateKind.DIM_MATCH:
            if pred.tensor in step.inputs:
                return "pattern_matching"
        if step.op == OpKind.TRANSPOSE and pred.tensor in step.inputs:
            if pred.kind in (PredicateKind.NDIM_EQ, PredicateKind.DIM_MATCH):
                return "pattern_matching"

    # 4. Check if the predicate corresponds to a layer parameter (api_stub)
    for layer_name, layer in graph.layers.items():
        if pred.kind == PredicateKind.DIM_EQ and pred.value is not None:
            if layer.kind == LayerKind.LINEAR and layer.in_features is not None:
                if pred.value == layer.in_features:
                    return "api_stub"
            if layer.kind == LayerKind.CONV2D and layer.in_channels is not None:
                if pred.value == layer.in_channels:
                    return "api_stub"
            if layer.kind == LayerKind.LAYERNORM:
                norm_shape = getattr(layer, "normalized_shape", None)
                if norm_shape and pred.value in (
                    norm_shape if isinstance(norm_shape, (list, tuple)) else [norm_shape]
                ):
                    return "api_stub"
            if layer.kind == LayerKind.EMBEDDING:
                edim = getattr(layer, "embedding_dim", None)
                if edim is not None and pred.value == edim:
                    return "api_stub"
            if layer.kind == LayerKind.MULTIHEAD_ATTENTION:
                edim = getattr(layer, "embed_dim", None)
                if edim is not None and pred.value == edim:
                    return "api_stub"
            if layer.kind in (LayerKind.BATCHNORM1D, LayerKind.BATCHNORM2D):
                nf = getattr(layer, "num_features", None)
                if nf is not None and pred.value == nf:
                    return "api_stub"

        if pred.kind == PredicateKind.DIM_MATCH:
            return "api_stub"
        if pred.kind == PredicateKind.NDIM_EQ:
            return "api_stub"
        if pred.kind == PredicateKind.DIM_DIVISIBLE:
            if layer.kind == LayerKind.MULTIHEAD_ATTENTION:
                nh = getattr(layer, "num_heads", None)
                if nh is not None and pred.divisor == nh:
                    return "api_stub"

    # 5. Default: CEGAR-discovered
    return "cegar_discovered"


def extract_guard_predicates(code: str) -> List[Dict]:
    """Extract guard predicates from source code and return summaries."""
    try:
        guards = extract_guards(code)
        return [
            {
                "kind": g.pattern.name.lower() if hasattr(g, "pattern") else "unknown",
                "expression": g.raw_source,
                "variable": g.variables[0] if g.variables else "",
                "provenance": g.predicate.provenance,
            }
            for g in guards
        ]
    except Exception:
        return []


def run_provenance_analysis() -> Dict[str, Any]:
    """Run provenance analysis on all benchmarks."""
    categories = ["explicit_guard", "api_stub", "pattern_matching", "cegar_discovered"]

    results: Dict[str, Any] = {
        "benchmarks": [],
        "aggregate": {cat: 0 for cat in categories},
        "aggregate_pct": {},
        "total_predicates": 0,
        "num_benchmarks_analyzed": 0,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }

    for bench in PROVENANCE_BENCHMARKS:
        name = bench["name"]
        print(f"  [{name}]...", end=" ", flush=True)

        try:
            graph = extract_computation_graph(bench["code"])
        except Exception as e:
            print(f"SKIP (parse: {e})")
            continue

        guard_preds = extract_guard_predicates(bench["code"])

        try:
            cegar_result = run_shape_cegar(
                bench["code"],
                input_shapes=bench.get("input_shapes"),
                max_iterations=10,
            )
        except Exception as e:
            print(f"SKIP (CEGAR: {e})")
            continue

        # Classify each discovered predicate
        source_counts: Dict[str, int] = {cat: 0 for cat in categories}
        pred_details = []

        for pred in cegar_result.discovered_predicates:
            prov = classify_predicate_provenance(pred, graph, guard_preds)
            source_counts[prov] += 1
            pred_details.append({
                "predicate": pred.pretty(),
                "kind": pred.kind.name,
                "provenance": prov,
            })

        # Also count guard-extracted predicates that don't appear in CEGAR
        # (they still contribute to the overall predicate landscape)
        for gp in guard_preds:
            prov = gp.get("provenance", "explicit_guard")
            if prov == "explicit_guard":
                source_counts["explicit_guard"] += 1
                pred_details.append({
                    "predicate": gp.get("expression", ""),
                    "kind": gp.get("kind", "guard"),
                    "provenance": "explicit_guard",
                })

        total = sum(source_counts.values())
        bench_result = {
            "name": name,
            "suite": bench.get("suite", ""),
            "total_predicates": total,
            "has_bug": bench.get("has_bug", False),
            "cegar_status": cegar_result.final_status.name,
            "cegar_iterations": cegar_result.iterations,
            "provenance_counts": dict(source_counts),
            "predicates": pred_details,
        }
        results["benchmarks"].append(bench_result)
        results["num_benchmarks_analyzed"] += 1

        for cat in categories:
            results["aggregate"][cat] += source_counts[cat]
        results["total_predicates"] += total

        print(f"OK ({total} preds: "
              + ", ".join(f"{c}={source_counts[c]}" for c in categories if source_counts[c])
              + ")")

    # Compute percentages
    total = results["total_predicates"]
    for cat in categories:
        if total > 0:
            results["aggregate_pct"][cat] = round(
                100 * results["aggregate"][cat] / total, 1
            )
        else:
            results["aggregate_pct"][cat] = 0.0

    return results


if __name__ == "__main__":
    print("=" * 70)
    print("Predicate Provenance Analysis for TensorGuard")
    print("=" * 70)
    results = run_provenance_analysis()

    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 70)
    print("AGGREGATE PROVENANCE BREAKDOWN")
    print("=" * 70)
    print(f"Benchmarks analyzed: {results['num_benchmarks_analyzed']}")
    print(f"Total predicates:    {results['total_predicates']}")
    print()
    for cat in ["explicit_guard", "api_stub", "pattern_matching", "cegar_discovered"]:
        count = results["aggregate"][cat]
        pct = results["aggregate_pct"][cat]
        print(f"  {cat:20s}: {count:4d}  ({pct:5.1f}%)")

    print(f"\nResults saved to: {RESULTS_PATH}")
