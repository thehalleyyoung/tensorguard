"""
DAG Composition Evaluation Experiment.

Tests the DAG compositional verification framework on non-sequential
architectures: ResNet, U-Net, DenseNet, Transformer, and FPN.
Compares sequential vs DAG decomposition and reports results.
"""

import json
import os
import sys
import time
import traceback

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.model_checker import verify_model, extract_computation_graph
from src.assume_guarantee import (
    verify_compositional,
    verify_compositional_dag,
    decompose_graph,
    decompose_graph_dag,
    DAGCompositionProofRule,
    validate_interface_dag,
    reset_default_cache,
)

# ─── Architecture definitions ────────────────────────────────────────────────

RESNET_BLOCK = """\
import torch
import torch.nn as nn

class ResNetBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(64, 64, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = out + identity
        out = self.relu(out)
        return out
"""

UNET_BLOCK = """\
import torch
import torch.nn as nn

class UNetBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc_conv1 = nn.Conv2d(3, 64, 3, padding=1)
        self.enc_bn1 = nn.BatchNorm2d(64)
        self.enc_conv2 = nn.Conv2d(64, 128, 3, padding=1)
        self.enc_bn2 = nn.BatchNorm2d(128)
        self.dec_conv1 = nn.Conv2d(128, 64, 3, padding=1)
        self.dec_bn1 = nn.BatchNorm2d(64)
        self.dec_conv2 = nn.Conv2d(64, 3, 3, padding=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        e1 = self.enc_conv1(x)
        e1 = self.enc_bn1(e1)
        e1 = self.relu(e1)
        e2 = self.enc_conv2(e1)
        e2 = self.enc_bn2(e2)
        e2 = self.relu(e2)
        d1 = self.dec_conv1(e2)
        d1 = self.dec_bn1(d1)
        d1 = self.relu(d1)
        d2 = self.dec_conv2(d1)
        return d2
"""

DENSENET_BLOCK = """\
import torch
import torch.nn as nn

class DenseBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(64, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(96, 32, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU()

    def forward(self, x):
        out1 = self.conv1(x)
        out1 = self.bn1(out1)
        out1 = self.relu(out1)
        cat1 = torch.cat([x, out1], dim=1)
        out2 = self.conv2(cat1)
        out2 = self.bn2(out2)
        out2 = self.relu(out2)
        return out2
"""

TRANSFORMER_BLOCK = """\
import torch
import torch.nn as nn

class TransformerBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc_linear = nn.Linear(512, 512)
        self.dec_linear = nn.Linear(512, 512)
        self.cross_attn = nn.MultiheadAttention(512, 8)
        self.fc_out = nn.Linear(512, 512)
        self.norm1 = nn.LayerNorm(512)
        self.norm2 = nn.LayerNorm(512)

    def forward(self, x):
        enc = self.enc_linear(x)
        enc = self.norm1(enc)
        dec = self.dec_linear(x)
        dec = self.norm2(dec)
        out = self.fc_out(dec)
        return out
"""

FPN_BLOCK = """\
import torch
import torch.nn as nn

class FPNBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.bottom_up1 = nn.Conv2d(3, 64, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.bottom_up2 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.lateral1 = nn.Conv2d(64, 128, 1)
        self.top_down = nn.Conv2d(128, 128, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.relu = nn.ReLU()

    def forward(self, x):
        c1 = self.bottom_up1(x)
        c1 = self.bn1(c1)
        c1 = self.relu(c1)
        c2 = self.bottom_up2(c1)
        c2 = self.bn2(c2)
        c2 = self.relu(c2)
        lat = self.lateral1(c1)
        out = self.top_down(c2)
        out = self.bn3(out)
        out = self.relu(out)
        return out
"""

ARCHITECTURES = [
    {
        "name": "ResNet Block",
        "type": "residual",
        "source": RESNET_BLOCK,
        "input_shapes": {"x": ("batch", 64, 32, 32)},
    },
    {
        "name": "U-Net Block",
        "type": "encoder_decoder",
        "source": UNET_BLOCK,
        "input_shapes": {"x": ("batch", 3, 256, 256)},
    },
    {
        "name": "DenseNet Block",
        "type": "dense",
        "source": DENSENET_BLOCK,
        "input_shapes": {"x": ("batch", 64, 32, 32)},
    },
    {
        "name": "Transformer Block",
        "type": "general_dag",
        "source": TRANSFORMER_BLOCK,
        "input_shapes": {"x": ("batch", 10, 512)},
    },
    {
        "name": "FPN Block",
        "type": "general_dag",
        "source": FPN_BLOCK,
        "input_shapes": {"x": ("batch", 3, 64, 64)},
    },
]


def evaluate_architecture(arch):
    """Evaluate a single architecture with both sequential and DAG decomposition."""
    name = arch["name"]
    source = arch["source"]
    input_shapes = arch["input_shapes"]
    result = {
        "name": name,
        "type": arch["type"],
    }

    # Step 1: Monolithic verification
    reset_default_cache()
    try:
        t0 = time.monotonic()
        mono = verify_model(source, input_shapes=input_shapes)
        mono_time = (time.monotonic() - t0) * 1000
        result["monolithic"] = {
            "safe": mono.safe,
            "time_ms": round(mono_time, 2),
            "num_errors": len(mono.errors) if mono.errors else 0,
        }
    except Exception as e:
        result["monolithic"] = {"error": str(e)}

    # Step 2: Sequential compositional verification
    reset_default_cache()
    try:
        t0 = time.monotonic()
        seq = verify_compositional(
            source, input_shapes=input_shapes, measure_monolithic=False,
        )
        seq_time = (time.monotonic() - t0) * 1000
        result["sequential_compositional"] = {
            "safe": seq.safe,
            "time_ms": round(seq_time, 2),
            "num_submodules": seq.num_submodules,
            "interface_checks": len(seq.interface_checks),
            "all_interfaces_ok": all(ic.compatible for ic in seq.interface_checks),
        }
    except Exception as e:
        result["sequential_compositional"] = {"error": str(e)}

    # Step 3: DAG compositional verification
    reset_default_cache()
    try:
        t0 = time.monotonic()
        dag = verify_compositional_dag(
            source, input_shapes=input_shapes, measure_monolithic=False,
        )
        dag_time = (time.monotonic() - t0) * 1000

        # Also get decomposition details
        graph = extract_computation_graph(source)
        subs, edges, topology = decompose_graph_dag(graph, input_shapes)
        dag_rule = DAGCompositionProofRule.from_submodules_and_edges(
            subs, edges, topology,
        )

        result["dag_compositional"] = {
            "safe": dag.safe,
            "time_ms": round(dag_time, 2),
            "num_submodules": dag.num_submodules,
            "num_edges": len(edges),
            "topology": topology,
            "interface_checks": len(dag.interface_checks),
            "all_interfaces_ok": all(ic.compatible for ic in dag.interface_checks),
            "topological_order": dag_rule.topological_order(),
            "edges": edges,
        }
    except Exception as e:
        result["dag_compositional"] = {"error": str(e), "traceback": traceback.format_exc()}

    # Step 4: Agreement check
    mono_safe = result.get("monolithic", {}).get("safe")
    seq_safe = result.get("sequential_compositional", {}).get("safe")
    dag_safe = result.get("dag_compositional", {}).get("safe")

    result["agreement"] = {
        "mono_vs_seq": mono_safe == seq_safe if mono_safe is not None and seq_safe is not None else None,
        "mono_vs_dag": mono_safe == dag_safe if mono_safe is not None and dag_safe is not None else None,
        "seq_vs_dag": seq_safe == dag_safe if seq_safe is not None and dag_safe is not None else None,
    }

    return result


def main():
    print("=" * 70)
    print("DAG Composition Evaluation")
    print("=" * 70)
    print()

    all_results = []
    summary = {
        "total_architectures": len(ARCHITECTURES),
        "dag_successes": 0,
        "dag_failures": 0,
        "agreement_count": 0,
        "disagreement_count": 0,
    }

    for arch in ARCHITECTURES:
        print(f"▶ Evaluating: {arch['name']} ({arch['type']})")
        try:
            result = evaluate_architecture(arch)
            all_results.append(result)

            dag_info = result.get("dag_compositional", {})
            if "error" not in dag_info:
                summary["dag_successes"] += 1
                print(f"  DAG: safe={dag_info.get('safe')}, "
                      f"topology={dag_info.get('topology')}, "
                      f"nodes={dag_info.get('num_submodules')}, "
                      f"edges={dag_info.get('num_edges')}")
            else:
                summary["dag_failures"] += 1
                print(f"  DAG: ERROR - {dag_info['error'][:80]}")

            agreement = result.get("agreement", {})
            if agreement.get("mono_vs_dag") is True:
                summary["agreement_count"] += 1
            elif agreement.get("mono_vs_dag") is False:
                summary["disagreement_count"] += 1

        except Exception as e:
            print(f"  FAILED: {e}")
            all_results.append({"name": arch["name"], "error": str(e)})
            summary["dag_failures"] += 1

        print()

    # Summary
    print("=" * 70)
    print("Summary")
    print("=" * 70)
    print(f"  Total architectures:    {summary['total_architectures']}")
    print(f"  DAG successes:          {summary['dag_successes']}")
    print(f"  DAG failures:           {summary['dag_failures']}")
    print(f"  Mono-DAG agreements:    {summary['agreement_count']}")
    print(f"  Mono-DAG disagreements: {summary['disagreement_count']}")

    # Save results
    output = {
        "experiment": "dag_composition_eval",
        "summary": summary,
        "results": all_results,
    }

    benchmarks_dir = os.path.join(os.path.dirname(__file__), "..", ".benchmarks")
    os.makedirs(benchmarks_dir, exist_ok=True)
    output_path = os.path.join(benchmarks_dir, "dag_composition_results.json")
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
