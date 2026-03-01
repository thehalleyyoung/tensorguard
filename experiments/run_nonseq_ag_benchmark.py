"""
Non-Sequential Assume-Guarantee Benchmark.

Exercises the BRANCH_MERGE decomposition strategy on five non-sequential
neural-network architectures:
  1. ResNet-50 with residual skip connections
  2. U-Net with encoder-decoder skip connections
  3. Transformer with cross-attention (encoder → decoder)
  4. Feature Pyramid Network (FPN) with lateral connections
  5. DenseNet with dense skip connections

Reports InterfaceContract specifications, verification time,
compositional vs monolithic results.
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

RESNET50_BLOCK = """\
import torch
import torch.nn as nn

class ResNet50Block(nn.Module):
    \"\"\"ResNet-50 bottleneck block with residual skip connection.\"\"\"
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
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        out = out + identity
        out = self.relu(out)
        return out
"""

UNET_SKIP = """\
import torch
import torch.nn as nn

class UNetSkip(nn.Module):
    \"\"\"U-Net encoder-decoder with skip connections.\"\"\"
    def __init__(self):
        super().__init__()
        self.enc1 = nn.Conv2d(3, 64, 3, padding=1)
        self.bn_e1 = nn.BatchNorm2d(64)
        self.enc2 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn_e2 = nn.BatchNorm2d(128)
        self.enc3 = nn.Conv2d(128, 256, 3, padding=1)
        self.bn_e3 = nn.BatchNorm2d(256)
        self.dec1 = nn.Conv2d(256, 128, 3, padding=1)
        self.bn_d1 = nn.BatchNorm2d(128)
        self.dec2 = nn.Conv2d(128, 64, 3, padding=1)
        self.bn_d2 = nn.BatchNorm2d(64)
        self.dec3 = nn.Conv2d(64, 3, 3, padding=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        e1 = self.relu(self.bn_e1(self.enc1(x)))
        e2 = self.relu(self.bn_e2(self.enc2(e1)))
        e3 = self.relu(self.bn_e3(self.enc3(e2)))
        d1 = self.relu(self.bn_d1(self.dec1(e3)))
        d2 = self.relu(self.bn_d2(self.dec2(d1)))
        d3 = self.dec3(d2)
        return d3
"""

TRANSFORMER_CROSS_ATTN = """\
import torch
import torch.nn as nn

class TransformerCrossAttn(nn.Module):
    \"\"\"Transformer with cross-attention from encoder to decoder.\"\"\"
    def __init__(self):
        super().__init__()
        self.enc_proj = nn.Linear(512, 512)
        self.enc_norm = nn.LayerNorm(512)
        self.dec_proj = nn.Linear(512, 512)
        self.dec_norm = nn.LayerNorm(512)
        self.cross_q = nn.Linear(512, 512)
        self.cross_k = nn.Linear(512, 512)
        self.cross_v = nn.Linear(512, 512)
        self.fc_out = nn.Linear(512, 512)
        self.out_norm = nn.LayerNorm(512)

    def forward(self, x):
        enc = self.enc_norm(self.enc_proj(x))
        dec = self.dec_norm(self.dec_proj(x))
        q = self.cross_q(dec)
        k = self.cross_k(enc)
        v = self.cross_v(enc)
        out = self.out_norm(self.fc_out(q))
        return out
"""

FPN_LATERAL = """\
import torch
import torch.nn as nn

class FPNLateral(nn.Module):
    \"\"\"Feature Pyramid Network with lateral connections.\"\"\"
    def __init__(self):
        super().__init__()
        self.c1 = nn.Conv2d(3, 64, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.c2 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.c3 = nn.Conv2d(128, 256, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.lat2 = nn.Conv2d(128, 256, 1)
        self.lat1 = nn.Conv2d(64, 256, 1)
        self.smooth = nn.Conv2d(256, 256, 3, padding=1)
        self.bn_out = nn.BatchNorm2d(256)
        self.relu = nn.ReLU()

    def forward(self, x):
        c1 = self.relu(self.bn1(self.c1(x)))
        c2 = self.relu(self.bn2(self.c2(c1)))
        c3 = self.relu(self.bn3(self.c3(c2)))
        l2 = self.lat2(c2)
        l1 = self.lat1(c1)
        out = self.relu(self.bn_out(self.smooth(c3)))
        return out
"""

DENSENET_DENSE = """\
import torch
import torch.nn as nn

class DenseNetBlock(nn.Module):
    \"\"\"DenseNet block with dense skip connections.\"\"\"
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(64, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(96, 32, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(128, 32, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU()

    def forward(self, x):
        out1 = self.relu(self.bn1(self.conv1(x)))
        cat1 = torch.cat([x, out1], dim=1)
        out2 = self.relu(self.bn2(self.conv2(cat1)))
        cat2 = torch.cat([x, out1, out2], dim=1)
        out3 = self.relu(self.bn3(self.conv3(cat2)))
        return out3
"""

ARCHITECTURES = [
    {
        "name": "ResNet-50 Bottleneck Block",
        "arch_type": "residual",
        "source": RESNET50_BLOCK,
        "input_shapes": {"x": ("batch", 256, 16, 16)},
        "description": "3-layer bottleneck with residual skip connection",
    },
    {
        "name": "U-Net Encoder-Decoder",
        "arch_type": "encoder_decoder",
        "source": UNET_SKIP,
        "input_shapes": {"x": ("batch", 3, 128, 128)},
        "description": "Encoder-decoder with skip connections",
    },
    {
        "name": "Transformer Cross-Attention",
        "arch_type": "cross_attention",
        "source": TRANSFORMER_CROSS_ATTN,
        "input_shapes": {"x": ("batch", 16, 512)},
        "description": "Encoder output feeds key/value into decoder cross-attention",
    },
    {
        "name": "FPN Lateral Connections",
        "arch_type": "feature_pyramid",
        "source": FPN_LATERAL,
        "input_shapes": {"x": ("batch", 3, 64, 64)},
        "description": "Bottom-up backbone with lateral 1x1 conv connections",
    },
    {
        "name": "DenseNet Dense Block",
        "arch_type": "dense",
        "source": DENSENET_DENSE,
        "input_shapes": {"x": ("batch", 64, 32, 32)},
        "description": "Dense skip connections via channel concatenation",
    },
]


# ─── Evaluation ──────────────────────────────────────────────────────────────

def evaluate_architecture(arch):
    """Evaluate one architecture with monolithic, sequential, and DAG verification."""
    name = arch["name"]
    source = arch["source"]
    input_shapes = arch["input_shapes"]
    result = {
        "name": name,
        "arch_type": arch["arch_type"],
        "description": arch["description"],
    }

    # 1. Monolithic verification
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

    # 2. Sequential compositional verification (baseline)
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
            "strategy": seq.decomposition_strategy.value,
        }
    except Exception as e:
        result["sequential_compositional"] = {"error": str(e)}

    # 3. BRANCH_MERGE DAG compositional verification
    reset_default_cache()
    try:
        t0 = time.monotonic()
        dag = verify_compositional_dag(
            source, input_shapes=input_shapes, measure_monolithic=False,
        )
        dag_time = (time.monotonic() - t0) * 1000

        # Extract decomposition details
        graph = extract_computation_graph(source)
        subs, edges, topology = decompose_graph_dag(graph, input_shapes)
        dag_rule = DAGCompositionProofRule.from_submodules_and_edges(
            subs, edges, topology,
        )

        # Collect interface contracts
        contracts = []
        for sm in subs:
            contracts.append({
                "module": sm.name,
                "input_contract": sm.input_contract.pretty(),
                "output_contract": sm.output_contract.pretty(),
                "step_range": list(sm.step_range),
            })

        # Interface checks
        iface_checks = validate_interface_dag(subs, edges)
        iface_details = []
        for ic in iface_checks:
            iface_details.append({
                "producer": ic.producer,
                "consumer": ic.consumer,
                "compatible": ic.compatible,
                "message": ic.message,
            })

        result["dag_compositional"] = {
            "safe": dag.safe,
            "time_ms": round(dag_time, 2),
            "strategy": dag.decomposition_strategy.value,
            "num_submodules": dag.num_submodules,
            "num_edges": len(edges),
            "topology": topology,
            "topological_order": dag_rule.topological_order(),
            "edges": edges,
            "interface_contracts": contracts,
            "interface_checks": iface_details,
            "all_interfaces_ok": all(ic.compatible for ic in iface_checks),
            "proof_rule": dag_rule.pretty(),
        }
    except Exception as e:
        result["dag_compositional"] = {
            "error": str(e),
            "traceback": traceback.format_exc(),
        }

    # 4. Agreement check
    mono_safe = result.get("monolithic", {}).get("safe")
    seq_safe = result.get("sequential_compositional", {}).get("safe")
    dag_safe = result.get("dag_compositional", {}).get("safe")

    result["agreement"] = {
        "mono_vs_seq": mono_safe == seq_safe if mono_safe is not None and seq_safe is not None else None,
        "mono_vs_dag": mono_safe == dag_safe if mono_safe is not None and dag_safe is not None else None,
        "seq_vs_dag": seq_safe == dag_safe if seq_safe is not None and dag_safe is not None else None,
    }

    # Timing comparison
    mono_ms = result.get("monolithic", {}).get("time_ms")
    dag_ms = result.get("dag_compositional", {}).get("time_ms")
    if mono_ms and dag_ms and dag_ms > 0:
        result["speedup_monolithic_over_dag"] = round(mono_ms / dag_ms, 3)

    return result


def main():
    print("=" * 72)
    print("Non-Sequential Assume-Guarantee Benchmark")
    print("Strategy: BRANCH_MERGE decomposition")
    print("=" * 72)
    print()

    all_results = []
    summary = {
        "total_architectures": len(ARCHITECTURES),
        "dag_successes": 0,
        "dag_failures": 0,
        "agreement_mono_dag": 0,
        "disagreement_mono_dag": 0,
        "topologies_detected": [],
    }

    for arch in ARCHITECTURES:
        print(f"▶ {arch['name']} ({arch['arch_type']})")
        print(f"  {arch['description']}")
        try:
            result = evaluate_architecture(arch)
            all_results.append(result)

            dag_info = result.get("dag_compositional", {})
            if "error" not in dag_info:
                summary["dag_successes"] += 1
                topo = dag_info.get("topology", "unknown")
                summary["topologies_detected"].append(topo)
                print(f"  Monolithic: safe={result.get('monolithic', {}).get('safe')}, "
                      f"time={result.get('monolithic', {}).get('time_ms', '?')}ms")
                print(f"  Sequential: safe={result.get('sequential_compositional', {}).get('safe')}, "
                      f"time={result.get('sequential_compositional', {}).get('time_ms', '?')}ms")
                print(f"  DAG:        safe={dag_info.get('safe')}, "
                      f"topology={topo}, "
                      f"nodes={dag_info.get('num_submodules')}, "
                      f"edges={dag_info.get('num_edges')}, "
                      f"time={dag_info.get('time_ms')}ms")
                print(f"  Interfaces: all_ok={dag_info.get('all_interfaces_ok')}")
            else:
                summary["dag_failures"] += 1
                print(f"  DAG ERROR: {dag_info['error'][:100]}")

            agr = result.get("agreement", {})
            if agr.get("mono_vs_dag") is True:
                summary["agreement_mono_dag"] += 1
            elif agr.get("mono_vs_dag") is False:
                summary["disagreement_mono_dag"] += 1

        except Exception as e:
            print(f"  FAILED: {e}")
            all_results.append({"name": arch["name"], "error": str(e)})
            summary["dag_failures"] += 1

        print()

    # Summary
    print("=" * 72)
    print("Summary")
    print("=" * 72)
    print(f"  Architectures tested:     {summary['total_architectures']}")
    print(f"  DAG successes:            {summary['dag_successes']}")
    print(f"  DAG failures:             {summary['dag_failures']}")
    print(f"  Mono-DAG agreements:      {summary['agreement_mono_dag']}")
    print(f"  Mono-DAG disagreements:   {summary['disagreement_mono_dag']}")
    print(f"  Topologies detected:      {summary['topologies_detected']}")

    # Save
    output = {
        "experiment": "nonseq_ag_benchmark",
        "strategy": "BRANCH_MERGE",
        "summary": summary,
        "results": all_results,
    }

    benchmarks_dir = os.path.join(os.path.dirname(__file__), "..", ".benchmarks")
    os.makedirs(benchmarks_dir, exist_ok=True)
    output_path = os.path.join(benchmarks_dir, "nonseq_ag_results.json")
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
