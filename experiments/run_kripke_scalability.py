"""Kripke structure scalability experiment.

Measures state count, transition count, verification time, and memory
for models of increasing depth and branching complexity.
"""

import json
import os
import sys
import time
import tracemalloc

# Ensure the project root is on sys.path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.model_checker import (
    extract_computation_graph,
    extract_kripke_structure,
    Device,
    Phase,
)


# ═══════════════════════════════════════════════════════════════════════════════
# Model templates
# ═══════════════════════════════════════════════════════════════════════════════

DEEP_MODEL_TEMPLATE = """\
import torch.nn as nn

class DeepModel(nn.Module):
    def __init__(self):
        super().__init__()
{layers}

    def forward(self, x):
{forward}
"""

RESNET_BLOCK_TEMPLATE = """\
import torch
import torch.nn as nn

class ResNetLike(nn.Module):
    def __init__(self):
        super().__init__()
{layers}

    def forward(self, x):
{forward}
"""


def make_deep_model(depth: int) -> str:
    layers = "\n".join(
        f"        self.fc{i} = nn.Linear(64, 64)" for i in range(depth)
    )
    forward_lines = []
    for i in range(depth):
        forward_lines.append(f"        x = self.fc{i}(x)")
    forward_lines.append("        return x")
    return DEEP_MODEL_TEMPLATE.format(
        layers=layers, forward="\n".join(forward_lines)
    )


def make_resnet_model(num_blocks: int) -> str:
    """Create a model with skip connections (ResNet-style)."""
    layers = []
    for i in range(num_blocks):
        layers.append(f"        self.fc{i}a = nn.Linear(64, 64)")
        layers.append(f"        self.fc{i}b = nn.Linear(64, 64)")

    forward_lines = []
    for i in range(num_blocks):
        forward_lines.append(f"        residual = x")
        forward_lines.append(f"        x = self.fc{i}a(x)")
        forward_lines.append(f"        x = self.fc{i}b(x)")
        # Skip connection is implicit — x = x + residual not directly
        # supported by the AST extractor, but the layers are still extracted.
    forward_lines.append("        return x")
    return RESNET_BLOCK_TEMPLATE.format(
        layers="\n".join(layers), forward="\n".join(forward_lines)
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Experiment runner
# ═══════════════════════════════════════════════════════════════════════════════

def run_experiment(source: str, input_shapes: dict, label: str) -> dict:
    """Run a single scalability experiment and collect metrics."""
    tracemalloc.start()
    t0 = time.monotonic()

    graph = extract_computation_graph(source)
    ks = extract_kripke_structure(
        graph,
        input_shapes=input_shapes,
        initial_device=Device.CPU,
        initial_phase=Phase.EVAL,
    )

    elapsed_ms = (time.monotonic() - t0) * 1000
    _, peak_kb = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    return {
        "label": label,
        "num_layers": len(graph.layers),
        "num_steps": graph.num_steps,
        "kripke_num_states": ks.num_states,
        "kripke_num_transitions": ks.num_transitions,
        "is_safe": ks.is_safe(),
        "verification_time_ms": round(elapsed_ms, 2),
        "peak_memory_kb": round(peak_kb / 1024, 2),
    }


def main():
    results = {"depth_scaling": [], "branching_scaling": []}

    # Depth scaling: 5, 10, 20, 50, 100 layers
    print("=== Depth Scaling ===")
    for depth in [5, 10, 20, 50, 100]:
        source = make_deep_model(depth)
        r = run_experiment(
            source,
            input_shapes={"x": ("batch", 64)},
            label=f"deep_{depth}",
        )
        results["depth_scaling"].append(r)
        print(
            f"  depth={depth:3d}  states={r['kripke_num_states']:4d}  "
            f"transitions={r['kripke_num_transitions']:4d}  "
            f"time={r['verification_time_ms']:8.2f}ms  "
            f"mem={r['peak_memory_kb']:8.2f}KB"
        )

    # Branching scaling: ResNet-style skip connections
    print("\n=== Branching Scaling (ResNet-style) ===")
    for num_blocks in [2, 5, 10, 20, 50]:
        source = make_resnet_model(num_blocks)
        r = run_experiment(
            source,
            input_shapes={"x": ("batch", 64)},
            label=f"resnet_{num_blocks}_blocks",
        )
        results["branching_scaling"].append(r)
        print(
            f"  blocks={num_blocks:3d}  states={r['kripke_num_states']:4d}  "
            f"transitions={r['kripke_num_transitions']:4d}  "
            f"time={r['verification_time_ms']:8.2f}ms  "
            f"mem={r['peak_memory_kb']:8.2f}KB"
        )

    # Save results
    out_dir = os.path.join(
        os.path.dirname(__file__), "..", ".benchmarks"
    )
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "kripke_scalability_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
