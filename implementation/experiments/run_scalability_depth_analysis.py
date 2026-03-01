"""
Scalability analysis: measure verification time as network depth and branching factor increase.
Generates data for paper scalability curves.
"""

import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.model_checker import verify_model


def generate_linear_chain(depth: int) -> str:
    """Generate a safe linear chain of depth `depth`."""
    layers = []
    fwd = []
    for i in range(depth):
        layers.append(f"        self.fc{i} = nn.Linear(256, 256)")
        fwd.append(f"        x = self.fc{i}(x)")
    layers_str = "\n".join(layers)
    fwd_str = "\n".join(fwd)
    return f'''
import torch.nn as nn
class LinearChain(nn.Module):
    def __init__(self):
        super().__init__()
{layers_str}
    def forward(self, x):
{fwd_str}
        return x
'''


def generate_branching_model(branches: int, depth_per_branch: int) -> str:
    """Generate a model with `branches` parallel branches, each of `depth_per_branch` layers."""
    layers = []
    fwd_branches = []
    for b in range(branches):
        for d in range(depth_per_branch):
            layers.append(f"        self.b{b}_fc{d} = nn.Linear(256, 256)")
        branch_fwd = [f"self.b{b}_fc{d}(" for d in range(depth_per_branch)]
        closes = ")" * depth_per_branch
        expr = "".join(branch_fwd) + f"x{closes}"
        fwd_branches.append(f"        out{b} = {expr}")
    
    layers_str = "\n".join(layers)
    fwd_str = "\n".join(fwd_branches)
    sum_expr = " + ".join([f"out{b}" for b in range(branches)])
    
    return f'''
import torch
import torch.nn as nn
class BranchingModel(nn.Module):
    def __init__(self):
        super().__init__()
{layers_str}
    def forward(self, x):
{fwd_str}
        return {sum_expr}
'''


def run_scalability_analysis():
    results = {
        "depth_scaling": [],
        "branching_scaling": [],
    }

    # Depth scaling: 2, 5, 10, 20, 50, 100 layers
    print("=== Depth Scaling Analysis ===")
    for depth in [2, 5, 10, 20, 50, 100]:
        source = generate_linear_chain(depth)
        times = []
        for trial in range(5):
            start = time.perf_counter()
            result = verify_model(source, input_shapes={"x": ("batch", 256)})
            elapsed_ms = (time.perf_counter() - start) * 1000
            times.append(elapsed_ms)
        
        median_ms = sorted(times)[len(times) // 2]
        mean_ms = sum(times) / len(times)
        cv = (max(times) - min(times)) / mean_ms * 100 if mean_ms > 0 else 0
        per_layer_ms = median_ms / depth
        
        entry = {
            "depth": depth,
            "median_ms": round(median_ms, 2),
            "mean_ms": round(mean_ms, 2),
            "min_ms": round(min(times), 2),
            "max_ms": round(max(times), 2),
            "cv_pct": round(cv, 1),
            "per_layer_ms": round(per_layer_ms, 2),
            "safe": result.safe,
        }
        results["depth_scaling"].append(entry)
        print(f"  depth={depth:3d}: median={median_ms:.1f}ms, "
              f"per_layer={per_layer_ms:.2f}ms, safe={result.safe}")

    # Branching scaling: 2, 4, 8, 16 branches × 5 layers each
    print("\n=== Branching Scaling Analysis ===")
    for branches in [2, 4, 8, 16]:
        source = generate_branching_model(branches, depth_per_branch=5)
        times = []
        for trial in range(5):
            start = time.perf_counter()
            try:
                result = verify_model(source, input_shapes={"x": ("batch", 256)})
                safe = result.safe
            except Exception as e:
                safe = "error"
            elapsed_ms = (time.perf_counter() - start) * 1000
            times.append(elapsed_ms)
        
        median_ms = sorted(times)[len(times) // 2]
        total_layers = branches * 5
        per_layer_ms = median_ms / total_layers
        
        entry = {
            "branches": branches,
            "depth_per_branch": 5,
            "total_layers": total_layers,
            "median_ms": round(median_ms, 2),
            "mean_ms": round(sum(times) / len(times), 2),
            "per_layer_ms": round(per_layer_ms, 2),
            "safe": safe,
        }
        results["branching_scaling"].append(entry)
        print(f"  branches={branches:2d} (total={total_layers:3d} layers): "
              f"median={median_ms:.1f}ms, per_layer={per_layer_ms:.2f}ms")

    # Save results
    out_path = os.path.join(os.path.dirname(__file__), "scalability_depth_analysis_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")
    return results


if __name__ == "__main__":
    run_scalability_analysis()
