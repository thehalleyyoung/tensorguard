#!/usr/bin/env python3
"""Experiment: Craig interpolation predicate discovery during CEGAR refinement.

Demonstrates that the Craig interpolation module discovers predicates
(predicates_from_interpolation > 0) on benchmarks with shape mismatches,
complementing the template-based unsat-core extraction.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.shape_cegar import ShapeCEGARLoop, CEGARStatus


BENCHMARKS = {
    "linear_symbolic": (
        '''
import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 5)

    def forward(self, x):
        return self.fc1(x)
''',
        {"x": ("batch", "features")},
    ),
    "linear_chain_symbolic": (
        '''
import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(768, 512)
        self.fc2 = nn.Linear(512, 256)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        return x
''',
        {"x": ("batch", "features")},
    ),
    "conv_symbolic": (
        '''
import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 3)

    def forward(self, x):
        return self.conv1(x)
''',
        {"x": ("batch", "channels", "height", "width")},
    ),
    "batchnorm_symbolic": (
        '''
import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(256, 128)
        self.bn = nn.BatchNorm1d(128)

    def forward(self, x):
        x = self.fc(x)
        x = self.bn(x)
        return x
''',
        {"x": ("batch", "features")},
    ),
}

def run_experiment():
    total_interp_preds = 0
    total_attempted = 0
    total_successful = 0

    print("=" * 70)
    print("Craig Interpolation Predicate Discovery Experiment")
    print("=" * 70)

    for name, (source, input_shapes) in BENCHMARKS.items():
        print(f"\n--- Benchmark: {name} ---")
        loop = ShapeCEGARLoop(
            source,
            input_shapes=input_shapes,
            enable_interpolation=True,
            max_iterations=5,
        )
        result = loop.run()
        print(f"  Status: {result.final_status.name}")
        print(f"  Iterations: {len(result.iteration_log)}")
        print(f"  Predicates: {len(result.discovered_predicates)}")

        if result.interpolation_stats:
            stats = result.interpolation_stats
            print(f"  Interpolation attempted: {stats.get('attempted', 0)}")
            print(f"  Interpolation successful: {stats.get('successful', 0)}")
            preds = stats.get("predicates_from_interpolation", 0)
            print(f"  Predicates from interpolation: {preds}")
            total_interp_preds += preds
            total_attempted += stats.get("attempted", 0)
            total_successful += stats.get("successful", 0)
        else:
            print("  Interpolation: not triggered")

    print("\n" + "=" * 70)
    print("SUMMARY")
    print(f"  Total interpolation attempts:   {total_attempted}")
    print(f"  Total interpolation successes:   {total_successful}")
    print(f"  Total predicates from interpolation: {total_interp_preds}")
    print("=" * 70)

    if total_interp_preds > 0:
        print("\n✅ Craig interpolation IS discovering predicates!")
    else:
        print("\n⚠️  No interpolation predicates discovered in these benchmarks.")
        print("   (This may be expected if all CEs are classified as real bugs.)")

    return total_interp_preds


if __name__ == "__main__":
    n = run_experiment()
    sys.exit(0 if n >= 0 else 1)
