"""
Explanation evaluation: demonstrates CEGAR explanation generation for
5 diverse models and saves structured results.

Usage:
    cd implementation && python -m experiments.run_explanation_eval
"""

from __future__ import annotations

import json
import os
import sys
import time

# Ensure the implementation directory is on the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.cegar_explanation import explain_verification


# ═══════════════════════════════════════════════════════════════════════════════
# Diverse model sources
# ═══════════════════════════════════════════════════════════════════════════════

MODELS = [
    {
        "name": "SimpleLinear",
        "description": "Single linear layer, concrete input shapes",
        "source": """\
import torch.nn as nn

class SimpleLinear(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 5)

    def forward(self, x):
        return self.fc(x)
""",
        "input_shapes": {"x": ("batch", 10)},
    },
    {
        "name": "TwoLayerMLP",
        "description": "Two-layer MLP, tests inter-layer shape propagation",
        "source": """\
import torch.nn as nn

class TwoLayerMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        return x
""",
        "input_shapes": {"x": ("batch", 784)},
    },
    {
        "name": "ShapeMismatch",
        "description": "Intentional shape mismatch between layers (UNSAFE)",
        "source": """\
import torch.nn as nn

class ShapeMismatch(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 20)
        self.fc2 = nn.Linear(50, 5)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        return x
""",
        "input_shapes": {"x": ("batch", 10)},
    },
    {
        "name": "DeepNetwork",
        "description": "Four-layer deep network, tests chain verification",
        "source": """\
import torch.nn as nn

class DeepNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(512, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 10)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)
        return x
""",
        "input_shapes": {"x": ("batch", 512)},
    },
    {
        "name": "SymbolicInput",
        "description": "Symbolic input dimension, tests predicate discovery",
        "source": """\
import torch.nn as nn

class SymbolicInput(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(768, 256)

    def forward(self, x):
        return self.fc(x)
""",
        "input_shapes": {"x": ("batch", "features")},
    },
]


def main():
    results = []

    for model_info in MODELS:
        print(f"\n{'='*60}")
        print(f"Model: {model_info['name']}")
        print(f"Description: {model_info['description']}")
        print(f"{'='*60}")

        t0 = time.monotonic()
        explanation = explain_verification(
            model_info["source"],
            input_shapes=model_info["input_shapes"],
            model_name=model_info["name"],
        )
        elapsed_ms = (time.monotonic() - t0) * 1000

        rendered = explanation.render()
        print(rendered)
        print(f"\n[Elapsed: {elapsed_ms:.1f}ms]")

        result_entry = explanation.to_dict()
        result_entry["description"] = model_info["description"]
        result_entry["elapsed_ms"] = round(elapsed_ms, 1)
        result_entry["rendered_explanation"] = rendered
        results.append(result_entry)

    # Save results
    out_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        ".benchmarks",
    )
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "explanation_eval_results.json")

    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Results saved to {out_path}")
    print(f"Models evaluated: {len(results)}")
    verdicts = [r["verdict"] for r in results]
    print(f"Verdicts: {', '.join(verdicts)}")


if __name__ == "__main__":
    main()
