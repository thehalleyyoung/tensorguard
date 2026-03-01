#!/usr/bin/env python3
"""Dynamo vs FX coverage experiment.

Compares ``torch.fx.symbolic_trace`` and ``dynamo_trace_to_graph`` on a
suite of models including data-dependent control flow, and saves results
to ``dynamo_coverage_results.json``.
"""

import json
import os
import sys
import time
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

HAS_DYNAMO = False
if HAS_TORCH:
    try:
        import torch._dynamo
        torch._dynamo.eval_frame.check_if_dynamo_supported()
        HAS_DYNAMO = True
    except (ImportError, RuntimeError):
        pass


# ═══════════════════════════════════════════════════════════════════════════════
# Test models
# ═══════════════════════════════════════════════════════════════════════════════

if HAS_TORCH:
    class SimpleMLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(64, 32)
            self.relu = nn.ReLU()
            self.fc2 = nn.Linear(32, 10)

        def forward(self, x):
            return self.fc2(self.relu(self.fc1(x)))

    class ConvNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(3, 16, 3, padding=1)
            self.bn = nn.BatchNorm2d(16)
            self.pool = nn.AdaptiveAvgPool2d(1)
            self.fc = nn.Linear(16, 10)

        def forward(self, x):
            x = self.pool(self.bn(self.conv(x)))
            return self.fc(x.flatten(1))

    class ShapeConditionalModel(nn.Module):
        """Branches on input shape."""
        def __init__(self):
            super().__init__()
            self.layer_a = nn.Linear(64, 32)
            self.layer_b = nn.Linear(64, 32)
            self.out = nn.Linear(32, 10)

        def forward(self, x):
            if x.shape[0] > 1:
                x = self.layer_a(x)
            else:
                x = self.layer_b(x)
            return self.out(x)

    class ValueConditionalModel(nn.Module):
        """Branches on tensor value."""
        def __init__(self):
            super().__init__()
            self.layer_a = nn.Linear(64, 32)
            self.layer_b = nn.Linear(64, 32)
            self.out = nn.Linear(32, 10)

        def forward(self, x):
            if x.mean() > 0:
                x = self.layer_a(x)
            else:
                x = self.layer_b(x)
            return self.out(x)

    class MoEGating(nn.Module):
        """Simplified mixture-of-experts."""
        def __init__(self):
            super().__init__()
            self.gate = nn.Linear(64, 2)
            self.expert_a = nn.Linear(64, 32)
            self.expert_b = nn.Linear(64, 32)
            self.out = nn.Linear(32, 10)

        def forward(self, x):
            g = self.gate(x)
            if g[:, 0].mean() > g[:, 1].mean():
                h = self.expert_a(x)
            else:
                h = self.expert_b(x)
            return self.out(h)

    class DynamicLoopModel(nn.Module):
        """Loop whose iteration count depends on input."""
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(64, 64)
            self.out = nn.Linear(64, 10)

        def forward(self, x):
            n = min(x.shape[0], 3)
            for _ in range(n):
                x = self.linear(x)
            return self.out(x)

    class ResidualBlock(nn.Module):
        """Simple residual block (traceable)."""
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(64, 64)
            self.fc2 = nn.Linear(64, 64)
            self.relu = nn.ReLU()

        def forward(self, x):
            return x + self.relu(self.fc2(self.relu(self.fc1(x))))


# ═══════════════════════════════════════════════════════════════════════════════
# Experiment runner
# ═══════════════════════════════════════════════════════════════════════════════

def _try_fx(model, example_inputs):
    """Try torch.fx.symbolic_trace and return (success, graph_or_error)."""
    from src.fx_extractor import fx_trace_to_graph
    try:
        model.eval()
        traced = torch.fx.symbolic_trace(model)
        graph = fx_trace_to_graph(traced, class_name=type(model).__name__)
        return True, graph
    except Exception as exc:
        return False, str(exc)


def _try_dynamo(model, example_inputs):
    """Try dynamo_trace_to_graph and return (success, graph_or_error)."""
    from src.dynamo_extractor import dynamo_trace_to_graph
    try:
        graph = dynamo_trace_to_graph(
            model,
            example_inputs=example_inputs,
            class_name=type(model).__name__,
        )
        return True, graph
    except Exception as exc:
        return False, str(exc)


def run_experiment():
    if not HAS_TORCH:
        print("PyTorch not available — skipping experiment")
        return

    models = [
        ("SimpleMLP", SimpleMLP(), (torch.randn(2, 64),), False),
        ("ConvNet", ConvNet(), (torch.randn(2, 3, 8, 8),), False),
        ("ResidualBlock", ResidualBlock(), (torch.randn(2, 64),), False),
        ("ShapeConditional", ShapeConditionalModel(), (torch.randn(2, 64),), True),
        ("ValueConditional", ValueConditionalModel(), (torch.randn(2, 64),), True),
        ("MoEGating", MoEGating(), (torch.randn(2, 64),), True),
        ("DynamicLoop", DynamicLoopModel(), (torch.randn(2, 64),), True),
    ]

    results = {
        "torch_version": torch.__version__,
        "dynamo_available": HAS_DYNAMO,
        "models": [],
    }

    fx_success = 0
    fx_total = 0
    dynamo_success = 0
    dynamo_total = 0
    fx_control_flow_success = 0
    dynamo_control_flow_success = 0
    control_flow_total = 0

    for name, model, example_inputs, has_control_flow in models:
        entry: Dict[str, Any] = {
            "name": name,
            "has_data_dependent_control_flow": has_control_flow,
        }
        fx_total += 1
        dynamo_total += 1
        if has_control_flow:
            control_flow_total += 1

        # FX
        t0 = time.monotonic()
        ok, res = _try_fx(model, example_inputs)
        fx_time = (time.monotonic() - t0) * 1000
        entry["fx_success"] = ok
        entry["fx_time_ms"] = round(fx_time, 2)
        if ok:
            entry["fx_steps"] = res.num_steps
            entry["fx_layers"] = len(res.layers)
            fx_success += 1
            if has_control_flow:
                fx_control_flow_success += 1
        else:
            entry["fx_error"] = res

        # Dynamo
        if HAS_DYNAMO:
            torch._dynamo.reset()
            t0 = time.monotonic()
            ok, res = _try_dynamo(model, example_inputs)
            dynamo_time = (time.monotonic() - t0) * 1000
            entry["dynamo_success"] = ok
            entry["dynamo_time_ms"] = round(dynamo_time, 2)
            if ok:
                entry["dynamo_steps"] = res.num_steps
                entry["dynamo_layers"] = len(res.layers)
                entry["dynamo_subgraphs"] = res.dynamic_features.get(
                    "num_dynamo_subgraphs", 1
                )
                entry["dynamo_graph_breaks"] = res.dynamic_features.get(
                    "graph_breaks", 0
                )
                dynamo_success += 1
                if has_control_flow:
                    dynamo_control_flow_success += 1
            else:
                entry["dynamo_error"] = res
        else:
            entry["dynamo_success"] = False
            entry["dynamo_error"] = "TorchDynamo not available"

        results["models"].append(entry)
        status_fx = "✓" if entry["fx_success"] else "✗"
        status_dy = "✓" if entry.get("dynamo_success") else "✗"
        tag = " [control-flow]" if has_control_flow else ""
        print(f"  {name:25s} FX={status_fx}  Dynamo={status_dy}{tag}")

    # Summary
    results["summary"] = {
        "fx_success_rate": f"{fx_success}/{fx_total}",
        "dynamo_success_rate": f"{dynamo_success}/{dynamo_total}",
        "fx_control_flow_success": f"{fx_control_flow_success}/{control_flow_total}",
        "dynamo_control_flow_success": f"{dynamo_control_flow_success}/{control_flow_total}",
        "coverage_improvement_on_control_flow": (
            f"{fx_control_flow_success}/{control_flow_total} → "
            f"{dynamo_control_flow_success}/{control_flow_total}"
        ),
    }

    print("\n── Summary ──")
    print(f"  FX total:                {fx_success}/{fx_total}")
    print(f"  Dynamo total:            {dynamo_success}/{dynamo_total}")
    print(f"  FX on control-flow:      {fx_control_flow_success}/{control_flow_total}")
    print(f"  Dynamo on control-flow:  {dynamo_control_flow_success}/{control_flow_total}")

    out_path = os.path.join(os.path.dirname(__file__), "dynamo_coverage_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    run_experiment()
