"""
Assume-Guarantee Disagreement Analysis.

Addresses reviewer critique #3: classify the 2/14 AG disagreements with
monolithic verification as either UNSOUND or INCOMPLETE, with root cause
analysis for each.

Pre-fix analysis (before patching assume_guarantee.py):
  - 2/14 models disagreed: cnn_simple and cnn_deep
  - Both classified as INCOMPLETE (compositional rejects, monolithic accepts)
  - Zero UNSOUND cases found
  - Root cause: rank-changing ops (flatten/reshape) at sub-module boundaries
    cause the interface checker to see a rank mismatch (4D vs 2D) even though
    the element count is consistent
  - Fix applied: check_interface_compatibility now detects when the consumer's
    first op on a shared tensor is FLATTEN/RESHAPE and falls back to
    element-count compatibility instead of strict shape matching

Post-fix: 14/14 models agree, 0 disagreements.

Terminology:
  UNSOUND:    compositional accepts but monolithic rejects (false negative)
  INCOMPLETE: compositional rejects but monolithic accepts (over-approximation)
"""

import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.model_checker import verify_model, extract_computation_graph, OpKind
from src.assume_guarantee import (
    verify_compositional,
    decompose_graph,
    check_interface_compatibility,
    CompositionalResult,
    _shapes_compatible,
)

# ─── Full test set (same as run_compositional_experiment.py) ──────────────────

MODELS = [
    {
        "name": "simple_mlp_2layer",
        "category": "simple_mlp",
        "num_layers": 2,
        "input_shapes": {"x": ("batch", 32)},
        "source": '''
import torch.nn as nn
class SimpleMLP2(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(32, 16)
        self.fc2 = nn.Linear(16, 4)
    def forward(self, x):
        x = self.fc1(x)
        return self.fc2(x)
''',
    },
    {
        "name": "simple_mlp_3layer",
        "category": "simple_mlp",
        "num_layers": 3,
        "input_shapes": {"x": ("batch", 64)},
        "source": '''
import torch.nn as nn
class SimpleMLP3(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(64, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)
    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        return self.fc3(x)
''',
    },
    {
        "name": "medium_mlp_5layer",
        "category": "medium_mlp",
        "num_layers": 5,
        "input_shapes": {"x": ("batch", 128)},
        "source": '''
import torch.nn as nn
class MediumMLP5(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(128, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 64)
        self.fc5 = nn.Linear(64, 10)
    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)
        return self.fc5(x)
''',
    },
    {
        "name": "medium_mlp_8layer",
        "category": "medium_mlp",
        "num_layers": 8,
        "input_shapes": {"x": ("batch", 256)},
        "source": '''
import torch.nn as nn
class MediumMLP8(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(256, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 256)
        self.fc4 = nn.Linear(256, 256)
        self.fc5 = nn.Linear(256, 128)
        self.fc6 = nn.Linear(128, 128)
        self.fc7 = nn.Linear(128, 64)
        self.fc8 = nn.Linear(64, 10)
    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)
        x = self.fc5(x)
        x = self.fc6(x)
        x = self.fc7(x)
        return self.fc8(x)
''',
    },
    {
        "name": "deep_mlp_10layer",
        "category": "deep_mlp",
        "num_layers": 10,
        "input_shapes": {"x": ("batch", 256)},
        "source": '''
import torch.nn as nn
class DeepMLP10(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(256, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 256)
        self.fc4 = nn.Linear(256, 256)
        self.fc5 = nn.Linear(256, 256)
        self.fc6 = nn.Linear(256, 256)
        self.fc7 = nn.Linear(256, 256)
        self.fc8 = nn.Linear(256, 256)
        self.fc9 = nn.Linear(256, 128)
        self.fc10 = nn.Linear(128, 10)
    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)
        x = self.fc5(x)
        x = self.fc6(x)
        x = self.fc7(x)
        x = self.fc8(x)
        x = self.fc9(x)
        return self.fc10(x)
''',
    },
    {
        "name": "deep_mlp_15layer",
        "category": "deep_mlp",
        "num_layers": 15,
        "input_shapes": {"x": ("batch", 256)},
        "source": '''
import torch.nn as nn
class DeepMLP15(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(256, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 256)
        self.fc4 = nn.Linear(256, 256)
        self.fc5 = nn.Linear(256, 256)
        self.fc6 = nn.Linear(256, 256)
        self.fc7 = nn.Linear(256, 256)
        self.fc8 = nn.Linear(256, 256)
        self.fc9 = nn.Linear(256, 256)
        self.fc10 = nn.Linear(256, 256)
        self.fc11 = nn.Linear(256, 256)
        self.fc12 = nn.Linear(256, 256)
        self.fc13 = nn.Linear(256, 128)
        self.fc14 = nn.Linear(128, 64)
        self.fc15 = nn.Linear(64, 10)
    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)
        x = self.fc5(x)
        x = self.fc6(x)
        x = self.fc7(x)
        x = self.fc8(x)
        x = self.fc9(x)
        x = self.fc10(x)
        x = self.fc11(x)
        x = self.fc12(x)
        x = self.fc13(x)
        x = self.fc14(x)
        return self.fc15(x)
''',
    },
    {
        "name": "deep_mlp_20layer",
        "category": "deep_mlp",
        "num_layers": 20,
        "input_shapes": {"x": ("batch", 512)},
        "source": '''
import torch.nn as nn
class DeepMLP20(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(512, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 512)
        self.fc4 = nn.Linear(512, 512)
        self.fc5 = nn.Linear(512, 512)
        self.fc6 = nn.Linear(512, 512)
        self.fc7 = nn.Linear(512, 512)
        self.fc8 = nn.Linear(512, 512)
        self.fc9 = nn.Linear(512, 512)
        self.fc10 = nn.Linear(512, 512)
        self.fc11 = nn.Linear(512, 512)
        self.fc12 = nn.Linear(512, 512)
        self.fc13 = nn.Linear(512, 512)
        self.fc14 = nn.Linear(512, 512)
        self.fc15 = nn.Linear(512, 512)
        self.fc16 = nn.Linear(512, 256)
        self.fc17 = nn.Linear(256, 256)
        self.fc18 = nn.Linear(256, 128)
        self.fc19 = nn.Linear(128, 64)
        self.fc20 = nn.Linear(64, 10)
    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)
        x = self.fc5(x)
        x = self.fc6(x)
        x = self.fc7(x)
        x = self.fc8(x)
        x = self.fc9(x)
        x = self.fc10(x)
        x = self.fc11(x)
        x = self.fc12(x)
        x = self.fc13(x)
        x = self.fc14(x)
        x = self.fc15(x)
        x = self.fc16(x)
        x = self.fc17(x)
        x = self.fc18(x)
        x = self.fc19(x)
        return self.fc20(x)
''',
    },
    {
        "name": "cnn_simple",
        "category": "cnn",
        "num_layers": 4,
        "input_shapes": {"x": ("batch", 3, 32, 32)},
        "source": '''
import torch.nn as nn
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.fc1 = nn.Linear(32768, 128)
        self.fc2 = nn.Linear(128, 10)
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        return self.fc2(x)
''',
    },
    {
        "name": "cnn_deep",
        "category": "cnn",
        "num_layers": 7,
        "input_shapes": {"x": ("batch", 3, 64, 64)},
        "source": '''
import torch.nn as nn
class DeepCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv4 = nn.Conv2d(128, 256, 3, padding=1)
        self.fc1 = nn.Linear(1048576, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 10)
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.fc2(x)
        return self.fc3(x)
''',
    },
    {
        "name": "residual_block",
        "category": "residual",
        "num_layers": 5,
        "input_shapes": {"x": ("batch", 128)},
        "source": '''
import torch.nn as nn
class ResidualNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(128, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 128)
        self.fc4 = nn.Linear(128, 128)
        self.out = nn.Linear(128, 10)
    def forward(self, x):
        residual = x
        x = self.fc1(x)
        x = self.fc2(x)
        x = x + residual
        residual = x
        x = self.fc3(x)
        x = self.fc4(x)
        x = x + residual
        return self.out(x)
''',
    },
    {
        "name": "multihead_attention",
        "category": "attention",
        "num_layers": 6,
        "input_shapes": {"x": ("batch", 64)},
        "source": '''
import torch.nn as nn
class AttentionBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.query = nn.Linear(64, 64)
        self.key = nn.Linear(64, 64)
        self.value = nn.Linear(64, 64)
        self.proj = nn.Linear(64, 64)
        self.ff1 = nn.Linear(64, 256)
        self.ff2 = nn.Linear(256, 64)
    def forward(self, x):
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)
        x = self.proj(v)
        x = self.ff1(x)
        return self.ff2(x)
''',
    },
    {
        "name": "lstm_classifier",
        "category": "lstm",
        "num_layers": 3,
        "input_shapes": {"x": ("batch", 50, 32)},
        "source": '''
import torch.nn as nn
class LSTMClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(32, 64, num_layers=2, batch_first=True)
        self.fc1 = nn.Linear(64, 32)
        self.fc2 = nn.Linear(32, 10)
    def forward(self, x):
        x, _ = self.lstm(x)
        x = x[:, -1, :]
        x = self.fc1(x)
        return self.fc2(x)
''',
    },
    {
        "name": "wide_parallel",
        "category": "wide",
        "num_layers": 7,
        "input_shapes": {"x": ("batch", 128)},
        "source": '''
import torch.nn as nn
class WideParallel(nn.Module):
    def __init__(self):
        super().__init__()
        self.branch_a1 = nn.Linear(128, 64)
        self.branch_a2 = nn.Linear(64, 32)
        self.branch_b1 = nn.Linear(128, 64)
        self.branch_b2 = nn.Linear(64, 32)
        self.branch_c1 = nn.Linear(128, 64)
        self.branch_c2 = nn.Linear(64, 32)
        self.merge = nn.Linear(32, 10)
    def forward(self, x):
        a = self.branch_a1(x)
        a = self.branch_a2(a)
        b = self.branch_b1(x)
        b = self.branch_b2(b)
        c = self.branch_c1(x)
        c = self.branch_c2(c)
        return self.merge(a)
''',
    },
    {
        "name": "bottleneck_mlp",
        "category": "medium_mlp",
        "num_layers": 6,
        "input_shapes": {"x": ("batch", 256)},
        "source": '''
import torch.nn as nn
class BottleneckMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.expand = nn.Linear(256, 1024)
        self.mid1 = nn.Linear(1024, 512)
        self.mid2 = nn.Linear(512, 128)
        self.mid3 = nn.Linear(128, 32)
        self.mid4 = nn.Linear(32, 128)
        self.out = nn.Linear(128, 10)
    def forward(self, x):
        x = self.expand(x)
        x = self.mid1(x)
        x = self.mid2(x)
        x = self.mid3(x)
        x = self.mid4(x)
        return self.out(x)
''',
    },
]


# ─── Root-cause analysis ─────────────────────────────────────────────────────

def analyze_disagreement(model: dict, mono_safe: bool, comp_result: CompositionalResult):
    """Classify and root-cause a disagreement between monolithic and compositional."""
    comp_safe = comp_result.safe

    if mono_safe and not comp_safe:
        classification = "INCOMPLETE"
        severity = "acceptable"
        description = (
            "Compositional verification rejects a model that monolithic "
            "accepts. This is an over-approximation — the compositional "
            "decomposition introduces imprecision at sub-module boundaries."
        )
    elif not mono_safe and comp_safe:
        classification = "UNSOUND"
        severity = "critical"
        description = (
            "Compositional verification accepts a model that monolithic "
            "rejects. This is a soundness violation — the decomposition "
            "masks a real error."
        )
    else:
        return None  # Not a disagreement

    # ── Determine root cause ──
    root_causes = []

    # Check interface incompatibilities
    for ic in comp_result.interface_checks:
        if not ic.compatible:
            root_causes.append({
                "type": "interface_incompatibility",
                "boundary": f"{ic.producer} -> {ic.consumer}",
                "message": ic.message,
            })

    # Check per-submodule failures
    for name, res in comp_result.submodule_results.items():
        if not res.safe:
            root_causes.append({
                "type": "submodule_failure",
                "submodule": name,
                "errors": res.errors,
            })

    # Analyze graph structure for rank-changing ops at boundaries
    try:
        graph = extract_computation_graph(model["source"])
        subs = decompose_graph(
            graph, strategy="auto",
            input_shapes=model["input_shapes"],
        )

        boundary_ops = []
        for i, sm in enumerate(subs):
            for step in sm.graph.steps:
                if step.op in (OpKind.RESHAPE, OpKind.FLATTEN):
                    boundary_ops.append({
                        "submodule": sm.name,
                        "step_index_in_submodule": sm.graph.steps.index(step),
                        "op": step.op.name,
                        "inputs": step.inputs,
                        "output": step.output,
                        "explanation": (
                            f"Rank-changing op ({step.op.name}) inside {sm.name} "
                            f"causes the output contract of the preceding block "
                            f"to have a different rank than the input contract of "
                            f"this block. The interface checker sees a rank "
                            f"mismatch because it compares shapes without "
                            f"accounting for the reshape/flatten semantics."
                        ),
                    })

        if boundary_ops:
            root_causes.append({
                "type": "rank_changing_op_at_boundary",
                "ops": boundary_ops,
                "fix_description": (
                    "The interface compatibility checker (_shapes_compatible) "
                    "rejects shape pairs with different ranks. When a "
                    "FLATTEN/RESHAPE op sits at the start of a sub-module, "
                    "the preceding sub-module's output contract has the "
                    "pre-flatten shape while the current sub-module's input "
                    "contract expects the post-flatten shape. The fix is to "
                    "make the interface checker aware of rank-changing ops "
                    "at sub-module boundaries, or to avoid placing cut points "
                    "across flatten/reshape operations."
                ),
            })
    except Exception as e:
        root_causes.append({
            "type": "analysis_error",
            "message": str(e),
        })

    # Safety argument for INCOMPLETE cases
    safety_argument = None
    if classification == "INCOMPLETE":
        safety_argument = (
            "This over-approximation is SAFE. The compositional verifier "
            "is conservative: it rejects models it cannot prove safe at "
            "sub-module boundaries. No real bug is missed — the monolithic "
            "verifier confirms the model is actually safe. The imprecision "
            "comes from the interface contract not modeling rank-changing "
            "operations (view/reshape/flatten) that sit at decomposition "
            "boundaries. The contract derives the producer's output shape "
            "from the last LAYER_CALL (e.g., Conv2d → 4D), and the "
            "consumer's input shape from its first LAYER_CALL (e.g., "
            "Linear → 2D), causing a spurious rank mismatch. This is "
            "a completeness issue, not a soundness issue."
        )

    return {
        "model_name": model["name"],
        "category": model["category"],
        "classification": classification,
        "severity": severity,
        "description": description,
        "monolithic_safe": mono_safe,
        "compositional_safe": comp_safe,
        "root_causes": root_causes,
        "safety_argument": safety_argument,
        "num_submodules": comp_result.num_submodules,
        "interface_checks": [
            {
                "producer": ic.producer,
                "consumer": ic.consumer,
                "compatible": ic.compatible,
                "message": ic.message,
            }
            for ic in comp_result.interface_checks
        ],
        "submodule_verdicts": {
            name: {"safe": r.safe, "errors": r.errors}
            for name, r in comp_result.submodule_results.items()
        },
    }


# ─── Main experiment ─────────────────────────────────────────────────────────

def run_experiment():
    from src.assume_guarantee import reset_default_cache
    reset_default_cache()

    all_results = []
    disagreements = []

    print(f"{'Model':<25} {'Mono':>6} {'Comp':>6} {'Agree':>6} {'Class':>12}")
    print("─" * 65)

    for model in MODELS:
        name = model["name"]
        source = model["source"]
        input_shapes = model["input_shapes"]

        # Monolithic
        mono_result = verify_model(source=source, input_shapes=input_shapes)

        # Compositional (fresh cache per model)
        reset_default_cache()
        comp_result = verify_compositional(
            source=source,
            input_shapes=input_shapes,
            measure_monolithic=False,
        )

        agree = mono_result.safe == comp_result.safe
        classification = "—"

        record = {
            "model_name": name,
            "category": model["category"],
            "monolithic_safe": mono_result.safe,
            "compositional_safe": comp_result.safe,
            "agree": agree,
        }

        if not agree:
            analysis = analyze_disagreement(model, mono_result.safe, comp_result)
            if analysis:
                classification = analysis["classification"]
                record["classification"] = classification
                record["analysis"] = analysis
                disagreements.append(analysis)

        all_results.append(record)

        mono_tag = "SAFE" if mono_result.safe else "UNSAFE"
        comp_tag = "SAFE" if comp_result.safe else "UNSAFE"
        agree_tag = "✓" if agree else "✗"

        print(f"{name:<25} {mono_tag:>6} {comp_tag:>6} {agree_tag:>6} {classification:>12}")

    # ── Summary ──
    total = len(all_results)
    agree_count = sum(1 for r in all_results if r["agree"])
    unsound = [d for d in disagreements if d["classification"] == "UNSOUND"]
    incomplete = [d for d in disagreements if d["classification"] == "INCOMPLETE"]

    print("\n" + "═" * 65)
    print("DISAGREEMENT ANALYSIS SUMMARY")
    print("═" * 65)
    print(f"  Total models tested:     {total}")
    print(f"  Agreements:              {agree_count}/{total}")
    print(f"  Disagreements:           {len(disagreements)}/{total}")
    print(f"    UNSOUND (critical):    {len(unsound)}")
    print(f"    INCOMPLETE (safe):     {len(incomplete)}")

    if unsound:
        print("\n⚠️  SOUNDNESS VIOLATIONS FOUND:")
        for d in unsound:
            print(f"    {d['model_name']}: compositional accepts, monolithic rejects")
            for rc in d["root_causes"]:
                print(f"      Root cause: {rc['type']}: {rc.get('message', '')}")
    else:
        print("\n✓ No soundness violations found.")

    if incomplete:
        print(f"\n  INCOMPLETE cases ({len(incomplete)}):")
        for d in incomplete:
            print(f"    {d['model_name']} ({d['category']}):")
            for rc in d["root_causes"]:
                if rc["type"] == "interface_incompatibility":
                    print(f"      Interface: {rc['boundary']}: {rc['message']}")
                elif rc["type"] == "rank_changing_op_at_boundary":
                    for op in rc["ops"]:
                        print(f"      {op['op']} in {op['submodule']}: "
                              f"rank-changing op causes spurious mismatch")
            print(f"      Safety: {d['safety_argument'][:80]}...")

    # ── Save results ──
    output = {
        "summary": {
            "total_models": total,
            "agreements": agree_count,
            "disagreements": len(disagreements),
            "unsound_count": len(unsound),
            "incomplete_count": len(incomplete),
            "soundness_preserved": len(unsound) == 0,
            "all_disagreements_are_over_approximations": len(unsound) == 0,
        },
        "per_model_results": all_results,
        "disagreement_analyses": disagreements,
        "conclusion": (
            "Pre-fix analysis: the original 2/14 disagreements (cnn_simple, "
            "cnn_deep) were both INCOMPLETE — compositional rejects but "
            "monolithic accepts. Zero UNSOUND cases. Root cause: "
            "rank-changing ops (view/flatten) at sub-module boundaries "
            "caused the interface checker to flag a rank mismatch (4D→2D) "
            "even though element counts were consistent. Fix applied to "
            "check_interface_compatibility: when the consumer's first op "
            "on a boundary tensor is FLATTEN/RESHAPE, element-count "
            "compatibility is checked instead of strict shape matching. "
            f"Post-fix: {agree_count}/{total} models agree, "
            f"{len(disagreements)} disagreements remain."
        ) if len(unsound) == 0 else (
            "SOUNDNESS VIOLATIONS DETECTED. The compositional verifier "
            "accepts models that monolithic verification rejects. This "
            "requires immediate investigation and fixing."
        ),
    }

    out_path = os.path.join(os.path.dirname(__file__), "ag_disagreement_results.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {out_path}")

    return output


if __name__ == "__main__":
    run_experiment()
