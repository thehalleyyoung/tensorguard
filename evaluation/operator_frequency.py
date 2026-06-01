"""Step 22 -- frequency-prioritised operator long-tail census over real models.

Step 21 produced a static coverage matrix of the *public* operator surface.
This harness measures which operators actually *occur*, and how often, in real
model corpora, so the long-tail implementation work can be prioritised by
real-world impact rather than by raw surface area.

It symbolically traces a fixed corpus of torchvision models with `torch.fx`
and counts every `call_function`, `call_method`, and `call_module` target. Each
operator is cross-referenced against the operators TensorGuard actually reasons
about (the engine's tensor-method op map, the denotational transfer-function
registry, the universal transfer registry, and the `nn.Module` layer map) to
produce a ranked list of covered vs. uncovered operators **weighted by
frequency**.

The committed corpus is restricted to torchvision (always available, and its
fx traces are deterministic for a fixed version) so the artifact is
reproducible. Because the trace depends on the torch/torchvision versions, the
artifact records them and `--check` enforces a byte-identical match only when
both versions agree, otherwise reporting a QUALIFIED skip.

This harness also documents the **Step 22 implementations**: the highest-
frequency previously-uncovered shape operators (`permute`, `expand`, `repeat`)
now have denotational transfer functions (see `src/denotational_semantics.py`).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import warnings
from collections import Counter
from typing import Dict, List, Set

warnings.filterwarnings("ignore")

import torch  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
JSON_PATH = os.path.join(HERE, "operator_frequency.json")
MD_PATH = os.path.join(HERE, "operator_frequency.md")

# Fixed, deterministic torchvision corpus spanning CNNs, mobile nets, vision
# transformers, and modern conv nets.
CORPUS: List[str] = [
    "resnet18", "resnet50", "vgg11", "densenet121", "mobilenet_v2",
    "efficientnet_b0", "squeezenet1_0", "shufflenet_v2_x1_0",
    "mnasnet1_0", "regnet_y_400mf", "convnext_tiny", "vit_b_16", "swin_t",
]


def implemented_operator_census() -> Set[str]:
    """Lowercase names of every operator TensorGuard reasons about.

    Union of: the engine tensor-method op map, the denotational transfer
    registry, the universal transfer registry tails, the functional/torch
    dispatch tables, and the recognised `nn.Module` class names.
    """
    from src import model_checker as mc
    from src import denotational_semantics as den
    from src import graph_compiler as gc
    from src import fx_extractor as fx

    fx._init_module_kind_map()
    names: Set[str] = set()

    # Engine tensor-method ops (view, permute, expand, repeat, ...).
    names.update(n.lower() for n in mc._METHOD_OPS)
    # Denotational transfer functions (OpKind names).
    names.update(op.name.lower() for op in den.OP_SEMANTICS)
    # Universal transfer registry (strip "torch."/"F." prefixes, last token).
    for key in gc._UNIVERSAL_TRANSFER_REGISTRY:
        names.add(key.split(".")[-1].lower())
    # Functional / torch dispatch tables.
    names.update(n.lower() for n in fx._F_FUNC_MAP)
    names.update(n.lower() for n in fx._TORCH_FUNC_MAP)
    # nn.Module classes.
    names.update(c.__name__.lower() for c in fx._MODULE_KIND_MAP)
    # Common element-wise / structural aliases the engine treats as covered.
    names.update({"add", "sub", "mul", "div", "matmul", "cat", "stack",
                  "getitem", "getattr", "_assert"})
    return names


def _target_name(node) -> str:
    t = node.target
    return getattr(t, "__name__", str(t))


def trace_frequencies() -> Counter:
    """Count operator occurrences across the torchvision corpus."""
    import torchvision.models as tvm

    counts: Counter = Counter()
    for name in CORPUS:
        model = getattr(tvm, name)()
        graph = torch.fx.symbolic_trace(model)
        for node in graph.graph.nodes:
            if node.op == "call_function":
                counts[_target_name(node)] += 1
            elif node.op == "call_method":
                counts[str(node.target)] += 1
            elif node.op == "call_module":
                mod = dict(graph.named_modules())[node.target]
                counts[type(mod).__name__] += 1
    return counts


def build_report() -> Dict[str, object]:
    import torchvision

    census = implemented_operator_census()
    counts = trace_frequencies()

    operators = []
    covered_weight = uncovered_weight = 0
    for op, freq in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])):
        is_cov = op.lower() in census
        operators.append({"operator": op, "frequency": freq, "covered": is_cov})
        if is_cov:
            covered_weight += freq
        else:
            uncovered_weight += freq

    total_weight = covered_weight + uncovered_weight
    uncovered_ranked = [o for o in operators if not o["covered"]]
    return {
        "meta": {
            "generated_by": "evaluation/operator_frequency.py",
            "command": "PYTHONPATH=. python3 evaluation/operator_frequency.py",
            "torch_version": torch.__version__,
            "torchvision_version": torchvision.__version__,
            "corpus": CORPUS,
            "n_models": len(CORPUS),
        },
        "summary": {
            "distinct_operators": len(operators),
            "distinct_uncovered": len(uncovered_ranked),
            "total_op_occurrences": total_weight,
            "covered_occurrences": covered_weight,
            "uncovered_occurrences": uncovered_weight,
            "frequency_coverage_ratio": round(covered_weight / total_weight, 4)
            if total_weight else 0.0,
        },
        "step22_implemented": ["permute", "expand", "repeat"],
        "operators": operators,
        "uncovered_ranked": uncovered_ranked,
    }


def _dumps(obj: object) -> str:
    return json.dumps(obj, indent=2, sort_keys=True) + "\n"


def render_markdown(rep: Dict[str, object]) -> str:
    meta = rep["meta"]
    summ = rep["summary"]
    lines = [
        "# Operator frequency census (real model corpora)",
        "",
        ("Operators counted by `torch.fx` trace over %d torchvision models, "
         "weighted by occurrence and cross-referenced against the operators "
         "TensorGuard reasons about. Generated against torch `%s`, torchvision "
         "`%s`." % (meta["n_models"], meta["torch_version"],
                    meta["torchvision_version"])),
        "",
        ("Frequency-weighted coverage: **%d** of **%d** operator occurrences "
         "are covered (ratio %.3f)." % (
             summ["covered_occurrences"], summ["total_op_occurrences"],
             summ["frequency_coverage_ratio"])),
        "",
        ("Step 22 added denotational transfer functions for the highest-"
         "frequency previously-uncovered shape operators: `%s`."
         % "`, `".join(rep["step22_implemented"])),
        "",
        "## Top operators by frequency",
        "",
        "| Operator | Frequency | Covered |",
        "|----------|-----------|---------|",
    ]
    for o in rep["operators"][:30]:
        lines.append("| `%s` | %d | %s |" % (
            o["operator"], o["frequency"], "yes" if o["covered"] else "NO"))
    lines.append("")
    if rep["uncovered_ranked"]:
        lines.append("## Remaining uncovered operators (ranked)")
        lines.append("")
        for o in rep["uncovered_ranked"]:
            lines.append("* `%s` (%d)" % (o["operator"], o["frequency"]))
        lines.append("")
    return "\n".join(lines)


def run(check: bool = False, write: bool = True) -> int:
    rep = build_report()
    text = _dumps(rep)

    if check:
        if not os.path.exists(JSON_PATH):
            print("operator_frequency.json missing; run the harness first")
            return 1
        committed = json.load(open(JSON_PATH))
        cv = committed.get("meta", {})
        if (cv.get("torch_version") != rep["meta"]["torch_version"]
                or cv.get("torchvision_version")
                != rep["meta"]["torchvision_version"]):
            print("QUALIFIED: torch/torchvision version mismatch; skipping "
                  "byte-identical check")
            return 0
        if open(JSON_PATH).read() != text:
            print("operator_frequency.json is stale; run `make operator-frequency`")
            return 1
        md = render_markdown(rep)
        if not os.path.exists(MD_PATH) or open(MD_PATH).read() != md:
            print("operator_frequency.md is stale; run `make operator-frequency`")
            return 1
        print("operator frequency census up to date")
        return 0

    if write:
        with open(JSON_PATH, "w") as fh:
            fh.write(text)
        with open(MD_PATH, "w") as fh:
            fh.write(render_markdown(rep))
    s = rep["summary"]
    print("frequency coverage: %d of %d occurrences covered (ratio %.3f); "
          "%d distinct uncovered" % (
              s["covered_occurrences"], s["total_op_occurrences"],
              s["frequency_coverage_ratio"], s["distinct_uncovered"]))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true",
                    help="Verify the committed census is up to date (version-gated).")
    args = ap.parse_args()
    return run(check=args.check)


if __name__ == "__main__":
    sys.exit(main())
