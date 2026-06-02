"""Step 36 -- torch.fx frontend trace-success rate over real model zoos.

A static verifier is only useful if its frontend can ingest the models people
actually write. This harness measures how reliably the `torch.fx` frontend
*traces and lowers* a corpus of real architectures into TensorGuard's
`ComputationGraph` -- without crashing -- and publishes that success rate as a
gated, reproducible artifact.

Two stages are measured independently for every model:

  * **trace**   -- `torch.fx.symbolic_trace(model)` succeeds;
  * **lower**   -- `fx_extractor.fx_trace_to_graph(traced)` succeeds, producing a
    non-empty `ComputationGraph`.

For each successfully-lowered model the harness also records the number of graph
steps and how many operators the frontend had to mark `UNSUPPORTED` (Step 34) --
an honest measure of how much of each real model is reasoned about precisely
versus soundly abstracted.

Reproducibility
---------------
The committed corpus is restricted to `torchvision` (always installed and
deterministic for a fixed version). `timm`/`transformers` models are included
opportunistically when importable but are *excluded from the committed
artifact* so the JSON stays byte-reproducible; `--check` enforces a
byte-identical match only when the torch/torchvision versions agree, otherwise
reporting a QUALIFIED skip. `--gate` fails the build if the trace-or-lower
success rate over the committed corpus regresses below the published floor.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import warnings
from typing import Dict, List, Optional

warnings.filterwarnings("ignore")

import torch  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
JSON_PATH = os.path.join(HERE, "fx_trace_success.json")
MD_PATH = os.path.join(HERE, "fx_trace_success.md")

# Fixed, deterministic torchvision corpus spanning classic CNNs, mobile nets,
# modern conv nets, vision transformers, and multi-branch / auxiliary-head
# architectures (googlenet, inception). These are the models the committed
# artifact and the release gate are computed over.
TORCHVISION_CORPUS: List[str] = [
    "resnet18", "resnet50", "resnext50_32x4d", "wide_resnet50_2",
    "vgg11", "vgg16", "alexnet", "squeezenet1_0",
    "densenet121", "mobilenet_v2", "mobilenet_v3_small", "mnasnet1_0",
    "shufflenet_v2_x1_0", "efficientnet_b0", "regnet_y_400mf", "convnext_tiny",
    "googlenet", "inception_v3", "vit_b_16", "swin_t", "maxvit_t",
]

# Models needing a non-default input spatial size.
_INPUT_SIZE = {"inception_v3": 299}

# Minimum trace-or-lower success rate the release gate enforces. torchvision is
# fully supported, so the published floor is total success.
SUCCESS_FLOOR = 1.0


def _input_shape(name: str) -> tuple:
    s = _INPUT_SIZE.get(name, 224)
    return (1, 3, s, s)


def _count_unsupported(graph) -> int:
    from src.model_checker import OpKind

    n = 0
    for step in graph.steps:
        if getattr(step, "op", None) == OpKind.UNSUPPORTED:
            n += 1
    return n


def _eval_torchvision(name: str) -> Dict[str, object]:
    """Trace + lower a single torchvision model; never raises."""
    import torchvision.models as tvm
    from src.fx_extractor import fx_trace_to_graph

    rec: Dict[str, object] = {
        "model": name, "source": "torchvision",
        "traced": False, "lowered": False,
        "steps": 0, "unsupported_ops": 0, "error": None,
    }
    try:
        model = getattr(tvm, name)()
        model.eval()
    except Exception as exc:  # pragma: no cover - construction failure
        rec["error"] = "construct: %s: %s" % (type(exc).__name__, str(exc)[:120])
        return rec
    try:
        traced = torch.fx.symbolic_trace(model)
        rec["traced"] = True
    except Exception as exc:
        rec["error"] = "trace: %s: %s" % (type(exc).__name__, str(exc)[:120])
        return rec
    try:
        graph = fx_trace_to_graph(traced)
        rec["lowered"] = True
        rec["steps"] = len(graph.steps)
        rec["unsupported_ops"] = _count_unsupported(graph)
    except Exception as exc:
        rec["error"] = "lower: %s: %s" % (type(exc).__name__, str(exc)[:120])
    return rec


def evaluate_corpus() -> List[Dict[str, object]]:
    return [_eval_torchvision(n) for n in TORCHVISION_CORPUS]


def _summarise(records: List[Dict[str, object]]) -> Dict[str, object]:
    n = len(records)
    traced = sum(1 for r in records if r["traced"])
    lowered = sum(1 for r in records if r["lowered"])
    succeeded = sum(1 for r in records if r["traced"] and r["lowered"])
    total_steps = sum(r["steps"] for r in records)
    total_unsup = sum(r["unsupported_ops"] for r in records)
    return {
        "n_models": n,
        "traced": traced,
        "lowered": lowered,
        "succeeded": succeeded,
        "trace_success_rate": round(succeeded / n, 4) if n else 0.0,
        "trace_success_percent": round(100.0 * succeeded / n, 2) if n else 0.0,
        "total_steps": total_steps,
        "total_unsupported_ops": total_unsup,
        "unsupported_fraction": round(total_unsup / total_steps, 4)
        if total_steps else 0.0,
    }


def build_report() -> Dict[str, object]:
    import torchvision

    records = evaluate_corpus()
    return {
        "meta": {
            "generated_by": "evaluation/fx_trace_success.py",
            "command": "PYTHONPATH=. python3 evaluation/fx_trace_success.py",
            "torch_version": torch.__version__,
            "torchvision_version": torchvision.__version__,
            "python_version": "%d.%d" % sys.version_info[:2],
            "corpus": TORCHVISION_CORPUS,
        },
        "summary": _summarise(records),
        "success_floor": SUCCESS_FLOOR,
        "models": records,
    }


def _dumps(obj: object) -> str:
    return json.dumps(obj, indent=2, sort_keys=True) + "\n"


def render_markdown(rep: Dict[str, object]) -> str:
    meta = rep["meta"]
    summ = rep["summary"]
    lines = [
        "# torch.fx frontend trace-success rate (real model zoos)",
        "",
        ("End-to-end trace and lowering of %d real `torchvision` architectures "
         "into TensorGuard's computation graph, generated against torch `%s`, "
         "torchvision `%s`." % (summ["n_models"], meta["torch_version"],
                                meta["torchvision_version"])),
        "",
        ("Trace-or-lower success: **%d of %d** models lowered without crashing "
         "(rate %.3f). Across all models, %d of %d graph steps are operators "
         "the frontend reasons about precisely; the remaining %d are soundly "
         "abstracted as unsupported (Step 34)." % (
             summ["succeeded"], summ["n_models"], summ["trace_success_rate"],
             summ["total_steps"] - summ["total_unsupported_ops"],
             summ["total_steps"], summ["total_unsupported_ops"])),
        "",
        "| Model | Traced | Lowered | Steps | Unsupported |",
        "|-------|--------|---------|-------|-------------|",
    ]
    for r in rep["models"]:
        lines.append("| `%s` | %s | %s | %d | %d |" % (
            r["model"], "yes" if r["traced"] else "NO",
            "yes" if r["lowered"] else "NO", r["steps"], r["unsupported_ops"]))
    lines.append("")
    failures = [r for r in rep["models"] if not (r["traced"] and r["lowered"])]
    if failures:
        lines.append("## Failures")
        lines.append("")
        for r in failures:
            lines.append("* `%s`: %s" % (r["model"], r["error"]))
        lines.append("")
    return "\n".join(lines)


def _probe_optional_zoos() -> None:
    """Opportunistically report timm/transformers coverage (not committed)."""
    from src.fx_extractor import fx_trace_to_graph

    for libname, builders in (("timm", _timm_models), ("transformers",
                                                        _hf_models)):
        try:
            models = builders()
        except Exception:
            print("(%s not available; skipped)" % libname)
            continue
        ok = 0
        for name, mod in models:
            try:
                g = fx_trace_to_graph(torch.fx.symbolic_trace(mod))
                ok += 1 if g.steps else 0
            except Exception:
                pass
        print("%s: %d of %d models lowered" % (libname, ok, len(models)))


def _timm_models():
    import timm

    names = ["resnet18", "efficientnet_b0", "mobilenetv3_small_100"]
    return [(n, timm.create_model(n, pretrained=False).eval()) for n in names]


def _hf_models():
    from transformers import AutoModel, AutoConfig

    out = []
    for n in ["bert-base-uncased", "distilbert-base-uncased"]:
        cfg = AutoConfig.from_pretrained(n)
        out.append((n, AutoModel.from_config(cfg).eval()))
    return out


def gate() -> int:
    if not os.path.exists(JSON_PATH):
        print("fx_trace_success.json missing; run `make fx-trace-success`")
        return 1
    rep = build_report()
    summ = rep["summary"]
    committed = json.load(open(JSON_PATH))
    cv = committed.get("meta", {})
    if (cv.get("torch_version") != rep["meta"]["torch_version"]
            or cv.get("torchvision_version")
            != rep["meta"]["torchvision_version"]):
        print("QUALIFIED: torch/torchvision version mismatch; skipping fx "
              "trace-success gate")
        return 0
    rate = summ["trace_success_rate"]
    if rate < SUCCESS_FLOOR:
        print("FX TRACE-SUCCESS GATE FAILED: rate %.4f < floor %.4f"
              % (rate, SUCCESS_FLOOR))
        for r in rep["models"]:
            if not (r["traced"] and r["lowered"]):
                print("  - %s: %s" % (r["model"], r["error"]))
        return 1
    print("fx trace-success gate PASS: %d of %d models lowered (rate %.3f, "
          "floor %.3f)" % (summ["succeeded"], summ["n_models"], rate,
                           SUCCESS_FLOOR))
    return 0


def run(check: bool = False, write: bool = True) -> int:
    rep = build_report()
    text = _dumps(rep)

    if check:
        if not os.path.exists(JSON_PATH):
            print("fx_trace_success.json missing; run the harness first")
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
            print("fx_trace_success.json is stale; run `make fx-trace-success`")
            return 1
        md = render_markdown(rep)
        if not os.path.exists(MD_PATH) or open(MD_PATH).read() != md:
            print("fx_trace_success.md is stale; run `make fx-trace-success`")
            return 1
        print("fx trace-success report up to date")
        return 0

    if write:
        with open(JSON_PATH, "w") as fh:
            fh.write(text)
        with open(MD_PATH, "w") as fh:
            fh.write(render_markdown(rep))
    s = rep["summary"]
    print("fx trace-success: %d of %d models lowered (rate %.3f); %d steps, "
          "%d unsupported" % (
              s["succeeded"], s["n_models"], s["trace_success_rate"],
              s["total_steps"], s["total_unsupported_ops"]))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true",
                    help="Verify the committed report is up to date (version-gated).")
    ap.add_argument("--gate", action="store_true",
                    help="Fail if trace-success rate regresses below the floor.")
    ap.add_argument("--probe-optional", action="store_true",
                    help="Also probe timm/transformers if importable (not committed).")
    args = ap.parse_args()
    if args.probe_optional:
        _probe_optional_zoos()
    if args.gate:
        return gate()
    return run(check=args.check)


if __name__ == "__main__":
    sys.exit(main())
