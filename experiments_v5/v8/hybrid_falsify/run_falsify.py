"""Hybrid falsification experiment: TensorGuard vs FakeTensorMode.

For each of the 25 blocks, runs:
  - TG: verify_architecture(source, input_shapes=TG_INPUT_SHAPES)
  - For grad blocks (13-20): also verify_grad_flags on a manually built ForwardGraph
  - FT: FakeTensorMode forward with FT_INPUT_SHAPES

Records per-block verdicts and a 2x2 contingency table.
"""
from __future__ import annotations

import contextlib
import importlib
import inspect
import json
import os
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, Optional

import torch

HERE = Path(__file__).resolve().parent
BLOCKS_DIR = HERE / "blocks"
TG_ROOT = Path(__file__).resolve().parents[3]  # tensorguard/
sys.path.insert(0, str(TG_ROOT))

from src.api import verify_architecture  # noqa: E402
from src.v5.grad_flag_verifier import verify_grad_flags  # noqa: E402
from src.v5.backward_shape import ForwardGraph, TensorSpec, Node  # noqa: E402
from torch._subclasses.fake_tensor import FakeTensorMode  # noqa: E402

PREAMBLE = "import torch\nimport torch.nn as nn\nimport torch.nn.functional as F\n"

BLOCK_IDS = [
    "blk_01_hardcoded_batch8",
    "blk_02_hardcoded_batch16",
    "blk_03_else_branch_linear32",
    "blk_04_if_branch_conv_wrong",
    "blk_05_hardcoded_spatial_flatten",
    "blk_06_else_branch_wrong_embed",
    "blk_07_hardcoded_batch32_fc",
    "blk_08_else_wrong_linear_seq",
    "blk_09_hardcoded_pool_view",
    "blk_10_if_branch_wrong_fc_size",
    "blk_11_hardcoded_batch4_transformer",
    "blk_12_else_branch_wrong_conv_channels",
    "blk_13_no_grad_trunk_b1",
    "blk_14_detach_kills_grad_b1",
    "blk_15_frozen_param_b2",
    "blk_16_inplace_relu_leaf_b3",
    "blk_17_no_rg_leaf_b4",
    "blk_18_nested_no_grad_b1",
    "blk_19_frozen_backbone_b2",
    "blk_20_detach_skip_b1",
    "blk_21_view_size0_wrong_fc",
    "blk_22_shape_product_wrong_fc",
    "blk_23_halved_slice_wrong_fc",
    "blk_24_list_stack_mean_wrong_fc",
    "blk_25_adaptive_pool_size0_wrong_fc",
]


# ---------------------------------------------------------------------------
# Hand-built ForwardGraphs for grad blocks
# ---------------------------------------------------------------------------

def _spec(name: str, *, requires_grad: bool, is_leaf: bool = True,
          detached: bool = False) -> TensorSpec:
    return TensorSpec(
        name=name, shape=(2, 32), dtype="float32",
        requires_grad=requires_grad, is_leaf=is_leaf, detached=detached,
    )


def _build_graph_blk_13() -> tuple[ForwardGraph, list[str]]:
    # trunk in no_grad, head outside
    tensors = {
        "x":            _spec("x", requires_grad=False),
        "trunk.weight": _spec("trunk.weight", requires_grad=True),
        "trunk.bias":   _spec("trunk.bias", requires_grad=True),
        "head.weight":  _spec("head.weight", requires_grad=True),
        "head.bias":    _spec("head.bias", requires_grad=True),
        "h1":           _spec("h1", requires_grad=False, is_leaf=False),
        "h2":           _spec("h2", requires_grad=True, is_leaf=False),
        "loss":         _spec("loss", requires_grad=True, is_leaf=False),
    }
    nodes = [
        Node("linear", ["trunk.weight", "trunk.bias", "x"], ["h1"],
             attrs={"no_grad": True}),
        Node("linear", ["head.weight", "head.bias", "h1"], ["h2"]),
        Node("sum", ["h2"], ["loss"]),
    ]
    return ForwardGraph(tensors, nodes, "loss"), [
        "trunk.weight", "trunk.bias", "head.weight", "head.bias"]


def _build_graph_blk_14() -> tuple[ForwardGraph, list[str]]:
    # x.detach() before fc; we model the downstream linear as no_grad to capture the
    # "user thinks fc trains but the whole branch is severed" intent of the spec.
    tensors = {
        "x":         _spec("x", requires_grad=False),
        "x_det":     _spec("x_det", requires_grad=False, is_leaf=False, detached=True),
        "fc.weight": _spec("fc.weight", requires_grad=True),
        "fc.bias":   _spec("fc.bias", requires_grad=True),
        "h":         _spec("h", requires_grad=False, is_leaf=False),
        "loss":      _spec("loss", requires_grad=False, is_leaf=False),
    }
    nodes = [
        Node("detach", ["x"], ["x_det"]),
        Node("linear", ["fc.weight", "fc.bias", "x_det"], ["h"],
             attrs={"no_grad": True}),
        Node("sum", ["h"], ["loss"]),
    ]
    return ForwardGraph(tensors, nodes, "loss"), ["fc.weight", "fc.bias"]


def _build_graph_blk_15() -> tuple[ForwardGraph, list[str]]:
    tensors = {
        "x":         _spec("x", requires_grad=False),
        "fc.weight": _spec("fc.weight", requires_grad=False),  # frozen
        "fc.bias":   _spec("fc.bias", requires_grad=True),
        "h":         _spec("h", requires_grad=True, is_leaf=False),
        "loss":      _spec("loss", requires_grad=True, is_leaf=False),
    }
    nodes = [
        Node("linear", ["fc.weight", "fc.bias", "x"], ["h"]),
        Node("sum", ["h"], ["loss"]),
    ]
    return ForwardGraph(tensors, nodes, "loss"), ["fc.weight", "fc.bias"]


def _build_graph_blk_16() -> tuple[ForwardGraph, list[str]]:
    # data.zero_() in-place on weight inside no_grad: B3 catch
    tensors = {
        "x":         _spec("x", requires_grad=False),
        "fc.weight": _spec("fc.weight", requires_grad=True),
        "fc.bias":   _spec("fc.bias", requires_grad=True),
        "h":         _spec("h", requires_grad=True, is_leaf=False),
        "loss":      _spec("loss", requires_grad=True, is_leaf=False),
    }
    nodes = [
        Node("zero_", ["fc.weight"], ["fc.weight"], inplace=True,
             attrs={"no_grad": True}),
        Node("linear", ["fc.weight", "fc.bias", "x"], ["h"]),
        Node("sum", ["h"], ["loss"]),
    ]
    return ForwardGraph(tensors, nodes, "loss"), ["fc.weight", "fc.bias"]


def _build_graph_blk_17() -> tuple[ForwardGraph, list[str]]:
    tensors = {
        "x":         _spec("x", requires_grad=False),
        "fc.weight": _spec("fc.weight", requires_grad=False),
        "fc.bias":   _spec("fc.bias", requires_grad=False),
        "h":         _spec("h", requires_grad=False, is_leaf=False),
        "loss":      _spec("loss", requires_grad=False, is_leaf=False),
    }
    nodes = [
        Node("linear", ["fc.weight", "fc.bias", "x"], ["h"]),
        Node("sum", ["h"], ["loss"]),
    ]
    return ForwardGraph(tensors, nodes, "loss"), ["fc.weight", "fc.bias"]


def _build_graph_blk_18() -> tuple[ForwardGraph, list[str]]:
    tensors = {
        "x":          _spec("x", requires_grad=False),
        "l1.weight":  _spec("l1.weight", requires_grad=True),
        "l1.bias":    _spec("l1.bias", requires_grad=True),
        "l2.weight":  _spec("l2.weight", requires_grad=True),
        "l2.bias":    _spec("l2.bias", requires_grad=True),
        "head.weight":_spec("head.weight", requires_grad=True),
        "head.bias":  _spec("head.bias", requires_grad=True),
        "h1":         _spec("h1", requires_grad=False, is_leaf=False),
        "h2":         _spec("h2", requires_grad=False, is_leaf=False),
        "h3":         _spec("h3", requires_grad=True, is_leaf=False),
        "loss":       _spec("loss", requires_grad=True, is_leaf=False),
    }
    nodes = [
        Node("linear", ["l1.weight", "l1.bias", "x"], ["h1"], attrs={"no_grad": True}),
        Node("linear", ["l2.weight", "l2.bias", "h1"], ["h2"], attrs={"no_grad": True}),
        Node("linear", ["head.weight", "head.bias", "h2"], ["h3"]),
        Node("sum", ["h3"], ["loss"]),
    ]
    return ForwardGraph(tensors, nodes, "loss"), [
        "l1.weight", "l1.bias", "l2.weight", "l2.bias",
        "head.weight", "head.bias"]


def _build_graph_blk_19() -> tuple[ForwardGraph, list[str]]:
    tensors = {
        "x":               _spec("x", requires_grad=False),
        "backbone.weight": _spec("backbone.weight", requires_grad=False),
        "backbone.bias":   _spec("backbone.bias", requires_grad=False),
        "head.weight":     _spec("head.weight", requires_grad=True),
        "head.bias":       _spec("head.bias", requires_grad=True),
        "h1":              _spec("h1", requires_grad=False, is_leaf=False),
        "h2":              _spec("h2", requires_grad=True, is_leaf=False),
        "loss":            _spec("loss", requires_grad=True, is_leaf=False),
    }
    nodes = [
        Node("linear", ["backbone.weight", "backbone.bias", "x"], ["h1"]),
        Node("linear", ["head.weight", "head.bias", "h1"], ["h2"]),
        Node("sum", ["h2"], ["loss"]),
    ]
    return ForwardGraph(tensors, nodes, "loss"), [
        "backbone.weight", "backbone.bias", "head.weight", "head.bias"]


def _build_graph_blk_20() -> tuple[ForwardGraph, list[str]]:
    tensors = {
        "x":           _spec("x", requires_grad=False),
        "proj.weight": _spec("proj.weight", requires_grad=True),
        "proj.bias":   _spec("proj.bias", requires_grad=True),
        "fc.weight":   _spec("fc.weight", requires_grad=True),
        "fc.bias":     _spec("fc.bias", requires_grad=True),
        "r0":          _spec("r0", requires_grad=True, is_leaf=False),
        "r1":          _spec("r1", requires_grad=False, is_leaf=False, detached=True),
        "s":           _spec("s", requires_grad=False, is_leaf=False),
        "h":           _spec("h", requires_grad=True, is_leaf=False),
        "loss":        _spec("loss", requires_grad=True, is_leaf=False),
    }
    nodes = [
        Node("linear", ["proj.weight", "proj.bias", "x"], ["r0"]),
        Node("detach", ["r0"], ["r1"]),
        Node("add", ["x", "r1"], ["s"]),
        Node("linear", ["fc.weight", "fc.bias", "s"], ["h"]),
        Node("sum", ["h"], ["loss"]),
    ]
    return ForwardGraph(tensors, nodes, "loss"), [
        "proj.weight", "proj.bias", "fc.weight", "fc.bias"]


GRAD_GRAPH_BUILDERS = {
    "blk_13_no_grad_trunk_b1":     _build_graph_blk_13,
    "blk_14_detach_kills_grad_b1": _build_graph_blk_14,
    "blk_15_frozen_param_b2":      _build_graph_blk_15,
    "blk_16_inplace_relu_leaf_b3": _build_graph_blk_16,
    "blk_17_no_rg_leaf_b4":        _build_graph_blk_17,
    "blk_18_nested_no_grad_b1":    _build_graph_blk_18,
    "blk_19_frozen_backbone_b2":   _build_graph_blk_19,
    "blk_20_detach_skip_b1":       _build_graph_blk_20,
}


# ---------------------------------------------------------------------------
# Verdict logic
# ---------------------------------------------------------------------------

def run_tg_shape(block_path: Path, input_shapes: dict) -> Dict[str, Any]:
    try:
        src = block_path.read_text()
        full_source = PREAMBLE + src
        result = verify_architecture(
            full_source,
            input_shapes=input_shapes,
            max_cegar_iterations=3,
            filename=block_path.name,
        )
        bug_count = result.bug_count
        abstained = bool(getattr(result, "abstained", False))
        bugs = [
            {"category": str(b.category), "message": b.message,
             "severity": b.severity}
            for b in result.bugs
        ]
        if bug_count > 0:
            verdict = "Refuted"
        elif abstained:
            verdict = "Abstain"
        else:
            verdict = "Verified"
        return {"verdict": verdict, "bug_count": bug_count,
                "abstained": abstained, "bugs": bugs, "error": None}
    except Exception as e:
        return {"verdict": "Abstain", "bug_count": 0, "abstained": True,
                "bugs": [], "error": f"{type(e).__name__}: {e}"}


def run_tg_grad(block_id: str) -> Dict[str, Any]:
    builder = GRAD_GRAPH_BUILDERS.get(block_id)
    if builder is None:
        return {"verdict": "N/A", "issues": []}
    try:
        graph, params = builder()
        report = verify_grad_flags(graph, params)
        issues = [{"kind": i.kind, "param": i.param, "detail": i.detail}
                  for i in report.issues]
        verdict = "Verified" if report.ok else "Refuted"
        return {"verdict": verdict, "issues": issues, "error": None}
    except Exception as e:
        return {"verdict": "Abstain", "issues": [],
                "error": f"{type(e).__name__}: {e}"}


def _materialize_shape(shape):
    out = []
    for d in shape:
        if isinstance(d, int):
            out.append(d)
        else:
            out.append(2)  # arbitrary concrete value for any symbolic dim
    return tuple(out)


def run_ft(module_cls, input_shapes: dict) -> Dict[str, Any]:
    devnull = open(os.devnull, "w")
    try:
        with contextlib.redirect_stderr(devnull), contextlib.redirect_stdout(devnull):
            try:
                with FakeTensorMode():
                    model = module_cls()
                    args = []
                    for _name, shape in input_shapes.items():
                        args.append(torch.empty(_materialize_shape(shape)))
                    model(*args)
                return {"verdict": "Verified", "error": None}
            except RuntimeError as e:
                msg = str(e).lower()
                shape_kw = ("shape", "size", "mat1", "mat2", "channels",
                            "expected", "dimension", "must match",
                            "reduction dim", "leaf variable", "in-place",
                            "invalid for input")
                if any(k in msg for k in shape_kw):
                    return {"verdict": "Refuted", "error": str(e)[:300]}
                return {"verdict": "Abstain", "error": str(e)[:300]}
            except Exception as e:
                return {"verdict": "Abstain",
                        "error": f"{type(e).__name__}: {str(e)[:300]}"}
    finally:
        devnull.close()


def combine_tg(shape_v: str, grad_v: str) -> str:
    if shape_v == "Refuted" or grad_v == "Refuted":
        return "Refuted"
    if grad_v == "N/A":
        return shape_v
    if shape_v == "Verified" and grad_v == "Verified":
        return "Verified"
    return "Abstain"


def cell_for(tg: str, ft: str) -> str:
    if tg == "Refuted" and ft == "Verified":
        return "TG-only"
    if tg == "Refuted" and ft == "Refuted":
        return "Both"
    if tg == "Verified" and ft == "Refuted":
        return "FT-only"
    if tg == "Verified" and ft == "Verified":
        return "Neither"
    return f"Mixed({tg}/{ft})"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    blocks_results = []
    contingency = {"TG-only": 0, "Both": 0, "FT-only": 0, "Neither": 0,
                   "Other": 0}

    for bid in BLOCK_IDS:
        block_path = BLOCKS_DIR / f"{bid}.py"
        record: Dict[str, Any] = {"id": bid}
        try:
            mod_name = f"experiments_v5.v8.hybrid_falsify.blocks.{bid}"
            mod = importlib.import_module(mod_name)
        except Exception as e:
            record["error"] = f"import failed: {e}"
            blocks_results.append(record)
            continue

        record["category"] = getattr(mod, "CATEGORY", "unknown")
        record["reason"] = getattr(mod, "REASON", "")

        tg_shapes = getattr(mod, "TG_INPUT_SHAPES", {})
        ft_shapes = getattr(mod, "FT_INPUT_SHAPES", {})

        # TG shape
        tg_shape = run_tg_shape(block_path, tg_shapes)
        record["tg_shape_verdict"] = tg_shape["verdict"]
        record["tg_bugs"] = tg_shape["bugs"]
        record["tg_shape_error"] = tg_shape["error"]

        # TG grad (only category B)
        tg_grad = run_tg_grad(bid)
        record["tg_grad_verdict"] = tg_grad["verdict"]
        record["tg_grad_issues"] = tg_grad.get("issues", [])

        record["tg_verdict"] = combine_tg(tg_shape["verdict"], tg_grad["verdict"])

        # FT
        cls = getattr(mod, bid, None)
        if cls is None:
            # try any nn.Module subclass
            for n, obj in inspect.getmembers(mod, inspect.isclass):
                if issubclass(obj, torch.nn.Module) and obj is not torch.nn.Module:
                    cls = obj
                    break
        if cls is None:
            record["ft_verdict"] = "Abstain"
            record["ft_error"] = "no Module class found"
        else:
            ft = run_ft(cls, ft_shapes)
            record["ft_verdict"] = ft["verdict"]
            record["ft_error"] = ft["error"]

        cell = cell_for(record["tg_verdict"], record["ft_verdict"])
        record["cell"] = cell
        if cell in contingency:
            contingency[cell] += 1
        else:
            contingency["Other"] += 1

        blocks_results.append(record)

        print(f"{bid:42s}  TG={record['tg_verdict']:<8s} "
              f"(shape={record['tg_shape_verdict']}, "
              f"grad={record['tg_grad_verdict']})  "
              f"FT={record['ft_verdict']:<8s}  cell={cell}")

    # Per-category breakdown
    by_cat = {}
    for r in blocks_results:
        cat = r.get("category", "unknown")
        by_cat.setdefault(cat, {"TG-only": 0, "Both": 0, "FT-only": 0,
                                "Neither": 0, "Other": 0})
        c = r.get("cell", "Other")
        if c not in by_cat[cat]:
            by_cat[cat]["Other"] += 1
        else:
            by_cat[cat][c] += 1

    summary = (
        f"TG-only={contingency['TG-only']}, Both={contingency['Both']}, "
        f"FT-only={contingency['FT-only']}, Neither={contingency['Neither']}, "
        f"Other={contingency['Other']}"
    )

    out = {
        "blocks": blocks_results,
        "contingency": contingency,
        "by_category": by_cat,
        "summary": summary,
    }
    out_path = HERE / "results.json"
    out_path.write_text(json.dumps(out, indent=2, default=str))

    # Print contingency table
    print("\n" + "=" * 60)
    print("Contingency table (rows=TG, cols=FT):")
    print(f"{'':<12}{'FT Refuted':>14}{'FT Verified':>14}")
    print(f"{'TG Refuted':<12}{contingency['Both']:>14}{contingency['TG-only']:>14}")
    print(f"{'TG Verified':<12}{contingency['FT-only']:>14}{contingency['Neither']:>14}")
    print(f"Other / Abstain: {contingency['Other']}")
    print("\nSummary:", summary)
    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
