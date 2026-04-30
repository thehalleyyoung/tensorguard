"""Track D experiment driver.

Runs 500 random small models drawn from a tiny grammar, compares the
static grad-flag verifier against PyTorch's actual runtime behavior,
runs a small real-bug case-study suite, and writes
``experiments_v5/track_D_results.json``.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import sys
import time
from dataclasses import dataclass
from typing import Dict, List, Tuple

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import torch
import torch.nn as nn

from src.v5.backward_shape import (
    TensorSpec, Node, ForwardGraph, verify_backward,
)
from src.v5.grad_flag_verifier import (
    verify_grad_flags, verify_optimizer_step_preconditions, runtime_grad_flags,
)


# ---------------------------------------------------------------------------
# Random model grammar
#
#   model     := layer (',' layer)*
#   layer     := linear(in,out, freeze?, no_grad?, detach?, inplace_relu?)
# ---------------------------------------------------------------------------

@dataclass
class L:
    in_f: int
    out_f: int
    freeze: bool
    no_grad: bool
    detach: bool
    inplace_relu: bool


def gen_model(rng: random.Random, max_layers=3) -> List[L]:
    n = rng.randint(1, max_layers)
    layers = []
    prev = rng.randint(2, 6)
    for _ in range(n):
        out_f = rng.randint(2, 6)
        layers.append(L(
            in_f=prev, out_f=out_f,
            freeze=rng.random() < 0.2,
            no_grad=rng.random() < 0.2,
            detach=rng.random() < 0.2,
            inplace_relu=rng.random() < 0.15,
        ))
        prev = out_f
    return layers


def build_torch(arch: List[L]) -> nn.Module:
    linears = [nn.Linear(l.in_f, l.out_f) for l in arch]
    for l, mod in zip(arch, linears):
        if l.freeze:
            for p in mod.parameters():
                p.requires_grad_(False)

    class M(nn.Module):
        def __init__(self):
            super().__init__()
            for i, lin in enumerate(linears):
                self.add_module(f"l{i}", lin)
        def forward(self, x):
            for l, lin in zip(arch, linears):
                if l.no_grad:
                    with torch.no_grad():
                        y = lin(x)
                else:
                    y = lin(x)
                if l.detach:
                    y = y.detach()
                if l.inplace_relu:
                    y = torch.relu_(y)  # in-place but not on a leaf
                x = y
            return x
    return M()


def build_graph(arch: List[L], model: nn.Module) -> ForwardGraph:
    tensors = {"x": TensorSpec("x", (1, arch[0].in_f), requires_grad=False)}
    nodes = []
    prev = "x"
    for i, (l, (lname, mod)) in enumerate(zip(arch, model.named_children())):
        wname, bname = f"{lname}.weight", f"{lname}.bias"
        tensors[wname] = TensorSpec(wname, tuple(mod.weight.shape),
                                    requires_grad=mod.weight.requires_grad)
        tensors[bname] = TensorSpec(bname, tuple(mod.bias.shape),
                                    requires_grad=mod.bias.requires_grad)
        out = f"y{i}"
        tensors[out] = TensorSpec(out, (1, l.out_f),
                                  requires_grad=True, is_leaf=False)
        nodes.append(Node("linear", [prev, wname, bname], [out],
                          attrs={"no_grad": l.no_grad}))
        prev = out
        if l.detach:
            d = f"y{i}_d"
            tensors[d] = TensorSpec(d, (1, l.out_f),
                                    requires_grad=True, is_leaf=False, detached=True)
            nodes.append(Node("detach", [prev], [d]))
            prev = d
        if l.inplace_relu:
            r = f"y{i}_r"
            tensors[r] = TensorSpec(r, (1, l.out_f),
                                    requires_grad=True, is_leaf=False)
            nodes.append(Node("relu", [prev], [r], inplace=True))
            prev = r
    tensors["loss"] = TensorSpec("loss", (), requires_grad=True, is_leaf=False)
    nodes.append(Node("sum", [prev], ["loss"]))
    return ForwardGraph(tensors, nodes, "loss")


# ---------------------------------------------------------------------------
# Property test harness
# ---------------------------------------------------------------------------

def run_property(n_samples: int, seed: int = 0xD) -> Dict:
    rng = random.Random(seed)
    agree = disagree = errors = 0
    disagreements: List[Dict] = []
    sample_hashes: List[str] = []
    t0 = time.time()

    for i in range(n_samples):
        arch = gen_model(rng)
        try:
            model = build_torch(arch)
            g = build_graph(arch, model)
            pnames = [n for n, _ in model.named_parameters()]
            static = verify_optimizer_step_preconditions(g, pnames)
            runtime = runtime_grad_flags(model, [torch.randn(1, arch[0].in_f)])
            runtime_skipped = {p for p, has in runtime.items() if not has}
            static_skipped = set(static.silently_skipped)

            sig = json.dumps([(l.__dict__) for l in arch], sort_keys=True)
            sample_hashes.append(hashlib.sha256(sig.encode()).hexdigest()[:16])

            if runtime_skipped == static_skipped:
                agree += 1
            else:
                disagree += 1
                if len(disagreements) < 10:
                    disagreements.append({
                        "arch": [l.__dict__ for l in arch],
                        "runtime_skipped": sorted(runtime_skipped),
                        "static_skipped": sorted(static_skipped),
                    })
        except Exception as e:
            errors += 1
            if len(disagreements) < 20:
                disagreements.append({
                    "arch": [l.__dict__ for l in arch],
                    "error": f"{type(e).__name__}: {e}",
                })

    dt = time.time() - t0
    rate = agree / max(1, agree + disagree)
    return {
        "n_samples": n_samples,
        "agree": agree,
        "disagree": disagree,
        "errors": errors,
        "agreement_rate": rate,
        "elapsed_sec": round(dt, 2),
        "disagreement_examples": disagreements,
        "sample_sha256_first10": sample_hashes[:10],
    }


# ---------------------------------------------------------------------------
# Real-bug case studies (small, locally constructed reproducers based on
# patterns documented in PyTorch issues #2769, #4132, #7613, #20580,
# #39279, #56380, #69991, #82064 -- all about silently-None grads or
# wrong-shape grads).  Each case constructs the buggy model, builds the
# graph, and asks the verifier to flag it.
# ---------------------------------------------------------------------------

CASES = []
def case(tag, descr):
    def deco(fn):
        CASES.append((tag, descr, fn))
        return fn
    return deco


@case("BUG-A", "param wrapped in no_grad block")
def bug_no_grad():
    arch = [L(4, 4, freeze=False, no_grad=True, detach=False, inplace_relu=False)]
    model = build_torch(arch); g = build_graph(arch, model)
    rep = verify_grad_flags(g, [n for n, _ in model.named_parameters()])
    return any(i.kind == "B1" for i in rep.issues)


@case("BUG-B", "param after .detach()")
def bug_detach():
    arch = [L(4, 4, freeze=False, no_grad=False, detach=True, inplace_relu=False)]
    model = build_torch(arch); g = build_graph(arch, model)
    rep = verify_grad_flags(g, [n for n, _ in model.named_parameters()])
    return any(i.kind == "B1" for i in rep.issues)


@case("BUG-C", "param frozen with requires_grad=False")
def bug_frozen():
    arch = [L(4, 4, freeze=True, no_grad=False, detach=False, inplace_relu=False)]
    model = build_torch(arch); g = build_graph(arch, model)
    pnames = [n for n, _ in model.named_parameters()]
    rep = verify_grad_flags(g, pnames, expected_to_learn=pnames)
    return any(i.kind in ("B1", "B2") for i in rep.issues)


@case("BUG-D", "in-place op on leaf with requires_grad")
def bug_inplace_leaf():
    t = {
        "W": TensorSpec("W", (4,), requires_grad=True),
        "W2": TensorSpec("W2", (4,), requires_grad=True, is_leaf=False),
        "loss": TensorSpec("loss", (), requires_grad=True, is_leaf=False),
    }
    g = ForwardGraph(t, [
        Node("relu", ["W"], ["W2"], inplace=True),
        Node("sum", ["W2"], ["loss"]),
    ], "loss")
    rep = verify_grad_flags(g, ["W"])
    return any(i.kind == "B3" for i in rep.issues)


@case("BUG-E", "no leaf has requires_grad (backward precondition)")
def bug_no_grad_leaf():
    t = {
        "x": TensorSpec("x", (4,), requires_grad=False),
        "loss": TensorSpec("loss", (), requires_grad=False, is_leaf=False),
    }
    g = ForwardGraph(t, [Node("sum", ["x"], ["loss"])], "loss")
    rep = verify_grad_flags(g, [])
    return any(i.kind == "B4" for i in rep.issues)


@case("BUG-F", "wrong-shape grad: in-place alias mutates needed tensor")
def bug_inplace_alias():
    t = {
        "x":  TensorSpec("x", (4,), requires_grad=True, storage_id=1),
        "x2": TensorSpec("x2", (4,), requires_grad=True, storage_id=1, is_leaf=False),
        "y":  TensorSpec("y", (4,), requires_grad=True, is_leaf=False),
        "loss": TensorSpec("loss", (), requires_grad=True, is_leaf=False),
    }
    g = ForwardGraph(t, [
        Node("relu", ["x"], ["x2"], inplace=True),
        Node("add",  ["x", "x2"], ["y"]),
        Node("sum",  ["y"], ["loss"]),
    ], "loss")
    rep = verify_backward(g)
    return any(i.kind == "inplace_alias" for i in rep.issues)


@case("BUG-G", "loss is non-scalar")
def bug_nonscalar_loss():
    t = {"x": TensorSpec("x", (4,), requires_grad=True),
         "loss": TensorSpec("loss", (4,), requires_grad=True, is_leaf=False)}
    g = ForwardGraph(t, [Node("relu", ["x"], ["loss"])], "loss")
    return not verify_backward(g).ok


@case("BUG-H", "param never used in forward (silent skip)")
def bug_unused():
    t = {
        "W_used":   TensorSpec("W_used", (4,), requires_grad=True),
        "W_unused": TensorSpec("W_unused", (4,), requires_grad=True),
        "loss": TensorSpec("loss", (), requires_grad=True, is_leaf=False),
    }
    g = ForwardGraph(t, [Node("sum", ["W_used"], ["loss"])], "loss")
    rep = verify_optimizer_step_preconditions(g, ["W_used", "W_unused"])
    return "W_unused" in rep.silently_skipped


def run_case_studies() -> Dict:
    results = []
    caught = 0
    for tag, descr, fn in CASES:
        ok = bool(fn())
        results.append({"tag": tag, "description": descr, "caught": ok})
        if ok:
            caught += 1
    return {
        "total": len(CASES),
        "caught": caught,
        "recall": caught / len(CASES),
        "details": results,
    }


# ---------------------------------------------------------------------------
# False-positive sanity: 50 hand-built CORRECT models -- verifier must not
# flag them.
# ---------------------------------------------------------------------------

def run_false_positive_sweep(n: int = 50, seed: int = 0xFA15E) -> Dict:
    rng = random.Random(seed)
    fps = 0
    for _ in range(n):
        # Generate arch with no freeze / no_grad / detach to be a "clean" model.
        arch = []
        prev = rng.randint(2, 6)
        for _ in range(rng.randint(1, 3)):
            out_f = rng.randint(2, 6)
            arch.append(L(prev, out_f, False, False, False, False))
            prev = out_f
        model = build_torch(arch)
        g = build_graph(arch, model)
        pnames = [n for n, _ in model.named_parameters()]
        rep = verify_grad_flags(g, pnames)
        if not rep.ok:
            fps += 1
    return {"n": n, "false_positives": fps}


def main():
    out_path = os.path.join(ROOT, "experiments_v5", "track_D_results.json")
    print("[Track D] running 500-sample property test...")
    prop = run_property(500)
    print(f"  agreement rate: {prop['agreement_rate']:.4f} "
          f"({prop['agree']}/{prop['agree']+prop['disagree']})")

    print("[Track D] running real-bug case studies...")
    cases = run_case_studies()
    print(f"  recall: {cases['caught']}/{cases['total']}")

    print("[Track D] running false-positive sweep...")
    fps = run_false_positive_sweep(50)
    print(f"  false positives: {fps['false_positives']}/{fps['n']}")

    payload = {
        "track": "D",
        "version": "v5",
        "torch_version": torch.__version__,
        "property_test": prop,
        "case_studies": cases,
        "false_positive_sweep": fps,
    }
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"[Track D] wrote {out_path}")


if __name__ == "__main__":
    main()
