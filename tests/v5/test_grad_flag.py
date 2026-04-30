"""Tests for src.v5.grad_flag_verifier.

Mixes static unit tests with a property-based cross-check against
PyTorch's runtime behavior (hypothesis), covering the four bug classes
B1..B4.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import pytest
import torch
import torch.nn as nn
from hypothesis import given, settings, strategies as st, HealthCheck

from src.v5.backward_shape import TensorSpec, Node, ForwardGraph
from src.v5.grad_flag_verifier import (
    verify_grad_flags, verify_optimizer_step_preconditions, runtime_grad_flags,
)


def _ts(name, shape=(), rg=True, leaf=True, det=False):
    return TensorSpec(name=name, shape=shape, requires_grad=rg,
                      is_leaf=leaf, detached=det)


# ---------------- B1..B4 unit tests ----------------

def test_ok_simple_graph():
    t = {
        "W": _ts("W", (4, 4)),
        "x": _ts("x", (4,), rg=False),
        "y": _ts("y", (4,), leaf=False),
        "loss": _ts("loss", (), leaf=False),
    }
    g = ForwardGraph(t, [
        Node("matmul", ["W", "x"], ["y"]),
        Node("sum", ["y"], ["loss"]),
    ], "loss")
    rep = verify_grad_flags(g, ["W"])
    assert rep.ok, rep.issues
    assert "W" in rep.will_have_grad


def test_B1_no_grad_block():
    # W is used but only inside no_grad -> grad will be None.
    t = {
        "W": _ts("W", (4, 4)),
        "x": _ts("x", (4,), rg=False),
        "y": _ts("y", (4,), leaf=False),
        "loss": _ts("loss", (), leaf=False),
    }
    g = ForwardGraph(t, [
        Node("matmul", ["W", "x"], ["y"], attrs={"no_grad": True}),
        Node("sum", ["y"], ["loss"]),
    ], "loss")
    rep = verify_grad_flags(g, ["W"])
    assert not rep.ok
    assert any(i.kind == "B1" for i in rep.issues)


def test_B1_after_detach():
    t = {
        "W":  _ts("W", (4, 4)),
        "Wd": _ts("Wd", (4, 4), det=True, leaf=False),
        "x":  _ts("x", (4,), rg=False),
        "y":  _ts("y", (4,), leaf=False),
        "loss": _ts("loss", (), leaf=False),
    }
    g = ForwardGraph(t, [
        Node("detach", ["W"], ["Wd"]),
        Node("matmul", ["Wd", "x"], ["y"]),
        Node("sum", ["y"], ["loss"]),
    ], "loss")
    rep = verify_grad_flags(g, ["W"])
    assert any(i.kind == "B1" for i in rep.issues)


def test_B2_requires_grad_false_but_used():
    t = {
        "W": _ts("W", (4,), rg=False),
        "loss": _ts("loss", (), leaf=False),
    }
    g = ForwardGraph(t, [Node("sum", ["W"], ["loss"])], "loss")
    rep = verify_grad_flags(g, ["W"])
    assert any(i.kind in ("B1", "B2") for i in rep.issues)


def test_B3_inplace_on_leaf():
    t = {
        "W": _ts("W", (4,), rg=True, leaf=True),
        "W2": _ts("W2", (4,), leaf=False),
        "loss": _ts("loss", (), leaf=False),
    }
    g = ForwardGraph(t, [
        Node("relu", ["W"], ["W2"], inplace=True),
        Node("sum", ["W2"], ["loss"]),
    ], "loss")
    rep = verify_grad_flags(g, ["W"])
    assert any(i.kind == "B3" for i in rep.issues)


def test_B4_no_leaf_requires_grad():
    t = {
        "x": _ts("x", (4,), rg=False),
        "loss": _ts("loss", (), rg=False, leaf=False),
    }
    g = ForwardGraph(t, [Node("sum", ["x"], ["loss"])], "loss")
    rep = verify_grad_flags(g, [])
    assert any(i.kind == "B4" for i in rep.issues)


def test_optimizer_step_silently_skipped():
    t = {
        "W1": _ts("W1", (4,)),
        "W2": _ts("W2", (4,)),                    # never used
        "loss": _ts("loss", (), leaf=False),
    }
    g = ForwardGraph(t, [Node("sum", ["W1"], ["loss"])], "loss")
    rep = verify_optimizer_step_preconditions(g, ["W1", "W2"])
    assert "W2" in rep.silently_skipped
    assert "W1" not in rep.silently_skipped


# ---------------- Runtime cross-check (small fixed models) ----------------

class TinyOK(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(4, 4)
    def forward(self, x): return self.fc(x)


class TinyFrozen(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(4, 4)
        for p in self.fc.parameters():
            p.requires_grad_(False)
    def forward(self, x): return self.fc(x)


class TinyNoGrad(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(4, 4)
    def forward(self, x):
        with torch.no_grad():
            return self.fc(x)


def test_runtime_tiny_ok():
    m = TinyOK()
    flags = runtime_grad_flags(m, [torch.randn(2, 4)])
    assert flags["fc.weight"] and flags["fc.bias"]


def test_runtime_tiny_frozen_no_grad():
    m = TinyFrozen()
    # backward will raise; helper guards via requires_grad check
    flags = runtime_grad_flags(m, [torch.randn(2, 4)])
    assert not any(flags.values())


def test_runtime_tiny_no_grad_block():
    m = TinyNoGrad()
    flags = runtime_grad_flags(m, [torch.randn(2, 4)])
    assert not any(flags.values())


# ---------------- Property-based: random small models ----------------

def _build_model(arch):
    """arch = list of ('linear', in, out, freeze, no_grad)."""
    layers = []
    for kind, *args in arch:
        in_f, out_f, freeze, ng = args
        l = nn.Linear(in_f, out_f)
        if freeze:
            for p in l.parameters():
                p.requires_grad_(False)
        layers.append((l, ng))

    class M(nn.Module):
        def __init__(self):
            super().__init__()
            for i, (l, _) in enumerate(layers):
                self.add_module(f"l{i}", l)
        def forward(self, x):
            for l, ng in layers:
                if ng:
                    with torch.no_grad():
                        x = l(x)
                else:
                    x = l(x)
            return x
    return M()


def _to_graph(model, in_dim):
    """Build a ForwardGraph from the model description."""
    tensors = {"x": _ts("x", (1, in_dim), rg=False, leaf=True)}
    nodes = []
    prev = "x"
    last_dim = in_dim
    for i, (name, mod) in enumerate(model.named_children()):
        wname, bname = f"{name}.weight", f"{name}.bias"
        out_dim = mod.out_features
        rg_w = mod.weight.requires_grad
        rg_b = mod.bias.requires_grad
        # Detect no_grad block by reading the closure -- simpler: trace once.
        tensors[wname] = _ts(wname, mod.weight.shape, rg=rg_w)
        tensors[bname] = _ts(bname, mod.bias.shape,   rg=rg_b)
        out_name = f"y{i}"
        tensors[out_name] = _ts(out_name, (1, out_dim), rg=True, leaf=False)
        nodes.append(Node("linear", [prev, wname, bname], [out_name]))
        prev = out_name
        last_dim = out_dim
    tensors["loss"] = _ts("loss", (), rg=True, leaf=False)
    nodes.append(Node("sum", [prev], ["loss"]))
    return ForwardGraph(tensors, nodes, "loss")


arch_st = st.lists(
    st.tuples(
        st.just("linear"),
        st.integers(2, 6),
        st.integers(2, 6),
        st.booleans(),         # freeze
        st.just(False),        # we don't encode no_grad in graph builder here
    ),
    min_size=1, max_size=3,
)


@settings(max_examples=200, deadline=None,
          suppress_health_check=[HealthCheck.too_slow])
@given(arch_st)
def test_static_agrees_with_runtime(arch):
    # chain dimension consistency
    fixed = []
    prev = arch[0][1]
    for kind, in_f, out_f, freeze, ng in arch:
        fixed.append((kind, prev, out_f, freeze, ng))
        prev = out_f
    model = _build_model(fixed)
    g = _to_graph(model, fixed[0][1])
    pnames = [n for n, _ in model.named_parameters()]
    static = verify_optimizer_step_preconditions(g, pnames)
    runtime = runtime_grad_flags(model, [torch.randn(1, fixed[0][1])])
    runtime_skipped = {p for p, has in runtime.items() if not has}
    static_skipped = set(static.silently_skipped)
    # The static set should equal the runtime set.
    assert runtime_skipped == static_skipped, (
        f"disagreement runtime={runtime_skipped} static={static_skipped}")
