"""Tests for src.v5.backward_shape."""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import pytest
from src.v5.backward_shape import (
    TensorSpec, Node, ForwardGraph, verify_backward, SHAPE_RULES,
)


def _ts(name, shape, rg=True, sid=None, leaf=True, det=False, dtype="float32"):
    return TensorSpec(name=name, shape=shape, requires_grad=rg,
                      storage_id=sid, is_leaf=leaf, detached=det, dtype=dtype)


# ---------------- baseline OK paths ----------------

def test_simple_linear_relu_ok():
    t = {
        "x":   _ts("x", (4, 8), rg=False, leaf=True),
        "W":   _ts("W", (8, 16)),
        "b":   _ts("b", (16,)),
        "y":   _ts("y", (4, 16)),
        "z":   _ts("z", (4, 16)),
        "loss": _ts("loss", ()),
    }
    g = ForwardGraph(
        tensors=t,
        nodes=[
            Node("linear", ["x", "W", "b"], ["y"]),
            Node("relu",   ["y"],            ["z"]),
            Node("sum",    ["z"],            ["loss"]),
        ],
        loss="loss",
    )
    rep = verify_backward(g)
    assert rep.ok, [str(i) for i in rep.issues]


def test_elementwise_add_broadcast_ok():
    t = {
        "a": _ts("a", (4, 16)),
        "b": _ts("b", (16,)),
        "c": _ts("c", (4, 16)),
        "loss": _ts("loss", ()),
    }
    g = ForwardGraph(t, [
        Node("add", ["a", "b"], ["c"]),
        Node("sum", ["c"], ["loss"]),
    ], "loss")
    rep = verify_backward(g)
    assert rep.ok, rep.issues


# ---------------- failure modes ----------------

def test_loss_not_scalar_flagged():
    t = {"x": _ts("x", (4,)), "loss": _ts("loss", (4,))}
    g = ForwardGraph(t, [Node("relu", ["x"], ["loss"])], "loss")
    rep = verify_backward(g)
    assert not rep.ok
    assert any(i.kind == "shape_mismatch" and "loss" in i.detail for i in rep.issues)


def test_loss_no_requires_grad_flagged():
    t = {"x": _ts("x", (), rg=False), "loss": _ts("loss", (), rg=False)}
    g = ForwardGraph(t, [Node("relu", ["x"], ["loss"])], "loss")
    rep = verify_backward(g)
    assert not rep.ok
    assert any(i.kind == "unreachable_grad" for i in rep.issues)


def test_unknown_op_flagged():
    t = {"x": _ts("x", (4,)), "y": _ts("y", (4,)), "loss": _ts("loss", ())}
    g = ForwardGraph(t, [
        Node("mystery_op", ["x"], ["y"]),
        Node("sum", ["y"], ["loss"]),
    ], "loss")
    rep = verify_backward(g)
    assert not rep.ok
    assert any(i.kind == "unknown_op" for i in rep.issues)


def test_inplace_alias_flagged():
    # x is mutated in-place, but a later node also reads it (same storage).
    t = {
        "x":  _ts("x", (4,), sid=1),
        "x2": _ts("x2", (4,), sid=1),  # alias from in-place
        "y":  _ts("y", (4,)),
        "loss": _ts("loss", ()),
    }
    g = ForwardGraph(t, [
        Node("relu",   ["x"], ["x2"], inplace=True),
        Node("add",    ["x", "x2"], ["y"]),     # still reads storage 1
        Node("sum",    ["y"], ["loss"]),
    ], "loss")
    rep = verify_backward(g)
    assert any(i.kind == "inplace_alias" for i in rep.issues)


def test_detach_severs_grad_path():
    t = {
        "x":  _ts("x", (4,)),
        "xd": _ts("xd", (4,), det=True),
        "loss": _ts("loss", ()),
    }
    g = ForwardGraph(t, [
        Node("detach", ["x"], ["xd"]),
        Node("sum",    ["xd"], ["loss"]),
    ], "loss")
    rep = verify_backward(g)
    # detach rule returns None for grad_input -> no shape mismatch.
    assert rep.ok or all(i.kind != "shape_mismatch" for i in rep.issues)


def test_view_and_reshape_propagate_shape():
    t = {
        "x": _ts("x", (4, 8)),
        "y": _ts("y", (32,)),
        "loss": _ts("loss", ()),
    }
    g = ForwardGraph(t, [
        Node("view", ["x"], ["y"]),
        Node("sum",  ["y"], ["loss"]),
    ], "loss")
    rep = verify_backward(g)
    assert rep.ok, rep.issues


def test_shape_mismatch_detected_when_primal_lies():
    # synthesize a buggy case where the rule says expected != primal
    # by manually corrupting the primal shape.
    t = {
        "x": _ts("x", (4, 8)),
        "y": _ts("y", (4, 8)),
        "loss": _ts("loss", ()),
    }
    g = ForwardGraph(t, [
        Node("relu", ["x"], ["y"]),
        Node("sum",  ["y"], ["loss"]),
    ], "loss")
    # corrupt the rule registration
    orig = SHAPE_RULES["relu"]
    SHAPE_RULES["relu"] = lambda n, env: [(99, 99)]
    try:
        rep = verify_backward(g)
        assert any(i.kind == "shape_mismatch" for i in rep.issues)
    finally:
        SHAPE_RULES["relu"] = orig
