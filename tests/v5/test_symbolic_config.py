"""Tests for v5 symbolic config attribute binding."""
from __future__ import annotations

import pytest

from src.v5.symbolic_config import (
    SymInt, SymExpr,
    symbolic_config, bind_symbolic_attrs, verify_against_instance,
    sym_to_dim, shape_with_config,
)
from src.tensor_shapes import ShapeDim


# ── Synthetic case ─────────────────────────────────────────────────────────

def test_sym_int_equality_by_name():
    a = SymInt("h"); b = SymInt("h")
    assert a == b
    assert hash(a) == hash(b)


def test_sym_int_arithmetic_builds_expr():
    h = SymInt("h"); n = SymInt("n")
    e = 3 * h
    assert isinstance(e, SymExpr) and e.op == "*"
    assert isinstance(h // n, SymExpr)
    assert isinstance(h + 1, SymExpr)


def test_decorator_registers_contract():
    @symbolic_config(attrs={"hidden_size": SymInt("h"),
                            "num_attention_heads": SymInt("n")})
    class Foo:
        pass
    assert hasattr(Foo, "__tensorguard_sym_config__")
    cfg = bind_symbolic_attrs(Foo)
    assert isinstance(cfg.hidden_size, SymInt)
    assert cfg.hidden_size.name == "h"


def test_verify_against_instance_ok():
    @symbolic_config(attrs={"hidden_size": SymInt("h"),
                            "num_attention_heads": SymInt("n")},
                    invariants=[lambda c: c["hidden_size"] % c["num_attention_heads"] == 0])
    class Foo:
        pass

    class Cfg:
        hidden_size = 768
        num_attention_heads = 12
    rep = verify_against_instance(Foo, Cfg())
    assert rep.ok, rep.detail


def test_verify_against_instance_bad_invariant():
    @symbolic_config(attrs={"hidden_size": SymInt("h"),
                            "num_attention_heads": SymInt("n")},
                    invariants=[lambda c: c["hidden_size"] % c["num_attention_heads"] == 0])
    class Bar:
        pass

    class Cfg:
        hidden_size = 770   # not divisible by 12
        num_attention_heads = 12
    rep = verify_against_instance(Bar, Cfg())
    assert not rep.ok
    assert rep.bad_invariants == [0]


def test_verify_missing_attrs():
    @symbolic_config(attrs={"hidden_size": SymInt("h")})
    class Baz:
        pass

    class Cfg:
        pass
    rep = verify_against_instance(Baz, Cfg())
    assert not rep.ok
    assert rep.missing_attrs == ["hidden_size"]


def test_shape_with_config_mixes_ints_and_syms():
    h = SymInt("h")
    s = shape_with_config(("B", 3, h, 64))
    assert s.ndim == 4
    assert s.dims[2].value == "h"


def test_contract_namespace_visible():
    import tensorguard
    assert hasattr(tensorguard, "contract")
    from tensorguard import contract  # noqa: F401
    assert contract.SymInt is SymInt


# ── HF-style stub ───────────────────────────────────────────────────────────

def test_hf_style_stub():
    # Stub mimicking a HuggingFace BertConfig
    class StubConfig:
        hidden_size = 768
        num_attention_heads = 12
        intermediate_size = 3072

    @symbolic_config(attrs={
        "hidden_size":         SymInt("h"),
        "num_attention_heads": SymInt("n"),
        "intermediate_size":   SymInt("i"),
    }, invariants=[lambda c: c["hidden_size"] % c["num_attention_heads"] == 0])
    class StubBertSelfAttention:
        pass

    cfg = bind_symbolic_attrs(StubBertSelfAttention)
    assert cfg.num_attention_heads.name == "n"
    rep = verify_against_instance(StubBertSelfAttention, StubConfig())
    assert rep.ok, rep.detail


def test_real_hf_config_if_importable():
    transformers = pytest.importorskip("transformers")
    cfg = transformers.BertConfig()

    @symbolic_config(attrs={
        "hidden_size":         SymInt("h"),
        "num_attention_heads": SymInt("n"),
    }, invariants=[lambda c: c["hidden_size"] % c["num_attention_heads"] == 0])
    class RealBertAttn:
        pass
    rep = verify_against_instance(RealBertAttn, cfg)
    assert rep.ok, rep.detail
