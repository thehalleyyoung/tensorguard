"""Roadmap step 16 — ``super().__init__()`` and inheritance chains.

Real PyTorch models routinely subclass a base module and rely on
``super().__init__(...)`` to register the inherited submodules::

    class MyAttention(BaseAttention):
        def __init__(self, d):
            super().__init__(d)        # registers q, k, v from the base
            self.out = nn.Linear(d, d)  # plus its own

For the weights contract to be sound (and to actually cover such models) the
deriver must *follow the inheritance chain*: run the base ``__init__`` so its
params land in the contract, resolve inherited methods via the MRO, and keep
``super()`` anchored to the **defining** class so multi-level chains delegate
one level at a time.  Every case here is cross-checked against the real torch
``state_dict``.
"""

from __future__ import annotations

import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).parent))

from src.symexec.model_contract import derive_model_contract  # noqa: E402
from src.symexec.model_contract import AbstainCode  # noqa: E402


def _derive(source: str, construction: str):
    return derive_model_contract(source, construction)


_NN = "import torch.nn as nn\n"


# --------------------------------------------------------------------------- #
# Base + derived submodules both appear.                                        #
# --------------------------------------------------------------------------- #
_BASE_DERIVED = _NN + """
class BaseAttention(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.q = nn.Linear(d, d)
        self.k = nn.Linear(d, d)
        self.v = nn.Linear(d, d)
class MyAttention(BaseAttention):
    def __init__(self, d):
        super().__init__(d)
        self.out = nn.Linear(d, d)
"""


def test_inherited_submodules_are_registered():
    c = _derive(_BASE_DERIVED, "MyAttention(d=8)")
    assert set(c.params) == {
        "q.weight", "q.bias", "k.weight", "k.bias",
        "v.weight", "v.bias", "out.weight", "out.bias",
    }
    assert c.abstained == ()


def test_inherited_param_shapes_are_exact():
    c = _derive(_BASE_DERIVED, "MyAttention(d=8)")
    assert c.params["q.weight"] == (8, 8)
    assert c.params["out.weight"] == (8, 8)


# --------------------------------------------------------------------------- #
# Multi-level chains delegate one level at a time.                              #
# --------------------------------------------------------------------------- #
_THREE_LEVEL = _NN + """
class A(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.a = nn.Linear(d, d)
class B(A):
    def __init__(self, d):
        super().__init__(d)
        self.b = nn.Linear(d, d)
class C(B):
    def __init__(self, d):
        super().__init__(d)
        self.c = nn.Linear(d, d)
"""


def test_three_level_chain_registers_every_level():
    c = _derive(_THREE_LEVEL, "C(d=8)")
    assert set(c.params) == {
        "a.weight", "a.bias", "b.weight", "b.bias", "c.weight", "c.bias",
    }
    assert c.abstained == ()


# --------------------------------------------------------------------------- #
# A subclass with no __init__ of its own inherits the base one.                 #
# --------------------------------------------------------------------------- #
_NO_OVERRIDE = _NN + """
class Base(nn.Module):
    def __init__(self, d=8):
        super().__init__()
        self.lin = nn.Linear(d, d)
        self.bn = nn.BatchNorm1d(d)
class Sub(Base):
    pass
"""


def test_subclass_without_init_inherits_base_init():
    c = _derive(_NO_OVERRIDE, "Sub(d=8)")
    assert "lin.weight" in c.params
    assert "bn.weight" in c.params and "bn.running_mean" in c.params
    assert c.abstained == ()


# --------------------------------------------------------------------------- #
# super().__init__ forwarding positional + keyword args.                        #
# --------------------------------------------------------------------------- #
_KW_BASE = _NN + """
class Block(nn.Module):
    def __init__(self, d, hidden):
        super().__init__()
        self.fc1 = nn.Linear(d, hidden)
        self.fc2 = nn.Linear(hidden, d)
class Net(Block):
    def __init__(self, d):
        super().__init__(d, hidden=d * 4)
        self.head = nn.Linear(d, 10)
"""


def test_super_init_forwards_keyword_arguments():
    c = _derive(_KW_BASE, "Net(d=8)")
    assert c.params["fc1.weight"] == (32, 8)   # hidden = 4*d = 32
    assert c.params["fc2.weight"] == (8, 32)
    assert c.params["head.weight"] == (10, 8)
    assert c.abstained == ()


# --------------------------------------------------------------------------- #
# Multiple inheritance: the (first) user base providing __init__ is followed.   #
# --------------------------------------------------------------------------- #
_MIXIN = _NN + """
class Base(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.base_lin = nn.Linear(d, d)
class Mixin:
    pass
class Net(Base, Mixin):
    def __init__(self, d):
        super().__init__(d)
        self.head = nn.Linear(d, d)
"""


def test_multiple_inheritance_follows_module_base():
    c = _derive(_MIXIN, "Net(d=8)")
    assert set(c.params) == {
        "base_lin.weight", "base_lin.bias", "head.weight", "head.bias",
    }
    assert c.abstained == ()


# --------------------------------------------------------------------------- #
# Composition with step 15: a conditional submodule registered in the BASE.     #
# --------------------------------------------------------------------------- #
_COND_IN_BASE = _NN + """
class Base(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.body = nn.Linear(4, 4)
        if cfg.x:
            self.opt = nn.Linear(4, 4)
class Sub(Base):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.head = nn.Linear(4, 4)
"""


def test_conditional_submodule_in_base_abstains_precisely():
    c = _derive(_COND_IN_BASE, "Sub(cfg)")
    assert set(c.params) == {
        "body.weight", "body.bias", "head.weight", "head.bias",
    }
    assert [(a.path, a.code) for a in c.abstained] == [
        ("opt", AbstainCode.CONDITIONAL_SUBMODULE),
    ]


# --------------------------------------------------------------------------- #
# Inherited base submodule that is itself a user module recurses correctly.     #
# --------------------------------------------------------------------------- #
_NESTED_USER_BASE = _NN + """
class MLP(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.up = nn.Linear(d, 4 * d)
        self.down = nn.Linear(4 * d, d)
class Base(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.mlp = MLP(d)
class Model(Base):
    def __init__(self, d):
        super().__init__(d)
        self.norm = nn.LayerNorm(d)
"""


def test_inherited_nested_user_submodule_recurses():
    c = _derive(_NESTED_USER_BASE, "Model(d=8)")
    assert c.params["mlp.up.weight"] == (32, 8)
    assert c.params["mlp.down.weight"] == (8, 32)
    assert "norm.weight" in c.params
    assert c.abstained == ()


# --------------------------------------------------------------------------- #
# Determinism.                                                                  #
# --------------------------------------------------------------------------- #
def test_inheritance_derivation_is_deterministic():
    first = _derive(_THREE_LEVEL, "C(d=8)")
    for _ in range(5):
        again = _derive(_THREE_LEVEL, "C(d=8)")
        assert again.params == first.params
        assert again.abstained == first.abstained


# --------------------------------------------------------------------------- #
# Differential cross-check against the real torch state_dict.                    #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("source, construction", [
    (_BASE_DERIVED, "MyAttention(d=8)"),
    (_THREE_LEVEL, "C(d=8)"),
    (_NO_OVERRIDE, "Sub(d=8)"),
    (_KW_BASE, "Net(d=8)"),
    (_MIXIN, "Net(d=8)"),
    (_NESTED_USER_BASE, "Model(d=8)"),
])
def test_inheritance_is_exact_against_torch(source, construction):
    pytest.importorskip("torch")
    from _torch_oracle import state_dict_shapes
    from _differential import subset_verdict

    contract = _derive(source, construction)
    oracle = state_dict_shapes(source, construction)
    verdict = subset_verdict(contract, oracle)
    assert verdict.is_sound, verdict.describe()
    # These chains are fully resolved -> exact match (every torch param emitted).
    assert set(contract.params) == set(oracle)
    for name, shape in contract.params.items():
        assert oracle[name] == shape
    assert contract.abstained == ()
