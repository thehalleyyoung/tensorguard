"""Roadmap step 14 — ``nn.Sequential(*list)`` splat and ``OrderedDict`` form.

PyTorch's ``nn.Sequential`` has two constructions:

* **positional / splat** — ``nn.Sequential(a, b, c)`` / ``nn.Sequential(*layers)``;
  children are named ``0``, ``1``, ``2``, … in ``state_dict``;
* **named** — ``nn.Sequential(OrderedDict([("conv", ...), ("bn", ...)]))`` (or the
  ``OrderedDict(conv=..., bn=...)`` keyword form, or a dict literal); children
  carry their *declared* names.

This suite pins that the derived weights contract names children faithfully in
both forms, cross-checks against the real torch ``state_dict``, and abstains
(never guesses) when the argument is not statically enumerable.
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


def _codes(contract):
    return {a.code for a in contract.abstained}


_ODICT_IMPORT = "from collections import OrderedDict\nimport torch.nn as nn\n"


# --------------------------------------------------------------------------- #
# named form: OrderedDict list-of-pairs                                        #
# --------------------------------------------------------------------------- #
def test_ordereddict_list_names_children():
    src = _ODICT_IMPORT + """
class M(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.net = nn.Sequential(OrderedDict([
            ("fc1", nn.Linear(d, d)),
            ("act", nn.ReLU()),
            ("fc2", nn.Linear(d, 2 * d)),
        ]))
"""
    c = _derive(src, "M(8)")
    assert c.params["net.fc1.weight"] == (8, 8)
    assert c.params["net.fc2.weight"] == (16, 8)
    assert c.params["net.fc2.bias"] == (16,)
    # the activation (no params) and index keys never appear.
    assert "net.0.weight" not in c.params
    assert "net.act.weight" not in c.params
    assert not c.abstained


def test_ordereddict_kwargs_names_children():
    src = _ODICT_IMPORT + """
class M(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.net = nn.Sequential(OrderedDict(lin=nn.Linear(d, d), bn=nn.BatchNorm1d(d)))
"""
    c = _derive(src, "M(8)")
    assert c.params["net.lin.weight"] == (8, 8)
    assert c.params["net.bn.weight"] == (8,)
    assert c.params["net.bn.running_mean"] == (8,)
    assert "net.0.weight" not in c.params


def test_collections_qualified_ordereddict():
    src = """
import collections
import torch.nn as nn
class M(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.net = nn.Sequential(collections.OrderedDict([("a", nn.Linear(d, d))]))
"""
    c = _derive(src, "M(8)")
    assert c.params["net.a.weight"] == (8, 8)


def test_named_sequential_preserves_declared_order():
    # Declared order is fc_b, fc_a (reverse-alphabetical) — a sorted map would
    # swap them; the contract must keep the *declared* names regardless.
    src = _ODICT_IMPORT + """
class M(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.net = nn.Sequential(OrderedDict([
            ("zeta", nn.Linear(d, 2 * d)),
            ("alpha", nn.Linear(2 * d, d)),
        ]))
"""
    c = _derive(src, "M(8)")
    assert c.params["net.zeta.weight"] == (16, 8)
    assert c.params["net.alpha.weight"] == (8, 16)


# --------------------------------------------------------------------------- #
# splat form                                                                   #
# --------------------------------------------------------------------------- #
def test_splat_list_indexed_children():
    src = """
import torch.nn as nn
class M(nn.Module):
    def __init__(self, d):
        super().__init__()
        layers = [nn.Linear(d, d), nn.ReLU(), nn.Linear(d, 2 * d)]
        self.net = nn.Sequential(*layers)
"""
    c = _derive(src, "M(8)")
    assert c.params["net.0.weight"] == (8, 8)
    assert c.params["net.2.weight"] == (16, 8)
    assert "net.1.weight" not in c.params  # ReLU has no params


def test_splat_comprehension_indexed_children():
    src = """
import torch.nn as nn
class M(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.net = nn.Sequential(*[nn.Linear(d, d) for _ in range(3)])
"""
    c = _derive(src, "M(8)")
    for i in range(3):
        assert c.params[f"net.{i}.weight"] == (8, 8)
    assert "net.3.weight" not in c.params


def test_positional_sequential_still_indexed():
    src = """
import torch.nn as nn
class M(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, d))
"""
    c = _derive(src, "M(8)")
    assert c.params["net.0.weight"] == (8, 8)
    assert c.params["net.2.weight"] == (8, 8)


# --------------------------------------------------------------------------- #
# equivalence: named OrderedDict literal == OrderedDict(kwargs)                 #
# --------------------------------------------------------------------------- #
def test_list_and_kwargs_ordereddict_agree():
    lst = _ODICT_IMPORT + """
class M(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.net = nn.Sequential(OrderedDict([("a", nn.Linear(d, d)), ("b", nn.Linear(d, d))]))
"""
    kwa = _ODICT_IMPORT + """
class M(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.net = nn.Sequential(OrderedDict(a=nn.Linear(d, d), b=nn.Linear(d, d)))
"""
    assert _derive(lst, "M(8)").params == _derive(kwa, "M(8)").params


# --------------------------------------------------------------------------- #
# bonus: nn.ModuleDict(OrderedDict(...))                                        #
# --------------------------------------------------------------------------- #
def test_moduledict_from_ordereddict():
    src = _ODICT_IMPORT + """
class Block(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.fc = nn.Linear(d, d)
class M(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.heads = nn.ModuleDict(OrderedDict([("x", Block(d)), ("y", Block(d))]))
"""
    c = _derive(src, "M(8)")
    assert c.params["heads.x.fc.weight"] == (8, 8)
    assert c.params["heads.y.fc.bias"] == (8,)


# --------------------------------------------------------------------------- #
# abstention: non-enumerable splat / dynamic OrderedDict                        #
# --------------------------------------------------------------------------- #
def test_unknown_splat_abstains():
    src = """
import torch.nn as nn
def build(d):
    pass
class M(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.net = nn.Sequential(*build(d))
"""
    c = _derive(src, "M(8)")
    assert not any(k.startswith("net.") for k in c.params)
    assert AbstainCode.UNENUMERABLE_CONTAINER in _codes(c)


def test_dynamic_ordereddict_abstains_not_misnamed():
    # The OrderedDict is built from a non-enumerable source: abstain on the
    # container rather than emit a single mis-named positional child.
    src = _ODICT_IMPORT + """
def pairs(d):
    pass
class M(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.net = nn.Sequential(OrderedDict(pairs(d)))
"""
    c = _derive(src, "M(8)")
    assert not any(k.startswith("net.") for k in c.params)
    assert AbstainCode.UNENUMERABLE_CONTAINER in _codes(c)


# --------------------------------------------------------------------------- #
# determinism + differential soundness vs torch                                #
# --------------------------------------------------------------------------- #
def test_determinism_named_sequential():
    src = _ODICT_IMPORT + """
class M(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.net = nn.Sequential(OrderedDict([("a", nn.Linear(d, d)), ("b", nn.Linear(d, d))]))
"""
    assert _derive(src, "M(8)").params == _derive(src, "M(8)").params


@pytest.mark.parametrize("construction,src", [
    ("M(8)", _ODICT_IMPORT + """
class M(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.net = nn.Sequential(OrderedDict([
            ("fc1", nn.Linear(d, 2 * d)),
            ("bn", nn.BatchNorm1d(2 * d)),
            ("act", nn.ReLU()),
            ("fc2", nn.Linear(2 * d, d)),
        ]))
        self.head = nn.Linear(d, d, bias=False)
"""),
    ("M(6)", """
import torch.nn as nn
class M(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.net = nn.Sequential(*[nn.Linear(d, d) for _ in range(4)])
"""),
])
def test_sequential_sound_subset_of_torch(construction, src):
    pytest.importorskip("torch")
    from _torch_oracle import state_dict_shapes
    from _differential import subset_verdict

    contract = _derive(src, construction)
    oracle = state_dict_shapes(src, construction)
    verdict = subset_verdict(contract, oracle)
    assert verdict.is_sound, verdict.describe()
    for name, shape in contract.params.items():
        assert oracle[name] == shape
    # named form: the OrderedDict children must be present under their names.
    if "OrderedDict" in src:
        assert "net.fc1.weight" in contract.params
        assert "net.fc2.weight" in contract.params
