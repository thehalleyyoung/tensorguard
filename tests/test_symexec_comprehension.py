"""Roadmap step 13 — list/dict comprehensions of modules.

``nn.ModuleList([Block(d) for _ in range(N)])`` and
``nn.ModuleDict({k: M(d) for k in keys})`` are the idiomatic way to build a
repeated stack.  Step 12 unrolled the explicit ``append`` loop; this step makes
the comprehension form resolve to the *same* contract when the iterable is
statically enumerable, and abstain cleanly otherwise.

The suite pins:

* a constant ``range`` comprehension resolves every ``layers.<i>.*`` child;
* the comprehension contract is **identical** to the explicit-loop contract
  (the roadmap's cross-check acceptance);
* dict comprehensions name children by their (constant-string) keys;
* generator-expression argument form, comprehensions over a list-valued name,
  constant ``if`` filters, and nested generators all resolve precisely;
* a symbolic count / non-constant key / undecidable filter abstains — never a
  guessed or phantom child;
* differential soundness against the real torch ``state_dict``.
"""

from __future__ import annotations

import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).parent))

from src.symexec.model_contract import derive_model_contract  # noqa: E402
from src.symexec.model_contract import AbstainCode  # noqa: E402


_BLOCK = """
import torch.nn as nn

class Block(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.attn = nn.Linear(d, d)
        self.ln = nn.LayerNorm(d)
"""


def _derive(source: str, construction: str):
    return derive_model_contract(source, construction)


def _codes(contract):
    return {a.code for a in contract.abstained}


# --------------------------------------------------------------------------- #
# constant range list-comprehension                                            #
# --------------------------------------------------------------------------- #
def test_listcomp_constant_range_resolves_each_block():
    src = _BLOCK + """
class M(nn.Module):
    def __init__(self, n, d):
        super().__init__()
        self.layers = nn.ModuleList([Block(d) for _ in range(n)])
        self.norm = nn.LayerNorm(d)
"""
    c = _derive(src, "M(4, 8)")
    for i in range(4):
        assert c.params[f"layers.{i}.attn.weight"] == (8, 8)
        assert c.params[f"layers.{i}.ln.bias"] == (8,)
    assert "layers.4.attn.weight" not in c.params
    assert c.params["norm.weight"] == (8,)
    assert not c.abstained


def test_listcomp_matches_explicit_loop_contract():
    """Roadmap acceptance: the comprehension-built stack resolves to the SAME
    contract as the explicit ``append`` loop (step 12)."""
    comp = _BLOCK + """
class M(nn.Module):
    def __init__(self, n, d):
        super().__init__()
        self.blocks = nn.ModuleList([Block(d) for _ in range(n)])
        self.head = nn.Linear(d, d, bias=False)
"""
    loop = _BLOCK + """
class M(nn.Module):
    def __init__(self, n, d):
        super().__init__()
        self.blocks = nn.ModuleList()
        for i in range(n):
            self.blocks.append(Block(d))
        self.head = nn.Linear(d, d, bias=False)
"""
    assert _derive(comp, "M(5, 8)").params == _derive(loop, "M(5, 8)").params


def test_listcomp_zero_trip_is_empty_not_abstain():
    src = _BLOCK + """
class M(nn.Module):
    def __init__(self, n, d):
        super().__init__()
        self.layers = nn.ModuleList([Block(d) for _ in range(n)])
"""
    c = _derive(src, "M(0, 8)")
    assert [k for k in c.params if k.startswith("layers.")] == []
    assert not c.abstained


def test_listcomp_per_element_dims_from_name_iterable():
    src = """
import torch.nn as nn

class M(nn.Module):
    def __init__(self):
        super().__init__()
        dims = [4, 8, 16]
        self.layers = nn.ModuleList([nn.Linear(d, d) for d in dims])
"""
    c = _derive(src, "M()")
    assert c.params["layers.0.weight"] == (4, 4)
    assert c.params["layers.1.weight"] == (8, 8)
    assert c.params["layers.2.weight"] == (16, 16)
    assert not c.abstained


def test_listcomp_constant_filter():
    src = _BLOCK + """
class M(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.layers = nn.ModuleList([Block(d) for i in range(4) if i % 2 == 0])
"""
    c = _derive(src, "M(8)")
    # i in {0, 2} -> 2 children, re-indexed 0, 1.
    assert c.params["layers.0.attn.weight"] == (8, 8)
    assert c.params["layers.1.ln.weight"] == (8,)
    assert "layers.2.attn.weight" not in c.params


def test_listcomp_nested_generators():
    src = _BLOCK + """
class M(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.layers = nn.ModuleList([Block(d) for i in range(2) for j in range(3)])
"""
    c = _derive(src, "M(8)")
    for i in range(6):  # 2 * 3
        assert c.params[f"layers.{i}.attn.weight"] == (8, 8)
    assert "layers.6.attn.weight" not in c.params


def test_generator_expression_argument_form():
    src = _BLOCK + """
class M(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.layers = nn.ModuleList(Block(d) for _ in range(3))
"""
    c = _derive(src, "M(8)")
    for i in range(3):
        assert c.params[f"layers.{i}.attn.weight"] == (8, 8)
    assert "layers.3.attn.weight" not in c.params


# --------------------------------------------------------------------------- #
# dict comprehension                                                           #
# --------------------------------------------------------------------------- #
def test_dictcomp_over_constant_keys():
    src = _BLOCK + """
class M(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.heads = nn.ModuleDict({k: Block(d) for k in ["attn", "mlp", "ff"]})
"""
    c = _derive(src, "M(8)")
    for k in ("attn", "mlp", "ff"):
        assert c.params[f"heads.{k}.attn.weight"] == (8, 8)
        assert c.params[f"heads.{k}.ln.bias"] == (8,)
    assert not c.abstained


def test_dictcomp_matches_explicit_literal():
    comp = _BLOCK + """
class M(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.heads = nn.ModuleDict({k: Block(d) for k in ["a", "b"]})
"""
    lit = _BLOCK + """
class M(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.heads = nn.ModuleDict({"a": Block(d), "b": Block(d)})
"""
    assert _derive(comp, "M(8)").params == _derive(lit, "M(8)").params


# --------------------------------------------------------------------------- #
# abstention: symbolic / undecidable -> never a phantom child                  #
# --------------------------------------------------------------------------- #
def test_symbolic_count_listcomp_abstains():
    src = _BLOCK + """
class M(nn.Module):
    def __init__(self, n, d):
        super().__init__()
        self.layers = nn.ModuleList([Block(d) for _ in range(n)])
        self.norm = nn.LayerNorm(d)
"""
    c = _derive(src, "M(n, 8)")  # n symbolic
    assert [k for k in c.params if k.startswith("layers.")] == []
    assert AbstainCode.UNENUMERABLE_CONTAINER in _codes(c)
    # the enumerable sibling is still emitted (partiality, not total abstention).
    assert c.params["norm.weight"] == (8,)


def test_over_cap_listcomp_abstains():
    src = _BLOCK + """
class M(nn.Module):
    def __init__(self, n, d):
        super().__init__()
        self.layers = nn.ModuleList([Block(d) for _ in range(n)])
"""
    c = _derive(src, "M(10000, 8)")
    assert [k for k in c.params if k.startswith("layers.")] == []
    assert AbstainCode.UNENUMERABLE_CONTAINER in _codes(c)


def test_dictcomp_nonconstant_key_abstains():
    # keys are ints (str(i) would be needed for a name) — non-string key -> the
    # ModuleDict cannot be faithfully named, so abstain rather than guess.
    src = _BLOCK + """
class M(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.heads = nn.ModuleDict({i: Block(d) for i in range(3)})
"""
    c = _derive(src, "M(8)")
    assert [k for k in c.params if k.startswith("heads.")] == []
    assert AbstainCode.UNENUMERABLE_CONTAINER in _codes(c)


# --------------------------------------------------------------------------- #
# determinism + differential soundness vs torch                                #
# --------------------------------------------------------------------------- #
def test_determinism_listcomp():
    src = _BLOCK + """
class M(nn.Module):
    def __init__(self, n, d):
        super().__init__()
        self.layers = nn.ModuleList([Block(d) for _ in range(n)])
"""
    assert _derive(src, "M(4, 8)").params == _derive(src, "M(4, 8)").params


@pytest.mark.parametrize("n", [0, 1, 3, 7])
def test_listcomp_sound_subset_of_torch(n):
    pytest.importorskip("torch")
    from _torch_oracle import state_dict_shapes
    from _differential import subset_verdict

    src = _BLOCK + """
class M(nn.Module):
    def __init__(self, n, d):
        super().__init__()
        self.layers = nn.ModuleList([Block(d) for _ in range(n)])
        self.norm = nn.LayerNorm(d)
"""
    contract = _derive(src, f"M({n}, 8)")
    oracle = state_dict_shapes(src, f"M({n}, 8)")
    verdict = subset_verdict(contract, oracle)
    assert verdict.is_sound, verdict.describe()
    for name, shape in contract.params.items():
        assert oracle[name] == shape


def test_dictcomp_sound_subset_of_torch():
    pytest.importorskip("torch")
    from _torch_oracle import state_dict_shapes
    from _differential import subset_verdict

    src = _BLOCK + """
class M(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.heads = nn.ModuleDict({k: Block(d) for k in ["x", "y", "z"]})
"""
    contract = _derive(src, "M(8)")
    oracle = state_dict_shapes(src, "M(8)")
    verdict = subset_verdict(contract, oracle)
    assert verdict.is_sound, verdict.describe()
    for name, shape in contract.params.items():
        assert oracle[name] == shape
