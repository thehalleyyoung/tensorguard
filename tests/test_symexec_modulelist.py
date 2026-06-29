"""Roadmap **step 11 — enumerate ``nn.ModuleList`` / ``nn.ModuleDict``**.

The deriver now resolves the *registered* children of an ``nn.ModuleList`` /
``nn.ModuleDict`` built from a statically-enumerable literal, naming them exactly
as PyTorch does (``list.<i>.*`` / ``dict.<key>.*``).  When the contents are not
statically enumerable (a comprehension over a symbolic count) it abstains on the
container subtree rather than guessing.

Critically, this step also closes a **soundness hole**: a *plain* Python
``list``/``tuple`` of modules is **not** registered by PyTorch (its children
never enter ``state_dict``), so the deriver must emit nothing for it — previously
it invented ``attr.0.*`` params, a false positive.  These tests pin both the new
enumeration capability and the soundness fix, including differential checks
against the real torch ``state_dict``.
"""

from __future__ import annotations

import pathlib
import sys

import pytest

from src.symexec import derive_model_contract
from src.symexec.model_contract import AbstainCode

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from _differential import subset_verdict  # noqa: E402
from _torch_oracle import state_dict_shapes  # noqa: E402

_BLOCK = """
import torch.nn as nn
class Block(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.fc = nn.Linear(d, d)
        self.ln = nn.LayerNorm(d)
"""


def _mc(body: str, ctor: str):
    return derive_model_contract(_BLOCK + body, ctor)


# --------------------------------------------------------------------------- #
# Enumeration of explicit literals.                                             #
# --------------------------------------------------------------------------- #
def test_modulelist_indexes_children():
    mc = _mc("""
class M(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.blocks = nn.ModuleList([Block(d), Block(d)])
""", "M(d=4)")
    assert mc.params["blocks.0.fc.weight"] == (4, 4)
    assert mc.params["blocks.1.ln.weight"] == (4,)
    assert not mc.abstained


def test_moduledict_keys_children():
    mc = _mc("""
class M(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.heads = nn.ModuleDict({"a": Block(d), "b": Block(d)})
""", "M(d=4)")
    assert mc.params["heads.a.fc.weight"] == (4, 4)
    assert mc.params["heads.b.fc.bias"] == (4,)
    assert not mc.abstained


def test_empty_modulelist_and_dict_emit_nothing_soundly():
    mc = _mc("""
class M(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.a = nn.ModuleList()
        self.b = nn.ModuleDict()
        self.fc = nn.Linear(d, d)
""", "M(d=4)")
    assert mc.params == {"fc.weight": (4, 4), "fc.bias": (4,)}
    assert not mc.abstained


def test_modulelist_from_tuple_literal():
    mc = _mc("""
class M(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.blocks = nn.ModuleList((Block(d), Block(d), Block(d)))
""", "M(d=4)")
    assert sum(k.endswith("fc.weight") for k in mc.params) == 3
    assert not mc.abstained


def test_nested_modulelist_of_moduledict():
    mc = _mc("""
class M(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.stack = nn.ModuleList([
            nn.ModuleDict({"x": Block(d)}),
            nn.ModuleDict({"y": Block(d)}),
        ])
""", "M(d=4)")
    assert mc.params["stack.0.x.fc.weight"] == (4, 4)
    assert mc.params["stack.1.y.ln.weight"] == (4,)
    assert not mc.abstained


def test_modulelist_with_nn_leaf_children():
    # Children may be leaf nn layers directly, not just user modules.
    mc = derive_model_contract("""
import torch.nn as nn
class M(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(d, d), nn.LayerNorm(d)])
""", "M(d=4)")
    assert mc.params["layers.0.weight"] == (4, 4)
    assert mc.params["layers.1.weight"] == (4,)
    assert not mc.abstained


# --------------------------------------------------------------------------- #
# Abstention on non-enumerable registered containers.                           #
# --------------------------------------------------------------------------- #
def test_symbolic_modulelist_abstains_not_guesses():
    mc = _mc("""
class M(nn.Module):
    def __init__(self, n, d):
        super().__init__()
        self.blocks = nn.ModuleList([Block(d) for _ in range(n)])
        self.norm = nn.LayerNorm(d)
""", "M(n, d=8)")
    assert mc.params["norm.weight"] == (8,)
    assert not any(k.startswith("blocks.") for k in mc.params)
    assert any(a.path == "blocks" and a.code is AbstainCode.UNENUMERABLE_CONTAINER
               for a in mc.abstained)


# --------------------------------------------------------------------------- #
# Soundness fix: plain python containers are NOT registered.                     #
# --------------------------------------------------------------------------- #
def test_plain_list_and_tuple_of_modules_emit_nothing():
    mc = _mc("""
class M(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.plain = [Block(d), Block(d)]
        self.pair = (Block(d),)
        self.norm = nn.LayerNorm(d)
""", "M(d=4)")
    assert mc.params == {"norm.weight": (4,), "norm.bias": (4,)}
    assert not any(k.startswith("plain") or k.startswith("pair") for k in mc.params)
    # A plain container is not an abstention either: PyTorch registers nothing, so
    # emitting nothing is exact, not partial.
    assert not mc.abstained


@pytest.mark.parametrize("container", ["list", "tuple"])
def test_plain_container_is_sound_against_torch(container):
    pytest.importorskip("torch")
    expr = "[Block(d), Block(d)]" if container == "list" else "(Block(d), Block(d))"
    body = f"""
class M(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.plain = {expr}
        self.norm = nn.LayerNorm(d)
"""
    src = _BLOCK + body
    oracle = state_dict_shapes(src, "M(d=4)")
    mc = derive_model_contract(src, "M(d=4)")
    v = subset_verdict(mc, oracle)
    assert v.is_sound, v.describe()
    # torch genuinely registers only the LayerNorm.
    assert set(oracle) == {"norm.weight", "norm.bias"}


# --------------------------------------------------------------------------- #
# Differential: enumerated containers match torch exactly.                       #
# --------------------------------------------------------------------------- #
def test_modulelist_matches_torch_state_dict_exactly():
    pytest.importorskip("torch")
    src = _BLOCK + """
class M(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.blocks = nn.ModuleList([Block(d), Block(d), Block(d)])
        self.head = nn.Linear(d, d, bias=False)
"""
    oracle = state_dict_shapes(src, "M(d=8)")
    mc = derive_model_contract(src, "M(d=8)")
    # Full, exact reproduction (no missing, no unsound).
    assert set(mc.params) == set(oracle)
    v = subset_verdict(mc, oracle)
    assert v.is_sound and v.fraction == 1.0


def test_moduledict_matches_torch_state_dict_exactly():
    pytest.importorskip("torch")
    src = _BLOCK + """
class M(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.heads = nn.ModuleDict({"attn": Block(d), "mlp": Block(d)})
"""
    oracle = state_dict_shapes(src, "M(d=8)")
    mc = derive_model_contract(src, "M(d=8)")
    assert set(mc.params) == set(oracle)
    assert subset_verdict(mc, oracle).is_sound


def test_twelve_block_modulelist_yields_blocks_0_to_11():
    # The roadmap-11 acceptance example.
    pytest.importorskip("torch")
    block_exprs = ", ".join(["Block(d)"] * 12)
    src = _BLOCK + f"""
class GPT(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.blocks = nn.ModuleList([{block_exprs}])
        self.head = nn.Linear(d, d, bias=False)
"""
    oracle = state_dict_shapes(src, "GPT(d=16)")
    mc = derive_model_contract(src, "GPT(d=16)")
    for i in range(12):
        assert mc.params[f"blocks.{i}.fc.weight"] == (16, 16)
    assert "blocks.12.fc.weight" not in mc.params
    assert set(mc.params) == set(oracle)
    assert subset_verdict(mc, oracle).is_sound
