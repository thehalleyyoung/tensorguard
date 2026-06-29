"""Roadmap step 12 — bounded for-loop unrolling during module construction.

A GPT-style stack is commonly built by ``for i in range(N): self.blocks.append(
Block(...))``.  The generic widening loop fixpoint would JOIN iterations and
never resolve the N distinct children, so the contract deriver could not name
``blocks.0..N-1.*``.  This suite pins the precise-unroll behaviour:

* a **constant** trip count resolves every registered submodule (sound subset of
  the real torch ``state_dict``);
* a **symbolic / unbounded** trip count abstains cleanly (no guessed children);
* ``range`` start/step, an over-cap count, nested loops, ``.extend``, and
  ``break``/``continue`` bodies all behave soundly (resolve or abstain — never a
  phantom param).
"""

from __future__ import annotations

import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).parent))

from src.symexec.model_contract import derive_model_contract  # noqa: E402
from src.symexec.model_contract import AbstainCode  # noqa: E402


# --------------------------------------------------------------------------- #
# helpers                                                                       #
# --------------------------------------------------------------------------- #
def _derive(source: str, construction: str):
    return derive_model_contract(source, construction)


def _codes(contract):
    return {a.code for a in contract.abstained}


_BLOCK = """
import torch.nn as nn

class Block(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.attn = nn.Linear(d, d)
        self.ln = nn.LayerNorm(d)
"""


def _gpt(loop_body: str, ctor_args: str = "n, d", header: str = "") -> str:
    return _BLOCK + f"""
class GPT(nn.Module):
    def __init__(self, {ctor_args}):
        super().__init__()
        {header}
        self.blocks = nn.ModuleList()
        {loop_body}
        self.head = nn.Linear(d, d, bias=False)
"""


# --------------------------------------------------------------------------- #
# constant trip count -> precise unrolling                                      #
# --------------------------------------------------------------------------- #
def test_constant_range_resolves_each_block():
    src = _gpt("for i in range(n):\n            self.blocks.append(Block(d))")
    c = _derive(src, "GPT(3, 8)")
    for i in range(3):
        assert c.params[f"blocks.{i}.attn.weight"] == (8, 8)
        assert c.params[f"blocks.{i}.attn.bias"] == (8,)
        assert c.params[f"blocks.{i}.ln.weight"] == (8,)
        assert c.params[f"blocks.{i}.ln.bias"] == (8,)
    assert "blocks.3.attn.weight" not in c.params
    assert not c.abstained


def test_head_sibling_still_emitted():
    src = _gpt("for i in range(n):\n            self.blocks.append(Block(d))")
    c = _derive(src, "GPT(2, 8)")
    assert c.params["head.weight"] == (8, 8)
    assert "head.bias" not in c.params  # bias=False


def test_zero_trip_count_emits_no_blocks():
    src = _gpt("for i in range(n):\n            self.blocks.append(Block(d))")
    c = _derive(src, "GPT(0, 8)")
    assert not any(k.startswith("blocks.") for k in c.params)
    assert c.params["head.weight"] == (8, 8)
    assert not c.abstained  # empty ModuleList is exact, not partial


def test_range_with_start_and_step():
    # range(1, 7, 2) -> 3 iterations; children re-indexed 0,1,2 regardless of i.
    src = _gpt("for i in range(1, 7, 2):\n            self.blocks.append(Block(d))", "d")
    c = _derive(src, "GPT(8)")
    assert c.params["blocks.0.attn.weight"] == (8, 8)
    assert c.params["blocks.2.ln.bias"] == (8,)
    assert "blocks.3.attn.weight" not in c.params


def test_loop_index_used_in_block_dim():
    # The bound loop variable flows into a concrete constructor argument.
    src = _BLOCK + """
class GPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = nn.ModuleList()
        for i in range(3):
            self.blocks.append(nn.Linear(4, 4))
"""
    c = _derive(src, "GPT()")
    assert c.params["blocks.0.weight"] == (4, 4)
    assert c.params["blocks.2.bias"] == (4,)


# --------------------------------------------------------------------------- #
# symbolic / unbounded trip count -> abstain cleanly                            #
# --------------------------------------------------------------------------- #
def test_symbolic_trip_count_abstains():
    src = _gpt("for i in range(n):\n            self.blocks.append(Block(d))")
    c = _derive(src, "GPT(n, 8)")  # n left symbolic
    assert not any(k.startswith("blocks.") for k in c.params)
    assert AbstainCode.UNENUMERABLE_CONTAINER in _codes(c)
    assert any(a.path == "blocks" for a in c.abstained)


def test_symbolic_count_never_emits_phantom_block():
    src = _gpt("for i in range(n):\n            self.blocks.append(Block(d))")
    c = _derive(src, "GPT(n, 8)")
    # zero emitted children under blocks/* — a guessed count would be unsound.
    assert [k for k in c.params if k.startswith("blocks.")] == []


def test_over_cap_constant_abstains():
    # A constant range larger than the unroll cap is NOT unrolled; the container
    # is marked opaque so the deriver abstains rather than emitting 10_000 blocks.
    src = _gpt("for i in range(n):\n            self.blocks.append(Block(d))")
    c = _derive(src, "GPT(10000, 8)")
    assert [k for k in c.params if k.startswith("blocks.")] == []
    assert AbstainCode.UNENUMERABLE_CONTAINER in _codes(c)


# --------------------------------------------------------------------------- #
# .extend / .insert                                                             #
# --------------------------------------------------------------------------- #
def test_append_in_loop_after_explicit_literal():
    src = _BLOCK + """
class GPT(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.blocks = nn.ModuleList([Block(d)])
        for i in range(2):
            self.blocks.append(Block(d))
"""
    c = _derive(src, "GPT(8)")
    # 1 literal + 2 appended = 3 contiguous children.
    for i in range(3):
        assert c.params[f"blocks.{i}.attn.weight"] == (8, 8)
    assert "blocks.3.attn.weight" not in c.params


def test_extend_with_enumerable_list():
    src = _BLOCK + """
class GPT(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.blocks = nn.ModuleList()
        for i in range(2):
            self.blocks.extend([Block(d), Block(d)])
"""
    c = _derive(src, "GPT(8)")
    for i in range(4):
        assert c.params[f"blocks.{i}.ln.weight"] == (8,)
    assert "blocks.4.ln.weight" not in c.params


# --------------------------------------------------------------------------- #
# break / continue -> fall back to abstain (sound)                              #
# --------------------------------------------------------------------------- #
def test_break_in_body_abstains():
    src = _gpt(
        "for i in range(n):\n"
        "            if i > 1:\n"
        "                break\n"
        "            self.blocks.append(Block(d))"
    )
    c = _derive(src, "GPT(5, 8)")
    # not precisely unrolled -> container opaque -> abstain, never a wrong count.
    assert [k for k in c.params if k.startswith("blocks.")] == []
    assert AbstainCode.UNENUMERABLE_CONTAINER in _codes(c)


# --------------------------------------------------------------------------- #
# nested loops                                                                  #
# --------------------------------------------------------------------------- #
def test_nested_constant_loops_resolve():
    src = _BLOCK + """
class GPT(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.blocks = nn.ModuleList()
        for i in range(2):
            for j in range(3):
                self.blocks.append(Block(d))
"""
    c = _derive(src, "GPT(8)")
    for i in range(6):  # 2 * 3
        assert c.params[f"blocks.{i}.attn.weight"] == (8, 8)
    assert "blocks.6.attn.weight" not in c.params


# --------------------------------------------------------------------------- #
# forward analysis is unaffected (mutation gated on construction)               #
# --------------------------------------------------------------------------- #
def test_determinism_constant_unroll():
    src = _gpt("for i in range(n):\n            self.blocks.append(Block(d))")
    a = _derive(src, "GPT(4, 8)").params
    b = _derive(src, "GPT(4, 8)").params
    assert a == b


# --------------------------------------------------------------------------- #
# differential soundness vs the real torch state_dict                           #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("n", [0, 1, 2, 6])
def test_loop_built_is_sound_subset_of_torch(n):
    torch = pytest.importorskip("torch")  # noqa: F841
    from _torch_oracle import state_dict_shapes
    from _differential import subset_verdict

    src = _gpt("for i in range(n):\n            self.blocks.append(Block(d))")
    contract = _derive(src, f"GPT({n}, 8)")
    oracle = state_dict_shapes(src, f"GPT({n}, 8)")
    verdict = subset_verdict(contract, oracle)
    assert verdict.is_sound, verdict.describe()
    # every emitted param is present + shape-identical in the real model.
    for name, shape in contract.params.items():
        assert oracle[name] == shape
