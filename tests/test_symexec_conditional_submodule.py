"""Roadmap step 15 — conditional submodule registration.

A PyTorch ``__init__`` routinely registers a submodule behind a flag::

    if cfg.use_proj:
        self.proj = nn.Linear(d, d)

The derived weights contract must be **path-precise**:

* when the guard is *statically resolved* (a concrete ``bool``/``int``/``None``
  flag, or a comparison of constants) the deriver takes the correct branch and
  emits exactly the params PyTorch would register;
* when the guard is *symbolic* (an unresolved attribute/parameter) the deriver
  **abstains on the conditional subtree only** — it never invents the params
  (treating the submodule as always present) nor silently drops them (treating
  it as always absent).  The rest of the model still resolves.

The abstention is reported with the dedicated
``AbstainCode.CONDITIONAL_SUBMODULE`` reason so contract coverage stays
measurable.  Concrete-flag cases are cross-checked against the real torch
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


def _codes(contract):
    return {a.code for a in contract.abstained}


def _abstain_paths(contract):
    return {a.path for a in contract.abstained}


_NN = "import torch.nn as nn\n"


# --------------------------------------------------------------------------- #
# Concrete (statically resolved) flags: take the correct branch exactly.        #
# --------------------------------------------------------------------------- #
_BOOL_FLAG = _NN + """
class M(nn.Module):
    def __init__(self, d=8, use_proj=True):
        super().__init__()
        self.fc = nn.Linear(d, d)
        if use_proj:
            self.proj = nn.Linear(d, d * 2)
"""


def test_concrete_true_flag_includes_optional_submodule():
    c = _derive(_BOOL_FLAG, "M(d=8, use_proj=True)")
    assert set(c.params) == {
        "fc.weight", "fc.bias", "proj.weight", "proj.bias",
    }
    assert c.abstained == ()
    assert c.params["proj.weight"] == (16, 8)


def test_concrete_false_flag_excludes_optional_submodule():
    c = _derive(_BOOL_FLAG, "M(d=8, use_proj=False)")
    assert set(c.params) == {"fc.weight", "fc.bias"}
    assert c.abstained == ()


def test_default_argument_resolves_the_flag():
    # No explicit flag -> the default (``True``) is used, fully resolved.
    c = _derive(_BOOL_FLAG, "M(d=8)")
    assert "proj.weight" in c.params
    assert c.abstained == ()


_INT_FLAG = _NN + """
class M(nn.Module):
    def __init__(self, depth=2):
        super().__init__()
        self.stem = nn.Linear(4, 4)
        if depth > 1:
            self.extra = nn.Linear(4, 4)
"""


def test_resolved_comparison_guard_true():
    c = _derive(_INT_FLAG, "M(depth=2)")
    assert "extra.weight" in c.params
    assert c.abstained == ()


def test_resolved_comparison_guard_false():
    c = _derive(_INT_FLAG, "M(depth=1)")
    assert "extra.weight" not in c.params
    assert set(c.params) == {"stem.weight", "stem.bias"}
    assert c.abstained == ()


_NONE_FLAG = _NN + """
class M(nn.Module):
    def __init__(self, head=None):
        super().__init__()
        self.body = nn.Linear(4, 4)
        if head is not None:
            self.head = nn.Linear(4, 4)
"""


def test_none_guard_resolves_absent():
    c = _derive(_NONE_FLAG, "M()")
    assert "head.weight" not in c.params
    assert c.abstained == ()


# --------------------------------------------------------------------------- #
# Symbolic flags: abstain on the conditional subtree only.                      #
# --------------------------------------------------------------------------- #
_SYMBOLIC = _NN + """
class M(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.fc = nn.Linear(8, 8)
        if cfg.use_proj:
            self.proj = nn.Linear(8, 16)
"""


def test_symbolic_flag_abstains_only_on_conditional_subtree():
    c = _derive(_SYMBOLIC, "M(cfg)")
    # The unconditional submodule still resolves fully.
    assert set(c.params) == {"fc.weight", "fc.bias"}
    # The conditional one is abstained, with the dedicated reason, on its path.
    assert _codes(c) == {AbstainCode.CONDITIONAL_SUBMODULE}
    assert _abstain_paths(c) == {"proj"}


def test_symbolic_flag_never_invents_conditional_params():
    # Soundness: ``proj.*`` must NOT appear among emitted params (would be a
    # false "must be present" claim about a possibly-absent submodule).
    c = _derive(_SYMBOLIC, "M(cfg)")
    assert not any(p.startswith("proj.") for p in c.params)


_SYMBOLIC_NESTED = _NN + """
class Block(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.l1 = nn.Linear(d, d)
        self.l2 = nn.Linear(d, d)
class M(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.stem = nn.Linear(8, 8)
        if cfg.use_block:
            self.block = Block(8)
"""


def test_symbolic_conditional_subtree_abstains_as_a_single_unit():
    c = _derive(_SYMBOLIC_NESTED, "M(cfg)")
    assert set(c.params) == {"stem.weight", "stem.bias"}
    # Exactly one abstention, on the subtree root -- not one per nested child.
    assert [(a.path, a.code) for a in c.abstained] == [
        ("block", AbstainCode.CONDITIONAL_SUBMODULE),
    ]


# --------------------------------------------------------------------------- #
# if/else: distinct submodules per branch.                                      #
# --------------------------------------------------------------------------- #
_SYMBOLIC_IF_ELSE = _NN + """
class M(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.shared = nn.Linear(4, 4)
        if cfg.flag:
            self.a = nn.Linear(4, 8)
        else:
            self.b = nn.Linear(4, 16)
"""


def test_symbolic_if_else_abstains_on_both_branch_submodules():
    c = _derive(_SYMBOLIC_IF_ELSE, "M(cfg)")
    assert set(c.params) == {"shared.weight", "shared.bias"}
    assert {(a.path, a.code) for a in c.abstained} == {
        ("a", AbstainCode.CONDITIONAL_SUBMODULE),
        ("b", AbstainCode.CONDITIONAL_SUBMODULE),
    }


_SAME_ATTR_DIFF_SHAPE = _NN + """
class M(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        if cfg.big:
            self.proj = nn.Linear(4, 128)
        else:
            self.proj = nn.Linear(4, 64)
"""


def test_same_attr_conflicting_shapes_abstains():
    # The attribute exists on both paths but with different shapes; the deriver
    # cannot know which, so it must abstain rather than pick one.
    c = _derive(_SAME_ATTR_DIFF_SHAPE, "M(cfg)")
    assert c.params == {}
    assert _abstain_paths(c) == {"proj"}
    assert _codes(c) == {AbstainCode.CONDITIONAL_SUBMODULE}


_SAME_ATTR_SAME_SHAPE = _NN + """
class M(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        if cfg.big:
            self.proj = nn.Linear(4, 64)
        else:
            self.proj = nn.Linear(4, 64)
"""


def test_same_attr_identical_shapes_resolves_without_abstaining():
    # Both paths register the *same* submodule shape -> unconditional in effect,
    # so it resolves precisely and no abstention is produced.
    c = _derive(_SAME_ATTR_SAME_SHAPE, "M(cfg)")
    assert set(c.params) == {"proj.weight", "proj.bias"}
    assert c.abstained == ()


# --------------------------------------------------------------------------- #
# Non-module conditionals must not pollute the contract.                        #
# --------------------------------------------------------------------------- #
_COND_INT = _NN + """
class M(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        if cfg.flag:
            h = 8
        else:
            h = 16
        self.fc = nn.Linear(4, 4)
"""


def test_conditional_non_module_value_does_not_abstain():
    c = _derive(_COND_INT, "M(cfg)")
    assert set(c.params) == {"fc.weight", "fc.bias"}
    assert c.abstained == ()


# --------------------------------------------------------------------------- #
# Forward-analysis isolation: the opacification is gated on construction only.  #
# --------------------------------------------------------------------------- #
def test_conditional_handling_is_construction_scoped():
    # A model whose forward (not __init__) contains an if must be unaffected:
    # only unconditional params are emitted, no spurious conditional abstain.
    src = _NN + """
class M(nn.Module):
    def __init__(self, d=4):
        super().__init__()
        self.a = nn.Linear(d, d)
        self.b = nn.Linear(d, d)
    def forward(self, x, flag):
        if flag:
            x = self.a(x)
        else:
            x = self.b(x)
        return x
"""
    c = _derive(src, "M(d=4)")
    assert set(c.params) == {"a.weight", "a.bias", "b.weight", "b.bias"}
    assert c.abstained == ()


# --------------------------------------------------------------------------- #
# Determinism.                                                                  #
# --------------------------------------------------------------------------- #
def test_contract_is_deterministic_across_repeated_derivations():
    first = _derive(_SYMBOLIC_IF_ELSE, "M(cfg)")
    for _ in range(5):
        again = _derive(_SYMBOLIC_IF_ELSE, "M(cfg)")
        assert again.params == first.params
        assert [(a.path, a.code.value) for a in again.abstained] == [
            (a.path, a.code.value) for a in first.abstained
        ]


# --------------------------------------------------------------------------- #
# Differential cross-check against the real torch state_dict (concrete flags).  #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("construction", [
    "M(d=8, use_proj=True, deep=True)",
    "M(d=8, use_proj=False, deep=True)",
    "M(d=8, use_proj=True, deep=False)",
    "M(d=8, use_proj=False, deep=False)",
])
def test_concrete_flags_are_sound_subset_of_torch(construction):
    pytest.importorskip("torch")
    from _torch_oracle import state_dict_shapes
    from _differential import subset_verdict

    src = _NN + """
class M(nn.Module):
    def __init__(self, d=8, use_proj=True, deep=False):
        super().__init__()
        self.fc = nn.Linear(d, d)
        if use_proj:
            self.proj = nn.Linear(d, d * 2)
        if deep:
            self.block = nn.Sequential(nn.Linear(d, d), nn.ReLU(), nn.Linear(d, d))
"""
    contract = _derive(src, construction)
    oracle = state_dict_shapes(src, construction)
    verdict = subset_verdict(contract, oracle)
    assert verdict.is_sound, verdict.describe()
    # Concrete flags fully resolve -> every torch param is emitted (exact match).
    assert set(contract.params) == set(oracle)
    for name, shape in contract.params.items():
        assert oracle[name] == shape
    assert contract.abstained == ()
