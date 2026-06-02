"""Step 48 -- incremental re-verification.

Validates the dependency-aware verdict cache: fingerprint stability, cache
hits/misses, soundness of dependency tracking (unrelated edits are reused, edits
to the dependency closure are recomputed) and cross-process persistence.
"""
import os

import pytest

from src.incremental import (
    IncrementalVerifier,
    changed_models,
    class_dependency_graph,
    model_fingerprint,
    root_dependency_closure,
)
from src.model_checker import verify_model

_SRC = """
import torch.nn as nn
class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(16, 16)
    def forward(self, x):
        return nn.functional.relu(self.fc(x))
class OtherBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.z = nn.Linear(3, 3)
    def forward(self, x):
        return self.z(x)
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.b = Block()
        self.head = nn.Linear(16, 4)
    def forward(self, x):
        return self.head(self.b(x))
"""

_SHAPES = {"x": ("n", 16)}


def _edit_other(src):
    return src.replace("self.z = nn.Linear(3, 3)", "self.z = nn.Linear(7, 7)")


def _edit_block(src):
    return src.replace("self.fc = nn.Linear(16, 16)", "self.fc = nn.Linear(16, 8)")


def test_root_closure_excludes_unrelated_classes():
    root, closure = root_dependency_closure(_SRC)
    assert root == "Net"
    assert closure == {"Net", "Block"}
    assert "OtherBlock" not in closure


def test_dependency_graph_edges():
    g = class_dependency_graph(_SRC)
    assert "Block" in g["Net"]
    assert g["OtherBlock"] == set()


def test_fingerprint_is_stable():
    assert model_fingerprint(_SRC, _SHAPES) == model_fingerprint(_SRC, _SHAPES)


def test_repeat_verify_hits_cache():
    iv = IncrementalVerifier()
    r1 = iv.verify(_SRC, input_shapes=_SHAPES)
    r2 = iv.verify(_SRC, input_shapes=_SHAPES)
    assert not r1.from_cache and r2.from_cache
    assert r1.safe == r2.safe
    assert iv.hits == 1 and iv.misses == 1


def test_unrelated_edit_is_reused_and_sound():
    iv = IncrementalVerifier()
    base = iv.verify(_SRC, input_shapes=_SHAPES)
    edited = _edit_other(_SRC)
    inc = iv.verify(edited, input_shapes=_SHAPES)
    assert inc.from_cache  # dependency-aware reuse
    assert not changed_models(_SRC, edited, _SHAPES)
    # Soundness: the reused verdict equals a full re-verification.
    full = verify_model(edited, input_shapes=_SHAPES)
    assert inc.safe == full.safe == base.safe


def test_dependency_edit_triggers_recompute():
    iv = IncrementalVerifier()
    iv.verify(_SRC, input_shapes=_SHAPES)
    edited = _edit_block(_SRC)
    inc = iv.verify(edited, input_shapes=_SHAPES)
    assert not inc.from_cache
    assert changed_models(_SRC, edited, _SHAPES)


def test_options_are_part_of_the_key():
    iv = IncrementalVerifier()
    iv.verify(_SRC, input_shapes=_SHAPES)
    # A different input shape is a different obligation -> miss.
    inc = iv.verify(_SRC, input_shapes={"x": ("n", 16, 16)})
    assert not inc.from_cache


def test_cache_persists_across_instances(tmp_path):
    path = os.path.join(str(tmp_path), "cache.json")
    iv = IncrementalVerifier(cache_path=path)
    iv.verify(_SRC, input_shapes=_SHAPES)
    iv.save()
    assert os.path.exists(path)
    iv2 = IncrementalVerifier(cache_path=path)
    inc = iv2.verify(_SRC, input_shapes=_SHAPES)
    assert inc.from_cache
    assert iv2.hits == 1 and iv2.misses == 0


def test_invalidate_forces_recompute():
    iv = IncrementalVerifier()
    iv.verify(_SRC, input_shapes=_SHAPES)
    assert iv.invalidate(_SRC, input_shapes=_SHAPES)
    inc = iv.verify(_SRC, input_shapes=_SHAPES)
    assert not inc.from_cache
