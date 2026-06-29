"""Tests for the **model→weights contract bridge** (even_more.md quantum leap):
derive a sound, partial, shape-only ``name→shape`` contract from ``nn.Module``
code by symbolically running ``__init__``, then certify a checkpoint against it
with no reference checkpoint."""

from __future__ import annotations

import array
import io
import json
import struct

import pytest

from src.symexec import (
    AbstainCode,
    certify_weights_against_model,
    derive_model_contract,
    model_contract_to_expected,
)
from src.symexec.certify import main

MODEL = """
import torch.nn as nn

class Block(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.attn = nn.Linear(d, d)
        self.ln = nn.LayerNorm(d)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))

class Tiny(nn.Module):
    def __init__(self, vocab, d):
        super().__init__()
        self.tok = nn.Embedding(vocab, d)
        self.block = Block(d)
        self.head = nn.Linear(d, vocab, bias=False)
"""


def _write_ckpt(path, name_to_shape, *, dtype="F32"):
    header = {}
    cursor = 0
    buf = b""
    for n, sh in name_to_shape.items():
        numel = 1
        for x in sh:
            numel *= x
        raw = array.array("f", [0.0] * numel).tobytes()
        header[n] = {"dtype": dtype, "shape": list(sh),
                     "data_offsets": [cursor, cursor + len(raw)]}
        cursor += len(raw)
        buf += raw
    hb = json.dumps(header).encode("utf-8")
    with open(path, "wb") as fh:
        fh.write(struct.pack("<Q", len(hb)))
        fh.write(hb)
        fh.write(buf)


# --------------------------------------------------------------------------- #
# Contract derivation: PyTorch-faithful naming and shapes.                      #
# --------------------------------------------------------------------------- #
def test_derives_pytorch_state_dict_names_and_shapes():
    mc = derive_model_contract(MODEL, "Tiny(vocab=100, d=8)")
    assert mc.model_class == "Tiny"
    assert mc.partial is True
    assert mc.params == {
        "tok.weight": (100, 8),
        "block.attn.weight": (8, 8),
        "block.attn.bias": (8,),
        "block.ln.weight": (8,),
        "block.ln.bias": (8,),
        "block.mlp.0.weight": (32, 8),
        "block.mlp.0.bias": (32,),
        "block.mlp.2.weight": (8, 32),
        "block.mlp.2.bias": (8,),
        "head.weight": (100, 8),  # bias=False -> no head.bias
    }
    assert "head.bias" not in mc.params


def test_bias_false_omits_bias():
    src = """
import torch.nn as nn
class M(nn.Module):
    def __init__(self):
        super().__init__()
        self.a = nn.Linear(4, 4, bias=False)
        self.b = nn.Linear(4, 4)
"""
    mc = derive_model_contract(src, "M()")
    assert "a.bias" not in mc.params
    assert mc.params["b.bias"] == (4,)


def test_unenumerable_container_is_abstained_not_guessed():
    # An nn.ModuleList built from a statically-unenumerable comprehension (the
    # trip count is symbolic) cannot be enumerated; the bridge abstains on it
    # (and records the boundary) instead of guessing, while still resolving the
    # sibling layers it can prove.
    src = """
import torch.nn as nn
class Block(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.fc = nn.Linear(d, d)
class M(nn.Module):
    def __init__(self, n, d):
        super().__init__()
        self.layers = nn.ModuleList([Block(d) for _ in range(n)])
        self.norm = nn.LayerNorm(d)
"""
    mc = derive_model_contract(src, "M(n, d=8)")  # n symbolic -> unenumerable
    # The enumerable sibling resolves...
    assert mc.params["norm.weight"] == (8,)
    assert mc.params["norm.bias"] == (8,)
    # ...the dynamic container is abstained, not invented.
    assert any(a.path == "layers" for a in mc.abstained)
    assert any(a.code is AbstainCode.UNENUMERABLE_CONTAINER
               for a in mc.abstained if a.path == "layers")
    assert not any(k.startswith("layers.") for k in mc.params)


def test_plain_python_list_of_modules_is_not_registered():
    # A *plain* Python list/tuple of modules is NOT registered by PyTorch (its
    # children never enter state_dict). The bridge must emit nothing for it —
    # matching torch exactly — rather than inventing ``layers.0.*`` params, which
    # would be a false positive (a registered ModuleList behaves differently).
    src = """
import torch.nn as nn
class Block(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.fc = nn.Linear(d, d)
class M(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.plain_list = [Block(d), Block(d)]
        self.plain_tuple = (Block(d),)
        self.norm = nn.LayerNorm(d)
"""
    mc = derive_model_contract(src, "M(d=8)")
    assert mc.params == {"norm.weight": (8,), "norm.bias": (8,)}
    assert not any(k.startswith("plain_list") for k in mc.params)
    assert not any(k.startswith("plain_tuple") for k in mc.params)


def test_modulelist_literal_is_enumerated():
    # An nn.ModuleList built from an explicit list literal registers each child
    # under ``blocks.<i>.*`` exactly as PyTorch does.
    src = """
import torch.nn as nn
class Block(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.fc = nn.Linear(d, d)
        self.ln = nn.LayerNorm(d)
class M(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.blocks = nn.ModuleList([Block(d), Block(d), Block(d)])
"""
    mc = derive_model_contract(src, "M(d=8)")
    for i in range(3):
        assert mc.params[f"blocks.{i}.fc.weight"] == (8, 8)
        assert mc.params[f"blocks.{i}.fc.bias"] == (8,)
        assert mc.params[f"blocks.{i}.ln.weight"] == (8,)
    assert "blocks.3.fc.weight" not in mc.params
    assert not mc.abstained


def test_moduledict_literal_is_key_addressed():
    # An nn.ModuleDict built from an explicit dict literal registers each child
    # under ``heads.<key>.*`` exactly as PyTorch does.
    src = """
import torch.nn as nn
class Block(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.fc = nn.Linear(d, d)
class M(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.heads = nn.ModuleDict({"attn": Block(d), "mlp": Block(d)})
"""
    mc = derive_model_contract(src, "M(d=8)")
    assert mc.params["heads.attn.fc.weight"] == (8, 8)
    assert mc.params["heads.mlp.fc.weight"] == (8, 8)
    assert not mc.abstained


def test_twelve_block_transformer_via_modulelist():
    # Acceptance: a 12-block transformer built with an nn.ModuleList literal yields
    # blocks.0..blocks.11.* params; a data-dependent count abstains cleanly.
    block_exprs = ", ".join(["Block(d)"] * 12)
    src = f"""
import torch.nn as nn
class Block(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.attn = nn.Linear(d, d)
        self.ln = nn.LayerNorm(d)
class GPT(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.blocks = nn.ModuleList([{block_exprs}])
        self.head = nn.Linear(d, d, bias=False)
"""
    mc = derive_model_contract(src, "GPT(d=16)")
    for i in range(12):
        assert mc.params[f"blocks.{i}.attn.weight"] == (16, 16)
        assert mc.params[f"blocks.{i}.ln.weight"] == (16,)
    assert "blocks.12.attn.weight" not in mc.params
    assert mc.params["head.weight"] == (16, 16)
    assert not mc.abstained

    # Data-dependent count -> abstain on the container, resolve the sibling.
    dyn = """
import torch.nn as nn
class Block(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.attn = nn.Linear(d, d)
class GPT(nn.Module):
    def __init__(self, n, d):
        super().__init__()
        self.blocks = nn.ModuleList([Block(d) for _ in range(n)])
        self.head = nn.Linear(d, d, bias=False)
"""
    mcd = derive_model_contract(dyn, "GPT(n, d=16)")  # symbolic n -> abstain
    assert mcd.params["head.weight"] == (16, 16)
    assert any(a.path == "blocks" and a.code is AbstainCode.UNENUMERABLE_CONTAINER
               for a in mcd.abstained)
    assert not any(k.startswith("blocks.") for k in mcd.params)


# --------------------------------------------------------------------------- #
# Certifying a checkpoint against model code (no reference checkpoint).          #
# --------------------------------------------------------------------------- #
def test_matching_checkpoint_is_certified(tmp_path):
    mc = derive_model_contract(MODEL, "Tiny(vocab=10, d=4)")
    ckpt = tmp_path / "good.safetensors"
    _write_ckpt(ckpt, mc.params)
    cert, contract = certify_weights_against_model(str(ckpt), MODEL, "Tiny(vocab=10, d=4)")
    assert cert.proven_safe
    assert cert.contract_checked
    assert contract.resolved_layers >= 6


def test_shape_mismatch_is_caught(tmp_path):
    mc = derive_model_contract(MODEL, "Tiny(vocab=10, d=4)")
    shapes = dict(mc.params)
    shapes["head.weight"] = (99, 4)
    ckpt = tmp_path / "bad.safetensors"
    _write_ckpt(ckpt, shapes)
    cert, _ = certify_weights_against_model(str(ckpt), MODEL, "Tiny(vocab=10, d=4)")
    assert not cert.proven_safe
    assert "contract_shape_mismatch" in cert.finding_kinds


def test_missing_required_param_is_caught(tmp_path):
    mc = derive_model_contract(MODEL, "Tiny(vocab=10, d=4)")
    shapes = dict(mc.params)
    del shapes["block.attn.weight"]
    ckpt = tmp_path / "miss.safetensors"
    _write_ckpt(ckpt, shapes)
    cert, _ = certify_weights_against_model(str(ckpt), MODEL, "Tiny(vocab=10, d=4)")
    assert not cert.proven_safe
    assert "contract_missing_key" in cert.finding_kinds


def test_partial_contract_allows_extra_checkpoint_tensors(tmp_path):
    """Soundness: a partial model contract must NOT flag unexpected keys (the
    checkpoint may carry buffers/params the contract could not derive)."""
    mc = derive_model_contract(MODEL, "Tiny(vocab=10, d=4)")
    shapes = dict(mc.params)
    shapes["block.some_registered_buffer"] = (4,)
    ckpt = tmp_path / "extra.safetensors"
    _write_ckpt(ckpt, shapes)
    cert, _ = certify_weights_against_model(str(ckpt), MODEL, "Tiny(vocab=10, d=4)")
    assert cert.proven_safe
    assert "contract_unexpected_key" not in cert.finding_kinds


def test_to_expected_is_shape_only():
    mc = derive_model_contract(MODEL, "Tiny(vocab=10, d=4)")
    exp = model_contract_to_expected(mc)
    for _name, (dtype, shape) in exp.items():
        assert dtype is None  # model code does not fix a dtype
        assert isinstance(shape, tuple)


# --------------------------------------------------------------------------- #
# CLI.                                                                          #
# --------------------------------------------------------------------------- #
def test_cli_weights_against_model(tmp_path):
    model_path = tmp_path / "model.py"
    model_path.write_text(MODEL)
    mc = derive_model_contract(MODEL, "Tiny(vocab=10, d=4)")
    ckpt = tmp_path / "m.safetensors"
    _write_ckpt(ckpt, mc.params)
    out = io.StringIO()
    rc = main(
        ["weights", str(ckpt), "--model", str(model_path),
         "--construct", "Tiny(vocab=10, d=4)"],
        out=out,
    )
    assert rc == 0
    assert "model contract" in out.getvalue()
    assert "CERTIFIED" in out.getvalue()


def test_cli_weights_against_model_mismatch(tmp_path):
    model_path = tmp_path / "model.py"
    model_path.write_text(MODEL)
    mc = derive_model_contract(MODEL, "Tiny(vocab=10, d=4)")
    shapes = dict(mc.params)
    shapes["head.weight"] = (5, 4)
    ckpt = tmp_path / "m.safetensors"
    _write_ckpt(ckpt, shapes)
    out = io.StringIO()
    rc = main(
        ["weights", str(ckpt), "--model", str(model_path),
         "--construct", "Tiny(vocab=10, d=4)"],
        out=out,
    )
    assert rc == 1
    assert "contract_shape_mismatch" in out.getvalue()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
