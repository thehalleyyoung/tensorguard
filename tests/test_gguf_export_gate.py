"""Step 215 - GGUF / llama.cpp transformer checkpoint export gates."""

from __future__ import annotations

from collections import OrderedDict

import pytest
import torch
import torch.nn as nn

from src.gguf_export import (
    GGUFExportGateResult,
    GGUFTensorInfo,
    TensorGuardGGUFExportError,
    guarded_gguf_export,
    verify_gguf_export_contract,
)


class TinyLlamaBlock(nn.Module):
    def __init__(self, hidden: int, heads: int, kv_heads: int, ffn: int):
        super().__init__()
        head_dim = hidden // heads
        self.self_attn = nn.Module()
        self.self_attn.q_proj = nn.Linear(hidden, heads * head_dim, bias=False)
        self.self_attn.k_proj = nn.Linear(hidden, kv_heads * head_dim, bias=False)
        self.self_attn.v_proj = nn.Linear(hidden, kv_heads * head_dim, bias=False)
        self.self_attn.o_proj = nn.Linear(heads * head_dim, hidden, bias=False)
        self.mlp = nn.Module()
        self.mlp.gate_proj = nn.Linear(hidden, ffn, bias=False)
        self.mlp.up_proj = nn.Linear(hidden, ffn, bias=False)
        self.mlp.down_proj = nn.Linear(ffn, hidden, bias=False)


class TinyLlama(nn.Module):
    def __init__(
        self,
        *,
        vocab: int = 32,
        hidden: int = 16,
        heads: int = 4,
        kv_heads: int = 2,
        layers: int = 2,
        ffn: int = 32,
    ):
        super().__init__()
        self.model = nn.Module()
        self.model.embed_tokens = nn.Embedding(vocab, hidden)
        self.model.layers = nn.ModuleList(
            [TinyLlamaBlock(hidden, heads, kv_heads, ffn) for _ in range(layers)]
        )
        self.lm_head = nn.Linear(hidden, vocab, bias=False)


def llama_metadata(**overrides):
    meta = {
        "llama.embedding_length": 16,
        "llama.block_count": 2,
        "llama.attention.head_count": 4,
        "llama.attention.head_count_kv": 2,
        "llama.feed_forward_length": 32,
        "llama.rope.dimension_count": 4,
        "tokenizer.ggml.tokens": [f"tok{i}" for i in range(32)],
    }
    meta.update(overrides)
    return meta


def clean_state():
    return TinyLlama().state_dict()


def test_clean_real_llama_state_dict_passes():
    result = verify_gguf_export_contract(clean_state(), llama_metadata())

    assert isinstance(result, GGUFExportGateResult)
    assert result.ok
    assert result.issues == ()
    assert result.layer_count == 2
    assert "model.layers.0.self_attn.q_proj.weight" in result.checked_tensors
    assert "lm_head.weight" in result.checked_tensors


def test_bad_kv_projection_rows_are_rejected():
    tensors = OrderedDict(clean_state())
    tensors["model.layers.0.self_attn.k_proj.weight"] = torch.empty(12, 16)

    result = verify_gguf_export_contract(tensors, llama_metadata())

    assert not result.ok
    issue = next(issue for issue in result.issues if issue.category == "qkv_projection")
    assert issue.tensor_name == "model.layers.0.self_attn.k_proj.weight"
    assert issue.expected_shape == (8, 16)
    assert issue.actual_shape == (12, 16)


def test_packed_qkv_linear_shape_and_gpt2_c_attn_orientation():
    hidden = 16
    q_out = 16
    kv_out = 8
    packed = {
        "model.embed_tokens.weight": torch.empty(32, hidden),
        "lm_head.weight": torch.empty(32, hidden),
        "model.layers.0.self_attn.qkv_proj.weight": torch.empty(q_out + kv_out + kv_out, hidden),
        "model.layers.0.self_attn.o_proj.weight": torch.empty(hidden, q_out),
    }
    meta = llama_metadata(**{"llama.block_count": 1})
    assert verify_gguf_export_contract(packed, meta).ok

    packed["model.layers.0.self_attn.qkv_proj.weight"] = torch.empty(q_out + kv_out, hidden)
    bad = verify_gguf_export_contract(packed, meta)
    assert not bad.ok
    assert any(issue.tensor_name == "model.layers.0.self_attn.qkv_proj.weight" for issue in bad.issues)

    c_attn = {
        "model.embed_tokens.weight": torch.empty(32, hidden),
        "lm_head.weight": torch.empty(32, hidden),
        "transformer.h.0.attn.c_attn.weight": torch.empty(hidden, q_out + kv_out + kv_out),
        "model.layers.0.self_attn.o_proj.weight": torch.empty(hidden, q_out),
    }
    assert verify_gguf_export_contract(c_attn, meta).ok


def test_rotary_dimension_must_be_even_and_fit_head_dim():
    odd = verify_gguf_export_contract(
        clean_state(),
        llama_metadata(**{"llama.rope.dimension_count": 3}),
    )
    too_large = verify_gguf_export_contract(
        clean_state(),
        llama_metadata(**{"llama.rope.dimension_count": 6}),
    )

    assert any(issue.category == "rotary_dims" and "even" in issue.message for issue in odd.issues)
    assert any(
        issue.category == "rotary_dims" and "exceeds" in issue.message
        for issue in too_large.issues
    )


def test_vocab_projection_uses_token_list_length_and_allows_padding():
    tensors = OrderedDict(clean_state())
    tensors["model.embed_tokens.weight"] = torch.empty(40, 16)
    tensors["lm_head.weight"] = torch.empty(40, 16)
    padded = verify_gguf_export_contract(tensors, llama_metadata())
    assert padded.ok

    tensors["lm_head.weight"] = torch.empty(31, 16)
    bad = verify_gguf_export_contract(tensors, llama_metadata())
    assert not bad.ok
    assert any(issue.category == "vocab_projection" for issue in bad.issues)


def test_metadata_defaults_kv_heads_and_rejects_layer_count_mismatch():
    meta = llama_metadata()
    del meta["llama.attention.head_count_kv"]
    mha_tensors = TinyLlama(kv_heads=4).state_dict()
    assert verify_gguf_export_contract(mha_tensors, meta).ok

    bad = verify_gguf_export_contract(clean_state(), llama_metadata(**{"llama.block_count": 3}))
    assert not bad.ok
    assert any(
        issue.category == "metadata_consistency" and "declares 3" in issue.message
        for issue in bad.issues
    )


def test_non_square_output_projection_allowed_when_head_dim_is_explicit():
    tensors = {
        "model.embed_tokens.weight": torch.empty(32, 18),
        "lm_head.weight": torch.empty(32, 18),
        "model.layers.0.self_attn.q_proj.weight": torch.empty(20, 18),
        "model.layers.0.self_attn.k_proj.weight": torch.empty(10, 18),
        "model.layers.0.self_attn.v_proj.weight": torch.empty(10, 18),
        "model.layers.0.self_attn.o_proj.weight": torch.empty(18, 20),
    }
    meta = {
        "llama.embedding_length": 18,
        "llama.block_count": 1,
        "llama.attention.head_count": 4,
        "llama.attention.head_count_kv": 2,
        "llama.attention.head_dim": 5,
        "llama.rope.dimension_count": 4,
        "vocab_size": 32,
    }

    assert verify_gguf_export_contract(tensors, meta).ok


def test_quant_block_sizes_are_checked_per_real_gguf_type():
    tensors = OrderedDict(clean_state())
    tensors["extra_quant.weight"] = GGUFTensorInfo(
        shape=(16, 30),
        quant_type="Q4_0",
    )
    bad = verify_gguf_export_contract(tensors, llama_metadata())
    assert not bad.ok
    assert any(issue.category == "quant_block_size" and "Q4_0" in issue.message for issue in bad.issues)

    tensors["extra_quant.weight"] = GGUFTensorInfo(
        shape=(16, 32),
        quant_type="IQ4_NL",
        block_size=32,
    )
    assert verify_gguf_export_contract(tensors, llama_metadata()).ok

    tensors["extra_quant.weight"] = GGUFTensorInfo(
        shape=(16, 32),
        quant_type="IQ4_NL",
        block_size=256,
    )
    wrong_block = verify_gguf_export_contract(tensors, llama_metadata())
    assert any("block_size=256" in issue.message for issue in wrong_block.issues)


def test_guarded_export_blocks_before_exporter_and_validates_mode():
    reached = []

    def exporter(path):
        reached.append(path)
        return path

    tensors = OrderedDict(clean_state())
    tensors["model.layers.0.self_attn.v_proj.weight"] = torch.empty(12, 16)

    with pytest.raises(TensorGuardGGUFExportError):
        guarded_gguf_export(tensors, llama_metadata(), exporter, "model.gguf")
    assert reached == []

    with pytest.raises(ValueError):
        guarded_gguf_export(clean_state(), llama_metadata(), exporter, "model.gguf", on_violation="raisee")

    assert guarded_gguf_export(clean_state(), llama_metadata(), exporter, "model.gguf") == "model.gguf"
    assert reached == ["model.gguf"]


def test_public_tensorguard_torch_exports_gguf_gate():
    import tensorguard
    from tensorguard.torch import (
        GGUFTensorInfo as PublicInfo,
        TensorGuardGGUFExportError as PublicError,
        guarded_gguf_export as public_guarded,
        verify_gguf_export_contract as public_gate,
    )

    assert public_gate is verify_gguf_export_contract
    assert public_guarded is guarded_gguf_export
    assert PublicError is TensorGuardGGUFExportError
    assert PublicInfo is GGUFTensorInfo
    assert tensorguard.verify_gguf_export_contract is verify_gguf_export_contract
    assert tensorguard.guarded_gguf_export is guarded_gguf_export
