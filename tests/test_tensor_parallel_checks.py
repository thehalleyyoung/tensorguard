"""Tests for the tensor-parallel sharding checker (Step 97, Phase 10)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.tensor_parallel_checks import (  # noqa: E402
    verify_tensor_parallel,
    verify_tensor_parallel_attention,
    llama_gqa_attention,
    megatron_attention,
    megatron_mlp,
    TPAttentionSpec,
    TPKVSharding,
    TPLinearSpec,
    TPKind,
    TPIssueKind,
)
import reproducibility.tensor_parallel_sharding as tps  # noqa: E402

VOLATILE_TOKENS = ("time", "elapsed", "timestamp", "wall", "clock", "_ms",
                   "seconds", "duration", "date")


def _walk_keys(obj):
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield k
            yield from _walk_keys(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from _walk_keys(v)


def test_correct_megatron_mlp_is_ok():
    assert verify_tensor_parallel(megatron_mlp(8, 16), 2).ok is True
    assert verify_tensor_parallel(megatron_mlp(64, 256), 8).ok is True


def test_indivisible_shard_flagged():
    res = verify_tensor_parallel(megatron_mlp(8, 15), 2)
    assert not res.ok
    assert TPIssueKind.INDIVISIBLE_SHARD in {i.kind for i in res.issues}


def test_inner_dim_mismatch_flagged():
    specs = [TPLinearSpec("fc1", 8, 16, TPKind.COLUMN),
             TPLinearSpec("fc2", 12, 8, TPKind.ROW, input_is_parallel=True)]
    res = verify_tensor_parallel(specs, 2)
    assert TPIssueKind.INNER_DIM_MISMATCH in {i.kind for i in res.issues}


def test_comm_flag_mismatch_flagged():
    # column gathers but row expects parallel input -> mismatch
    res = verify_tensor_parallel(
        megatron_mlp(8, 16, gather_output=True, input_is_parallel=True), 2)
    assert TPIssueKind.COMM_FLAG_MISMATCH in {i.kind for i in res.issues}
    # the other inconsistent combo: column does not gather, row not parallel
    res2 = verify_tensor_parallel(
        megatron_mlp(8, 16, gather_output=False, input_is_parallel=False), 2)
    assert TPIssueKind.COMM_FLAG_MISMATCH in {i.kind for i in res2.issues}


def test_correct_flag_combos_have_no_comm_issue():
    # gather + not-parallel is also valid (extra communication but consistent)
    res = verify_tensor_parallel(
        megatron_mlp(8, 16, gather_output=True, input_is_parallel=False), 2)
    assert TPIssueKind.COMM_FLAG_MISMATCH not in {i.kind for i in res.issues}


def test_tp_size_one_never_ragged():
    assert verify_tensor_parallel(megatron_mlp(8, 15), 1).ok is True


def test_tp_size_zero_rejected():
    try:
        verify_tensor_parallel(megatron_mlp(8, 16), 0)
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_hf_gqa_attention_allows_independent_head_dim():
    spec = llama_gqa_attention(
        hidden_size=32,
        num_attention_heads=8,
        num_key_value_heads=2,
        head_dim=40,
        sequence_parallel=True,
    )
    res = verify_tensor_parallel_attention(spec, 4)
    assert res.ok, [i.message for i in res.issues]


def test_megatron_mqa_replicated_kv_heads_is_valid():
    spec = megatron_attention(
        hidden_size=16,
        num_attention_heads=8,
        num_key_value_heads=1,
        tp_size=4,
        head_dim=2,
        kv_sharding=TPKVSharding.REPLICATE,
        sequence_parallel=True,
    )
    res = verify_tensor_parallel_attention(spec, 4)
    assert res.ok, [i.message for i in res.issues]


def test_attention_invalid_kv_partition_and_grouping_are_flagged():
    bad_partition = verify_tensor_parallel_attention(
        llama_gqa_attention(24, 12, 3, head_dim=2),
        4,
    )
    assert TPIssueKind.KV_HEAD_TP_INCOMPATIBLE in {
        i.kind for i in bad_partition.issues
    }

    bad_grouping = verify_tensor_parallel_attention(
        llama_gqa_attention(24, 10, 3, head_dim=2),
        2,
    )
    kinds = {i.kind for i in bad_grouping.issues}
    assert TPIssueKind.GQA_GROUP_MISMATCH in kinds


def test_attention_projection_shape_mismatch_is_flagged():
    spec = TPAttentionSpec(
        name="bad_projection",
        hidden_size=32,
        num_attention_heads=8,
        num_key_value_heads=2,
        head_dim=4,
        q_proj_shape=(31, 32),
        k_proj_shape=(8, 32),
        v_proj_shape=(8, 32),
        o_proj_shape=(32, 32),
    )
    res = verify_tensor_parallel_attention(spec, 4)
    assert TPIssueKind.PROJECTION_SHAPE_MISMATCH in {i.kind for i in res.issues}


def test_sequence_parallel_layernorm_must_not_shard_hidden_axis():
    spec = TPAttentionSpec(
        name="bad_sp_layernorm",
        hidden_size=16,
        num_attention_heads=8,
        num_key_value_heads=1,
        head_dim=2,
        sequence_parallel=True,
        activation_shape=(2, 5, 16),
        sequence_parallel_axis=-1,
        layer_norm_shape=(16,),
    )
    res = verify_tensor_parallel_attention(spec, 4)
    assert TPIssueKind.SEQUENCE_PARALLEL_AXIS in {i.kind for i in res.issues}


def test_real_huggingface_llama_attention_projection_shapes_when_available():
    pytest.importorskip("transformers")
    from transformers import LlamaConfig
    from transformers.models.llama.modeling_llama import LlamaAttention

    cfg = LlamaConfig(
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=8,
        num_key_value_heads=2,
        max_position_embeddings=16,
        vocab_size=100,
        head_dim=40,
    )
    attn = LlamaAttention(cfg, layer_idx=0)
    spec = TPAttentionSpec(
        name="real_hf_llama_attention",
        hidden_size=cfg.hidden_size,
        num_attention_heads=cfg.num_attention_heads,
        num_key_value_heads=cfg.num_key_value_heads,
        head_dim=cfg.head_dim,
        q_proj_shape=tuple(attn.q_proj.weight.shape),
        k_proj_shape=tuple(attn.k_proj.weight.shape),
        v_proj_shape=tuple(attn.v_proj.weight.shape),
        o_proj_shape=tuple(attn.o_proj.weight.shape),
    )
    res = verify_tensor_parallel_attention(spec, 4)
    assert res.ok, [i.message for i in res.issues]
    assert tuple(attn.q_proj.weight.shape) == (320, 32)
    assert tuple(attn.o_proj.weight.shape) == (32, 320)


# --- reproducibility harness: static verdict vs real sharded torch ---------
def test_static_matches_sharded_runtime():
    data = tps.measure()
    assert data["all_ok"] is True
    for c in data["cases"]:
        assert c["static_match"], f"{c['name']} static mismatch"
        assert c["live_match"], f"{c['name']} runtime mismatch"


def test_correct_megatron_reproduces_reference_at_runtime():
    sim = tps._simulate_megatron(8, 16, 2, gather_output=False,
                                 input_is_parallel=True)
    assert sim["ran"] and sim["matches_reference"]
    sim4 = tps._simulate_megatron(8, 16, 4, gather_output=False,
                                  input_is_parallel=True)
    assert sim4["ran"] and sim4["matches_reference"]


def test_artifact_is_byte_deterministic():
    assert tps.run(check=True) == 0


def test_artifact_has_no_volatile_fields():
    data = json.loads(tps.OUT_JSON.read_text())
    for key in _walk_keys(data):
        low = key.lower()
        for tok in VOLATILE_TOKENS:
            assert tok not in low, f"volatile key token {tok!r} in {key!r}"
