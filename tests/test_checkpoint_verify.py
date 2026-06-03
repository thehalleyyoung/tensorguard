"""Step 220 — model checkpoint schema verification."""

from __future__ import annotations

from collections import OrderedDict

import pytest
import torch
import torch.nn as nn

from src.checkpoint_verify import (
    TensorGuardCheckpointError,
    TensorParallelCheckpointShard,
    guarded_load_state_dict,
    verify_checkpoint_state_dict,
)


class TinyLinear(nn.Module):
    def __init__(self, in_features: int = 3, out_features: int = 2, *, bias: bool = True, dtype=torch.float32):
        super().__init__()
        self.fc = nn.Linear(in_features, out_features, bias=bias, dtype=dtype)

    def forward(self, x):
        return self.fc(x)


class TiedLM(nn.Module):
    def __init__(self, vocab: int = 5, hidden: int = 3):
        super().__init__()
        self.embed = nn.Embedding(vocab, hidden)
        self.head = nn.Linear(hidden, vocab, bias=False)
        self.head.weight = self.embed.weight

    def forward(self, tokens):
        return self.head(self.embed(tokens))


def _clone_state(model: nn.Module):
    return OrderedDict((name, tensor.detach().clone()) for name, tensor in model.state_dict().items())


def test_clean_checkpoint_passes_envelope_and_guarded_loads_real_model():
    source = TinyLinear()
    checkpoint = {"state_dict": _clone_state(source)}
    target = TinyLinear()
    before = target.fc.weight.detach().clone()

    result = guarded_load_state_dict(target, checkpoint)

    assert result.ok
    assert "fc.weight" in result.checked_keys
    assert torch.equal(target.fc.weight, checkpoint["state_dict"]["fc.weight"])
    assert not torch.equal(before, target.fc.weight)


def test_missing_unexpected_and_shape_mismatch_are_reported_before_mutation():
    target = TinyLinear(in_features=3, out_features=2)
    checkpoint = _clone_state(target)
    checkpoint.pop("fc.bias")
    checkpoint["stale.weight"] = torch.zeros(1)
    checkpoint["fc.weight"] = torch.zeros(4, 3)
    before = _clone_state(target)

    with pytest.raises(TensorGuardCheckpointError) as exc:
        guarded_load_state_dict(target, checkpoint)

    categories = {issue.category for issue in exc.value.issues}
    assert {"missing_key", "unexpected_key", "shape_mismatch"} <= categories
    assert torch.equal(target.fc.weight, before["fc.weight"])
    with pytest.raises(RuntimeError, match="size mismatch"):
        target.load_state_dict(checkpoint, strict=False)


def test_dtype_mismatch_is_actionable_even_though_pytorch_silently_casts():
    source = TinyLinear(dtype=torch.float64)
    checkpoint = _clone_state(source)
    target = TinyLinear(dtype=torch.float32)

    result = verify_checkpoint_state_dict(target, checkpoint, strict=True)

    assert not result.ok
    issue = next(
        issue
        for issue in result.issues
        if issue.category == "dtype_mismatch" and issue.key == "fc.weight"
    )
    assert issue.key == "fc.weight"
    assert issue.actual_dtype == "torch.float64"
    assert issue.expected_dtype == "torch.float32"

    raw_target = TinyLinear(dtype=torch.float32)
    raw_target.load_state_dict(checkpoint)
    assert raw_target.fc.weight.dtype == torch.float32

    permissive = verify_checkpoint_state_dict(target, checkpoint, allow_dtype_cast=True)
    assert permissive.ok
    assert any(w.category == "dtype_mismatch" for w in permissive.warnings)


def test_tied_weight_value_mismatch_blocks_pytorch_last_alias_overwrite():
    model = TiedLM()
    checkpoint = _clone_state(model)
    checkpoint["embed.weight"] = torch.ones_like(checkpoint["embed.weight"])
    checkpoint["head.weight"] = torch.zeros_like(checkpoint["head.weight"])

    result = verify_checkpoint_state_dict(model, checkpoint)

    assert not result.ok
    assert any(issue.category == "tied_weight_value_mismatch" for issue in result.issues)

    raw_model = TiedLM()
    raw_model.load_state_dict(checkpoint)
    assert torch.equal(raw_model.embed.weight, torch.zeros_like(raw_model.embed.weight))


def test_tied_weight_equal_aliases_pass():
    model = TiedLM()
    checkpoint = _clone_state(model)
    shared = torch.full_like(checkpoint["embed.weight"], 0.25)
    checkpoint["embed.weight"] = shared.clone()
    checkpoint["head.weight"] = shared.clone()

    result = verify_checkpoint_state_dict(model, checkpoint)

    assert result.ok


def test_adapter_only_lora_checkpoint_is_verified_without_spurious_missing_base_keys():
    model = TinyLinear(in_features=4, out_features=6, bias=False)
    adapter = OrderedDict(
        {
            "fc.lora_A.weight": torch.randn(2, 4),
            "fc.lora_B.weight": torch.randn(6, 2),
        }
    )

    result = verify_checkpoint_state_dict(model, adapter)

    assert result.ok
    assert result.missing_keys == ()
    assert result.unexpected_keys == ()
    assert "fc.lora_A.weight" in result.checked_keys


def test_lora_adapter_shape_dtype_and_target_mismatches_are_reported():
    model = TinyLinear(in_features=4, out_features=6, bias=False)
    adapter = OrderedDict(
        {
            "fc.lora_A.weight": torch.randn(3, 5),
            "fc.lora_B.weight": torch.randn(7, 2, dtype=torch.float64),
            "missing.lora_A.weight": torch.randn(2, 4),
            "missing.lora_B.weight": torch.randn(6, 2),
            "fc2.lora_A.weight": torch.randn(2, 4),
        }
    )

    result = verify_checkpoint_state_dict(model, adapter)
    categories = {issue.category for issue in result.issues}

    assert {
        "lora_rank_mismatch",
        "lora_input_mismatch",
        "lora_output_mismatch",
        "lora_dtype_mismatch",
        "lora_target_missing",
        "lora_pair_incomplete",
    } <= categories


def test_complete_tensor_parallel_shards_suppress_missing_full_param_and_unexpected_keys():
    model = TinyLinear(in_features=4, out_features=6, bias=False)
    checkpoint = OrderedDict(
        {
            "fc.weight.tp0": torch.randn(3, 4),
            "fc.weight.tp1": torch.randn(3, 4),
        }
    )
    shards = (
        TensorParallelCheckpointShard("fc.weight", "fc.weight.tp0", dim=0, start=0, length=3, rank=0),
        TensorParallelCheckpointShard("fc.weight", "fc.weight.tp1", dim=0, start=3, length=3, rank=1),
    )

    result = verify_checkpoint_state_dict(model, checkpoint, tensor_parallel_shards=shards)

    assert result.ok
    assert result.missing_keys == ()
    assert result.unexpected_keys == ()
    assert set(result.checked_keys) == {"fc.weight.tp0", "fc.weight.tp1"}


def test_tensor_parallel_shard_shape_dtype_gap_and_overlap_are_reported():
    model = TinyLinear(in_features=4, out_features=6, bias=False)
    checkpoint = OrderedDict(
        {
            "fc.weight.tp0": torch.randn(2, 4),
            "fc.weight.tp1": torch.randn(4, 4, dtype=torch.float64),
            "fc.weight.tp2": torch.randn(2, 4),
        }
    )
    shards = (
        TensorParallelCheckpointShard("fc.weight", "fc.weight.tp0", dim=0, start=0, length=2, rank=0),
        TensorParallelCheckpointShard("fc.weight", "fc.weight.tp1", dim=0, start=3, length=3, rank=1),
        TensorParallelCheckpointShard("fc.weight", "fc.weight.tp2", dim=0, start=4, length=2, rank=2),
    )

    result = verify_checkpoint_state_dict(model, checkpoint, tensor_parallel_shards=shards)
    categories = {issue.category for issue in result.issues}

    assert "tp_shard_shape_mismatch" in categories
    assert "tp_shard_dtype_mismatch" in categories
    assert "tp_shard_gap" in categories
    assert "tp_shard_overlap" in categories


def test_guarded_load_warn_policy_does_not_mutate_on_checkpoint_violation():
    source = TinyLinear(in_features=5, out_features=2)
    target = TinyLinear(in_features=3, out_features=2)
    checkpoint = _clone_state(source)
    before = _clone_state(target)

    with pytest.warns(RuntimeWarning, match="rejected checkpoint"):
        result = guarded_load_state_dict(target, checkpoint, on_violation="warn")

    assert not result.ok
    assert torch.equal(target.fc.weight, before["fc.weight"])


def test_public_exports_checkpoint_gate():
    import tensorguard
    from tensorguard.torch import (
        TensorGuardCheckpointError as PublicError,
        TensorParallelCheckpointShard as PublicShard,
        guarded_load_state_dict as public_guarded,
        verify_checkpoint_state_dict as public_verify,
    )

    assert public_verify is verify_checkpoint_state_dict
    assert public_guarded is guarded_load_state_dict
    assert PublicError is TensorGuardCheckpointError
    assert PublicShard is TensorParallelCheckpointShard
    assert tensorguard.verify_checkpoint_state_dict is verify_checkpoint_state_dict
    assert tensorguard.guarded_load_state_dict is guarded_load_state_dict
