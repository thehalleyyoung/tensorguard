"""Step 219 — optimizer-state shape/dtype and resume gates."""

from __future__ import annotations

from collections import OrderedDict

import pytest
import torch
import torch.nn as nn

from src.optimizer_state_verify import (
    OptimizerStateShard,
    TensorGuardOptimizerStateError,
    guarded_optimizer_load_state_dict,
    verify_optimizer_state,
)


class TinyMLP(nn.Module):
    def __init__(self, in_features: int = 3, out_features: int = 2, *, dtype=torch.float32):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=False, dtype=dtype)

    def forward(self, x):
        return self.linear(x).square().mean()


class Rank3Parameter(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(2, 3, 4))


def _step_optimizer(model: nn.Module, optimizer: torch.optim.Optimizer) -> None:
    optimizer.zero_grad()
    param = next(model.parameters())
    if isinstance(model, TinyMLP):
        x = torch.randn(5, model.linear.in_features, dtype=param.dtype)
        loss = model(x)
    else:
        loss = sum(p.square().mean() for p in model.parameters())
    loss.backward()
    optimizer.step()


def _adamw_model_and_state(**optim_kwargs):
    model = TinyMLP()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2, **optim_kwargs)
    _step_optimizer(model, optimizer)
    return model, optimizer, optimizer.state_dict()


def _first_param_id(state_dict):
    return state_dict["param_groups"][0]["params"][0]


def test_real_adamw_state_passes_and_checks_moments():
    model, optimizer, _ = _adamw_model_and_state()

    result = verify_optimizer_state(model, optimizer)

    assert result.ok
    assert result.issues == ()
    assert "linear.weight:exp_avg" in result.checked_states
    assert "linear.weight:exp_avg_sq" in result.checked_states


def test_adamw_exp_avg_shape_mismatch_is_rejected():
    model, _, state = _adamw_model_and_state()
    pid = _first_param_id(state)
    state["state"][pid]["exp_avg"] = torch.zeros(1)

    result = verify_optimizer_state(model, state, optimizer_name="AdamW")

    assert not result.ok
    issue = next(i for i in result.issues if i.state_key == "exp_avg")
    assert issue.category == "state_shape_mismatch"
    assert issue.expected_shape == tuple(model.linear.weight.shape)
    assert issue.actual_shape == (1,)


def test_raw_state_dict_uses_param_names_when_available():
    model, _, state = _adamw_model_and_state()
    pid = _first_param_id(state)
    state["state"][99] = state["state"].pop(pid)
    state["param_groups"][0]["params"] = [99]
    state["param_groups"][0]["param_names"] = ["linear.weight"]

    result = verify_optimizer_state(model, state, optimizer_name="AdamW")

    assert result.ok
    assert "linear.weight:exp_avg" in result.checked_states


def test_adamw_bad_state_dtype_is_rejected_but_fp32_master_is_warning():
    model, _, state = _adamw_model_and_state()
    pid = _first_param_id(state)
    state["state"][pid]["exp_avg_sq"] = state["state"][pid]["exp_avg_sq"].double()

    bad = verify_optimizer_state(model, state, optimizer_name="AdamW")

    assert any(i.category == "state_dtype_mismatch" for i in bad.issues)

    half_model = TinyMLP(dtype=torch.float16)
    half_opt = torch.optim.AdamW(half_model.parameters(), lr=1e-2)
    _step_optimizer(half_model, half_opt)
    half_state = half_opt.state_dict()
    half_pid = _first_param_id(half_state)
    half_state["state"][half_pid]["exp_avg"] = half_state["state"][half_pid]["exp_avg"].float()

    master = verify_optimizer_state(half_model, half_state, optimizer_name="AdamW")

    assert master.ok
    assert any(w.category == "master_state_dtype" for w in master.warnings)


def test_real_adafactor_factored_state_passes_for_rank2_and_rank3():
    rank2 = TinyMLP(in_features=4, out_features=3)
    opt2 = torch.optim.Adafactor(rank2.parameters(), lr=1e-2)
    _step_optimizer(rank2, opt2)

    result2 = verify_optimizer_state(rank2, opt2)

    assert result2.ok
    assert "linear.weight:row_var" in result2.checked_states
    assert "linear.weight:col_var" in result2.checked_states

    rank3 = Rank3Parameter()
    opt3 = torch.optim.Adafactor(rank3.parameters(), lr=1e-2)
    _step_optimizer(rank3, opt3)

    result3 = verify_optimizer_state(rank3, opt3)

    assert result3.ok
    state = opt3.state_dict()["state"][_first_param_id(opt3.state_dict())]
    assert tuple(state["row_var"].shape) == (2, 3, 1)
    assert tuple(state["col_var"].shape) == (2, 1, 4)


def test_adafactor_row_var_shape_mismatch_is_rejected():
    model = TinyMLP(in_features=4, out_features=3)
    optimizer = torch.optim.Adafactor(model.parameters(), lr=1e-2)
    _step_optimizer(model, optimizer)
    state = optimizer.state_dict()
    pid = _first_param_id(state)
    state["state"][pid]["row_var"] = torch.zeros(1, 1)

    result = verify_optimizer_state(model, state, optimizer_name="Adafactor")

    assert not result.ok
    issue = next(i for i in result.issues if i.state_key == "row_var")
    assert issue.expected_shape == (3, 1)
    assert issue.actual_shape == (1, 1)


def test_fused_adamw_state_uses_same_shape_contract():
    model = TinyMLP()
    try:
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2, fused=True)
        _step_optimizer(model, optimizer)
    except RuntimeError as exc:
        pytest.skip(f"fused AdamW unavailable on this torch/device combination: {exc}")

    result = verify_optimizer_state(model, optimizer, optimizer_name="FusedAdamW")

    assert result.ok
    assert "linear.weight:exp_avg" in result.checked_states


def test_lazy_uninitialized_state_warns_or_fails_by_policy():
    model = TinyMLP()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)

    permissive = verify_optimizer_state(model, optimizer, allow_lazy=True)
    strict = verify_optimizer_state(model, optimizer, allow_lazy=False)

    assert permissive.ok
    assert any(w.category == "lazy_uninitialized" for w in permissive.warnings)
    assert not strict.ok
    assert any(i.category == "lazy_uninitialized" for i in strict.issues)


def test_zero_style_sharded_state_checks_coverage_shape_and_dtype():
    model, optimizer, _ = _adamw_model_and_state()
    valid = [
        OptimizerStateShard("linear.weight", "exp_avg", (1, 3), torch.float32, start=0, length=1, shard_index=0),
        OptimizerStateShard("linear.weight", "exp_avg", (1, 3), torch.float32, start=1, length=1, shard_index=1),
    ]

    ok = verify_optimizer_state(model, optimizer, sharded_state=valid)

    assert ok.ok
    assert "linear.weight:exp_avg:shard0" in ok.checked_states

    invalid = [
        OptimizerStateShard("linear.weight", "exp_avg", (1, 3), torch.float32, start=0, length=1, shard_index=0),
        OptimizerStateShard("linear.weight", "exp_avg", (2, 3), torch.float64, start=1, length=1, shard_index=1),
        OptimizerStateShard("linear.weight", "exp_avg", (1, 3), torch.float32, start=1, length=1, shard_index=2),
    ]

    bad = verify_optimizer_state(model, optimizer, sharded_state=invalid)
    categories = {issue.category for issue in bad.issues}

    assert "shard_shape_mismatch" in categories
    assert "state_dtype_mismatch" in categories
    assert "shard_overlap" in categories


def test_guarded_optimizer_load_rejects_shape_incompatible_resume_before_loading():
    source, _, checkpoint = _adamw_model_and_state()
    target = TinyMLP(in_features=4, out_features=2)
    target_optimizer = torch.optim.AdamW(target.parameters(), lr=1e-2)
    before = OrderedDict((name, param.detach().clone()) for name, param in target.named_parameters())

    with pytest.raises(TensorGuardOptimizerStateError) as exc:
        guarded_optimizer_load_state_dict(target, target_optimizer, checkpoint)

    assert any(issue.category == "state_shape_mismatch" for issue in exc.value.issues)
    assert target_optimizer.state == {}
    assert all(torch.equal(before[name], param) for name, param in target.named_parameters())
    assert tuple(source.linear.weight.shape) != tuple(target.linear.weight.shape)


def test_guarded_optimizer_load_warn_policy_does_not_mutate_on_violation():
    _, _, checkpoint = _adamw_model_and_state()
    target = TinyMLP(in_features=4, out_features=2)
    target_optimizer = torch.optim.AdamW(target.parameters(), lr=1e-2)

    with pytest.warns(RuntimeWarning, match="rejected optimizer state"):
        result = guarded_optimizer_load_state_dict(
            target,
            target_optimizer,
            checkpoint,
            on_violation="warn",
        )

    assert not result.ok
    assert target_optimizer.state == {}


def test_guarded_optimizer_load_accepts_compatible_and_torch_castable_dtype_resume():
    source, _, checkpoint = _adamw_model_and_state()
    target = TinyMLP(dtype=torch.float16)
    target_optimizer = torch.optim.AdamW(target.parameters(), lr=1e-2)

    result = guarded_optimizer_load_state_dict(target, target_optimizer, checkpoint)

    assert result.ok
    loaded_state = target_optimizer.state[next(iter(target_optimizer.state))]
    assert loaded_state["exp_avg"].dtype == target.linear.weight.dtype
    assert tuple(source.linear.weight.shape) == tuple(target.linear.weight.shape)


def test_public_tensorguard_torch_exports_optimizer_gate():
    import tensorguard
    from tensorguard.torch import (
        OptimizerStateShard as PublicShard,
        TensorGuardOptimizerStateError as PublicError,
        guarded_optimizer_load_state_dict as public_guarded_load,
        verify_optimizer_state as public_verify,
    )

    assert public_verify is verify_optimizer_state
    assert public_guarded_load is guarded_optimizer_load_state_dict
    assert PublicError is TensorGuardOptimizerStateError
    assert PublicShard is OptimizerStateShard
    assert tensorguard.verify_optimizer_state is verify_optimizer_state
    assert tensorguard.guarded_optimizer_load_state_dict is guarded_optimizer_load_state_dict
