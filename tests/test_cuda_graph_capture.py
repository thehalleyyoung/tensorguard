"""Step 223 — CUDA graph capture eligibility diagnostics."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from src.cuda_graph_capture import (
    TensorGuardCudaGraphCaptureError,
    guarded_cuda_graph_capture,
    verify_cuda_graph_capture_eligibility,
)


class StaticLinearNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(4, 3)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.fc(x))


class HelperAllocatingNet(nn.Module):
    def forward(self, x):
        return x + self._scratch(x)

    def _scratch(self, x):
        return torch.zeros(x.shape[0], x.shape[1], device=x.device)


class NonzeroNet(nn.Module):
    def forward(self, x):
        return torch.nonzero(x > 0)


class BooleanIndexNet(nn.Module):
    def forward(self, x):
        return x[x > 0]


class ItemSyncNet(nn.Module):
    def forward(self, x):
        scale = x.sum().item()
        return x * scale


class IdentityNet(nn.Module):
    def forward(self, x):
        return x + 1


def _categories(result):
    return {issue.category for issue in result.issues}


def test_static_linear_model_is_eligible_and_records_input_signature():
    model = StaticLinearNet()
    x = torch.randn(2, 4)
    assert model(x).shape == (2, 3)

    result = verify_cuda_graph_capture_eligibility(model, (x,), replay_args=(x,))

    assert result.ok
    assert result.source_available
    assert result.fx_trace_available
    assert "source" in result.verification_scope
    assert "fx" in result.verification_scope
    assert result.input_signatures[0].name == "arg0"
    assert result.input_signatures[0].shape == (2, 4)
    assert result.input_signatures[0].stride == tuple(x.stride())
    assert not result.issues


def test_dynamic_allocation_in_reachable_helper_is_flagged_against_real_forward():
    model = HelperAllocatingNet()
    x = torch.randn(3, 5)
    assert model(x).shape == (3, 5)

    result = verify_cuda_graph_capture_eligibility(model, (x,))

    assert not result.ok
    assert "dynamic_allocation" in _categories(result)
    assert any(issue.op_name == "torch.zeros" for issue in result.issues)


def test_data_dependent_shape_ops_are_flagged_against_real_forward():
    x = torch.tensor([[1.0, -1.0], [0.0, 2.0]])
    assert NonzeroNet()(x).shape == (2, 2)

    result = verify_cuda_graph_capture_eligibility(NonzeroNet(), (x,))

    assert not result.ok
    assert any(
        issue.category == "data_dependent_shape" and issue.op_name == "torch.nonzero"
        for issue in result.issues
    )


def test_boolean_mask_indexing_is_reported_as_data_dependent_shape():
    x = torch.tensor([[1.0, -1.0], [0.0, 2.0]])
    assert BooleanIndexNet()(x).shape == (2,)

    result = verify_cuda_graph_capture_eligibility(BooleanIndexNet(), (x,))

    assert not result.ok
    assert any(
        issue.category == "data_dependent_shape" and issue.op_name == "boolean_indexing"
        for issue in result.issues
    )


def test_item_host_sync_is_reported_as_unsupported_op():
    x = torch.randn(2, 3)
    assert ItemSyncNet()(x).shape == (2, 3)

    result = verify_cuda_graph_capture_eligibility(ItemSyncNet(), (x,))

    assert not result.ok
    assert any(issue.category == "unsupported_op" and issue.op_name.endswith(".item") for issue in result.issues)


def test_static_replay_shape_dtype_device_and_stride_are_checked():
    x = torch.randn(2, 4)
    wrong_shape = torch.randn(3, 4)
    result = verify_cuda_graph_capture_eligibility(IdentityNet(), (x,), replay_args=(wrong_shape,))
    assert not result.ok
    assert any("shape" in issue.message for issue in result.issues if issue.category == "static_input_mismatch")

    x_nc = torch.randn(3, 4).t()
    replay_contiguous = torch.randn(4, 3)
    stride_result = verify_cuda_graph_capture_eligibility(
        IdentityNet(),
        (x_nc,),
        replay_args=(replay_contiguous,),
    )
    assert not stride_result.ok
    assert any("stride" in issue.message for issue in stride_result.issues)

    matching = verify_cuda_graph_capture_eligibility(IdentityNet(), (x,), replay_args=(x,))
    assert "static_input_mismatch" not in _categories(matching)


def test_cuda_and_static_address_requirements_are_explicit_policy_checks():
    x = torch.randn(2, 4)
    result = verify_cuda_graph_capture_eligibility(
        IdentityNet(),
        (x,),
        require_cuda_inputs=True,
    )
    assert not result.ok
    assert any(issue.category == "input_device" for issue in result.issues)

    same_shape_new_storage = torch.randn(2, 4)
    address_result = verify_cuda_graph_capture_eligibility(
        IdentityNet(),
        (x,),
        replay_args=(same_shape_new_storage,),
        require_static_input_addresses=True,
    )
    assert any(issue.category == "static_input_address" for issue in address_result.issues)


def test_source_unavailable_is_an_actionable_issue_not_a_silent_pass():
    Dynamic = type("DynamicCaptureNet", (nn.Module,), {"forward": lambda self, x: x + 1})

    result = verify_cuda_graph_capture_eligibility(Dynamic(), (torch.randn(2, 4),), check_fx=False)

    assert not result.ok
    assert any(issue.category == "source_unavailable" for issue in result.issues)


def test_guarded_capture_blocks_invocation_on_eligibility_failure():
    called = {"value": False}

    def capture(*_args):
        called["value"] = True
        return "captured"

    with pytest.raises(TensorGuardCudaGraphCaptureError):
        guarded_cuda_graph_capture(
            HelperAllocatingNet(),
            (torch.randn(2, 4),),
            capture=capture,
        )
    assert not called["value"]

    out = guarded_cuda_graph_capture(StaticLinearNet(), (torch.randn(2, 4),), capture=capture)
    assert out == "captured"
    assert called["value"]


def test_public_exports_cuda_graph_capture_gate():
    import tensorguard
    from tensorguard.torch import (
        CudaGraphCaptureEligibilityResult,
        CudaGraphCaptureIssue,
        CudaGraphInputSignature,
        TensorGuardCudaGraphCaptureError as PublicError,
        guarded_cuda_graph_capture as public_guarded,
        verify_cuda_graph_capture_eligibility as public_verify,
    )

    result = verify_cuda_graph_capture_eligibility(StaticLinearNet(), (torch.randn(2, 4),))

    assert isinstance(result, CudaGraphCaptureEligibilityResult)
    assert CudaGraphCaptureIssue.__name__ == "CudaGraphCaptureIssue"
    assert CudaGraphInputSignature.__name__ == "CudaGraphInputSignature"
    assert PublicError is TensorGuardCudaGraphCaptureError
    assert public_verify is verify_cuda_graph_capture_eligibility
    assert public_guarded is guarded_cuda_graph_capture
    assert tensorguard.verify_cuda_graph_capture_eligibility is verify_cuda_graph_capture_eligibility
    assert tensorguard.guarded_cuda_graph_capture is guarded_cuda_graph_capture

