"""Step 222 — model-serving request/response schema gates."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from src.serving_schema import (
    ServingTensorSpec,
    TensorGuardServingSchemaError,
    guarded_fastapi_endpoint,
    guarded_model_serving_call,
    guarded_torchserve_handler,
    verify_serving_schema,
)


class CountingLinear(nn.Module):
    def __init__(self, in_features: int = 4, out_features: int = 2):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.calls = 0

    def forward(self, x):
        self.calls += 1
        return self.linear(x)


class CountingConv(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 2, kernel_size=3)
        self.calls = 0

    def forward(self, x):
        self.calls += 1
        return self.conv(x)


def test_verify_serving_schema_is_pure_over_materialized_boundaries():
    result = verify_serving_schema(
        request={"instances": [[1.0, 2.0, 3.0, 4.0]]},
        request_specs=(ServingTensorSpec("instances", shape=("B", 4)),),
        inputs=torch.zeros(1, 4, dtype=torch.float32),
        input_specs=(ServingTensorSpec("$", shape=("B", 4), dtype="float32"),),
        outputs=torch.zeros(1, 2, dtype=torch.float32),
        output_specs=(ServingTensorSpec("$", shape=("B", 2), dtype="float32"),),
        response={"scores": torch.zeros(1, 2, dtype=torch.float32)},
        response_specs=(ServingTensorSpec("scores", shape=("B", 2), dtype="float32"),),
        bind_shared_symbols=True,
        framework="fastapi",
    )

    assert result.ok
    assert not result.model_invoked
    assert {
        "request.instances",
        "input.$",
        "output.$",
        "response.scores",
    } <= set(result.checked_paths)


def test_fastapi_pipeline_validates_request_preprocess_model_and_response():
    model = CountingLinear()

    def preprocess(request):
        return torch.tensor(request["instances"], dtype=torch.float32)

    def postprocess(logits):
        return {"scores": logits}

    result = guarded_fastapi_endpoint(
        preprocess,
        model,
        {"instances": [[0.0, 1.0, 2.0, 3.0], [4.0, 5.0, 6.0, 7.0]]},
        request_specs=(ServingTensorSpec("instances", shape=("B", 4)),),
        input_specs=(ServingTensorSpec("$", shape=("B", 4), dtype="torch.float32"),),
        output_specs=(ServingTensorSpec("$", shape=("B", 2), dtype="float32"),),
        response_specs=(ServingTensorSpec("scores", shape=("B", 2), dtype="float32"),),
        postprocess=postprocess,
        bind_shared_symbols=True,
    )

    assert result.ok
    assert result.framework == "fastapi"
    assert result.model_invoked
    assert result.postprocess_invoked
    assert model.calls == 1


def test_bad_preprocess_shape_is_rejected_before_real_model_invocation():
    model = CountingConv()

    def preprocess(_request):
        return torch.zeros(1, 8, 8, 3, dtype=torch.float32)  # NHWC, not NCHW

    with pytest.raises(TensorGuardServingSchemaError) as exc:
        guarded_fastapi_endpoint(
            preprocess,
            model,
            {"image": "opaque-bytes"},
            input_specs=(ServingTensorSpec("$", shape=(1, 3, 8, 8), dtype="float32"),),
        )

    result = exc.value.result
    assert not result.ok
    assert not result.model_invoked
    assert model.calls == 0
    assert any(issue.category == "shape_mismatch" for issue in result.issues)

    with pytest.raises(RuntimeError):
        nn.Conv2d(3, 2, kernel_size=3)(preprocess({}))


def test_missing_request_field_stops_before_preprocess_and_model():
    model = CountingLinear()
    preprocess_calls = 0

    def preprocess(request):
        nonlocal preprocess_calls
        preprocess_calls += 1
        return torch.tensor(request["instances"], dtype=torch.float32)

    with pytest.raises(TensorGuardServingSchemaError) as exc:
        guarded_fastapi_endpoint(
            preprocess,
            model,
            {"payload": [[1.0, 2.0, 3.0, 4.0]]},
            request_specs=(ServingTensorSpec("instances", shape=("B", 4)),),
            input_specs=(ServingTensorSpec("$", shape=("B", 4), dtype="float32"),),
        )

    assert not exc.value.result.model_invoked
    assert preprocess_calls == 0
    assert model.calls == 0
    assert exc.value.issues[0].category == "missing_field"


def test_output_and_response_schema_mismatches_are_actionable():
    model = CountingLinear(out_features=3)

    with pytest.raises(TensorGuardServingSchemaError) as output_exc:
        guarded_model_serving_call(
            model,
            torch.zeros(1, 4),
            input_specs=(ServingTensorSpec("$", shape=("B", 4), dtype="float32"),),
            output_specs=(ServingTensorSpec("$", shape=("B", 2), dtype="float32"),),
        )

    assert output_exc.value.result.model_invoked
    assert any(issue.category == "shape_mismatch" for issue in output_exc.value.issues)
    assert model.calls == 1

    model = CountingLinear(out_features=2)

    def bad_response(logits):
        return {"scores": logits.squeeze(0)}

    with pytest.raises(TensorGuardServingSchemaError) as response_exc:
        guarded_model_serving_call(
            model,
            torch.zeros(1, 4),
            input_specs=(ServingTensorSpec("$", shape=("B", 4), dtype="float32"),),
            output_specs=(ServingTensorSpec("$", shape=("B", 2), dtype="float32"),),
            response_specs=(ServingTensorSpec("scores", shape=("B", 2), dtype="float32"),),
            postprocess=bad_response,
            bind_shared_symbols=True,
        )

    assert response_exc.value.result.model_invoked
    assert response_exc.value.result.postprocess_invoked
    assert any(issue.category == "rank_mismatch" for issue in response_exc.value.issues)
    assert model.calls == 1


def test_torchserve_style_handler_preprocess_inference_postprocess_is_guarded():
    class Handler:
        def __init__(self):
            self.model = CountingLinear()

        def preprocess(self, requests):
            rows = []
            for item in requests:
                rows.extend(item.get("body", item.get("data")))
            return torch.tensor(rows, dtype=torch.float32)

        def inference(self, batch):
            return self.model(batch)

        def postprocess(self, outputs):
            return [{"scores": row} for row in outputs]

    handler = Handler()
    result = guarded_torchserve_handler(
        handler,
        [{"body": [[0.0, 1.0, 2.0, 3.0]]}, {"data": [[4.0, 5.0, 6.0, 7.0]]}],
        input_specs=(ServingTensorSpec("$", shape=("B", 4), dtype="float32"),),
        output_specs=(ServingTensorSpec("$", shape=("B", 2), dtype="float32"),),
        response_specs=(ServingTensorSpec("$", shape=("B",),),),
    )

    assert result.ok
    assert result.framework == "torchserve"
    assert result.model_invoked
    assert result.postprocess_invoked
    assert handler.model.calls == 1


def test_public_exports_serving_schema_gate():
    import tensorguard
    from tensorguard.torch import (
        ServingTensorSpec as PublicSpec,
        TensorGuardServingSchemaError as PublicError,
        guarded_fastapi_endpoint as public_fastapi,
        guarded_model_serving_call as public_guarded,
        guarded_torchserve_handler as public_torchserve,
        verify_serving_schema as public_verify,
    )

    assert PublicSpec is ServingTensorSpec
    assert PublicError is TensorGuardServingSchemaError
    assert public_verify is verify_serving_schema
    assert public_guarded is guarded_model_serving_call
    assert public_fastapi is guarded_fastapi_endpoint
    assert public_torchserve is guarded_torchserve_handler
    assert tensorguard.verify_serving_schema is verify_serving_schema
    assert tensorguard.guarded_model_serving_call is guarded_model_serving_call
