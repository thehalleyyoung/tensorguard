"""Step 224: compare TensorGuard constraints with Dynamo guard sets."""

from __future__ import annotations

import pytest

try:
    import torch
    import torch.nn as nn

    HAS_TORCH = True
except ImportError:  # pragma: no cover
    HAS_TORCH = False

from src.compile_guard_analysis import (
    DynamoGuard,
    verify_compile_guard_interactions,
)


pytestmark = pytest.mark.skipif(not HAS_TORCH, reason="PyTorch required")


class LinearBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(10, 4)

    def forward(self, x):
        y = self.proj(x)
        return y.relu()


class ResidualBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(10, 4)

    def forward(self, x, residual):
        return self.proj(x) + residual


class ConvStem(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 8, kernel_size=3, padding=1)

    def forward(self, x):
        return self.conv(x)


class FakeGuard:
    def __init__(self, code):
        self.name = "shape_env"
        self.source = "GuardSource.SHAPE_ENV"
        self.create_fn = "SHAPE_ENV"
        self.code_list = code


def _linear_guards():
    # These strings mirror real Dynamo shape-env guard code emitted through
    # ExplainOutput.out_guards[*].code_list: size()/shape predicates, rank
    # predicates, lower-bound ranges, and unrelated tensor-match noise.
    return [
        "len(L['x'].size()) == 2",
        "L['x'].size()[1] == 10",
        "2 <= L['x'].size()[0]",
        "hasattr(L['x'], '_dynamo_dynamic_indices') == False",
    ]


def test_linear_input_and_layer_constraints_match_dynamo_shape_guards():
    result = verify_compile_guard_interactions(
        LinearBlock(),
        (torch.randn(2, 10),),
        input_shapes={"x": ("batch", 10)},
        dynamo_guards=_linear_guards(),
    )

    assert result.ok
    assert result.dynamo_error is None
    assert result.tensorguard_constraints
    assert any(m.constraint.kind == "linear_in_features" for m in result.matched_constraints)
    assert not result.missing_constraints


def test_missing_dynamo_shape_guard_is_reported():
    result = verify_compile_guard_interactions(
        LinearBlock(),
        (torch.randn(2, 10),),
        input_shapes={"x": ("batch", 10)},
        dynamo_guards=[
            "len(L['x'].size()) == 2",
            "L['x'].size()[0] >= 1",
        ],
    )

    assert not result.ok
    missing = [issue for issue in result.issues if issue.category == "missing_dynamo_guard"]
    assert missing
    assert any("x[1] == 10" in issue.message for issue in missing)


def test_repeated_symbolic_dim_equality_matches_cross_input_guard():
    result = verify_compile_guard_interactions(
        ResidualBlock(),
        (torch.randn(3, 10), torch.randn(3, 4)),
        input_shapes={"x": ("b", 10), "residual": ("b", 4)},
        dynamo_guards=[
            "len(L['x'].size()) == 2",
            "len(L['residual'].size()) == 2",
            "L['x'].size()[0] == L['residual'].size()[0]",
            "L['x'].size()[0] >= 1",
            "L['residual'].size()[0] >= 1",
            "L['x'].size()[1] == 10",
            "L['residual'].size()[1] == 4",
        ],
    )

    assert result.ok
    assert any(
        match.constraint.kind == "symbolic_dim_equality"
        for match in result.matched_constraints
    )


def test_conv_channel_constraint_matches_shape_guard_object():
    result = verify_compile_guard_interactions(
        ConvStem(),
        (torch.randn(2, 3, 16, 16),),
        input_shapes={"x": ("b", 3, 16, 16)},
        dynamo_guards=[
            FakeGuard(
                [
                    "L['x'].ndimension() == 4",
                    "L['x'].size()[0] >= 1",
                    "L['x'].size()[1] == 3",
                    "L['x'].size()[2] == 16",
                    "L['x'].size()[3] == 16",
                ]
            )
        ],
    )

    assert result.ok
    assert any(match.constraint.kind == "conv_in_channels" for match in result.matched_constraints)


def test_public_import_surfaces_are_wired():
    from tensorguard import verify_compile_guard_interactions as top_level
    from tensorguard.torch import verify_compile_guard_interactions as torch_level

    assert top_level is verify_compile_guard_interactions
    assert torch_level is verify_compile_guard_interactions
    assert DynamoGuard(code=("L['x'].size()[1] == 10",)).code


def test_unrecoverable_module_source_surfaces_abstention_not_typeerror():
    DynamicNet = type(
        "DynamicNet",
        (nn.Module,),
        {"forward": lambda self, x: x},
    )

    result = verify_compile_guard_interactions(
        DynamicNet(),
        (torch.randn(2, 10),),
        input_shapes={"x": ("b", 10)},
        dynamo_guards=[],
    )

    assert not result.ok
    assert result.tensorguard_error
    assert any(issue.category == "tensorguard_unavailable" for issue in result.issues)


def _dynamo_explain_supported() -> bool:
    try:
        import torch._dynamo as dynamo

        dynamo.eval_frame.check_if_dynamo_supported()
        dynamo.explain(LinearBlock())(torch.randn(2, 10))
        return True
    except Exception:
        return False


@pytest.mark.skipif(
    not HAS_TORCH or not _dynamo_explain_supported(),
    reason="torch._dynamo.explain unsupported on this interpreter",
)
def test_real_dynamo_explain_guard_collection_smoke():
    result = verify_compile_guard_interactions(
        LinearBlock(),
        (torch.randn(2, 10),),
        input_shapes={"x": ("b", 10)},
    )

    assert result.dynamo_error is None
    assert result.graph_count >= 1
    assert result.dynamo_guards
