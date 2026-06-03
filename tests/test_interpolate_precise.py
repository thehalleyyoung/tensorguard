"""Step 196: precise F.interpolate / nn.Upsample output-size rules."""

from __future__ import annotations

import textwrap

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.fx_extractor import fx_trace_to_graph, verify_module
from src.model_checker import (
    ConstraintVerifier,
    LayerKind,
    extract_computation_graph,
    verify_model,
)


def _dims(shape):
    return tuple(d.value for d in shape.dims)


def _messages(result):
    if result.counterexample is None:
        return []
    return [v.message for v in result.counterexample.violations]


def _state_for_module(module: nn.Module, input_shapes):
    graph = fx_trace_to_graph(torch.fx.symbolic_trace(module))
    verifier = ConstraintVerifier(graph, input_shapes=input_shapes)
    violations, states, _ = verifier._bmc_base_case()
    return violations, states[-1], graph


def _output_dims(state, graph):
    return _dims(state.shape_env[graph.output_names[-1]])


class _FunctionalSize(nn.Module):
    def forward(self, x):
        return F.interpolate(
            x,
            size=(8, 4),
            mode="bilinear",
            align_corners=False,
        )


def test_functional_interpolate_size_source_and_fx_match_torch():
    x = torch.randn(2, 3, 5, 7)
    real = _FunctionalSize()(x)
    assert tuple(real.shape) == (2, 3, 8, 4)

    violations, state, graph = _state_for_module(
        _FunctionalSize(),
        {"x": tuple(x.shape)},
    )
    assert violations == []
    assert _output_dims(state, graph) == tuple(real.shape)

    source = """
        import torch
        import torch.nn as nn
        import torch.nn.functional as F

        class M(nn.Module):
            def forward(self, x):
                return F.interpolate(x, size=(8, 4), mode="bilinear", align_corners=False)
    """
    source = textwrap.dedent(source)
    result = verify_model(source, input_shapes={"x": tuple(x.shape)})
    assert result.safe

    source_graph = extract_computation_graph(source)
    verifier = ConstraintVerifier(source_graph, input_shapes={"x": tuple(x.shape)})
    source_violations, source_states, _ = verifier._bmc_base_case()
    assert source_violations == []
    assert _output_dims(source_states[-1], source_graph) == tuple(real.shape)


@pytest.mark.parametrize(
    ("shape", "scale_factor", "mode"),
    [
        ((2, 3, 11), 0.5, "linear"),
        ((2, 3, 5, 7), (2.5, 0.5), "bilinear"),
        ((1, 2, 4, 5, 6), (1.5, 0.5, 2.0), "trilinear"),
    ],
)
def test_fractional_and_tuple_scale_factors_match_torch(shape, scale_factor, mode):
    class M(nn.Module):
        def forward(self, x):
            return F.interpolate(
                x,
                scale_factor=scale_factor,
                mode=mode,
                align_corners=False,
            )

    x = torch.randn(*shape)
    real = M()(x)
    violations, state, graph = _state_for_module(M(), {"x": shape})
    assert violations == []
    assert _output_dims(state, graph) == tuple(real.shape)


class _UpsampleThenLinear(nn.Module):
    def __init__(self, in_features):
        super().__init__()
        self.up = nn.Upsample(scale_factor=(2.0, 3.0), mode="nearest")
        self.proj = nn.Linear(in_features, 4)

    def forward(self, x):
        return self.proj(self.up(x))


def test_nn_upsample_tuple_scale_flows_into_downstream_linear():
    x = torch.randn(2, 3, 4, 5)
    real = _UpsampleThenLinear(15)(x)
    assert tuple(real.shape) == (2, 3, 8, 4)

    good = verify_module(
        _UpsampleThenLinear(15),
        input_shapes={"x": tuple(x.shape)},
        backend="fx",
    )
    assert good.safe

    bad = verify_module(
        _UpsampleThenLinear(10),
        input_shapes={"x": tuple(x.shape)},
        backend="fx",
    )
    assert not bad.safe
    assert any("Linear" in message for message in _messages(bad))

    graph = fx_trace_to_graph(torch.fx.symbolic_trace(_UpsampleThenLinear(15)))
    assert graph.layers["up"].kind == LayerKind.UPSAMPLE


def test_recompute_scale_factor_does_not_change_output_size():
    class M(nn.Module):
        def forward(self, x):
            return F.interpolate(
                x,
                scale_factor=0.7,
                mode="bilinear",
                align_corners=False,
                recompute_scale_factor=True,
            )

    x = torch.randn(2, 3, 10, 13)
    real = M()(x)
    assert tuple(real.shape) == (2, 3, 7, 9)
    violations, state, graph = _state_for_module(M(), {"x": tuple(x.shape)})
    assert violations == []
    assert _output_dims(state, graph) == tuple(real.shape)


class _DynamicSize(nn.Module):
    def forward(self, x):
        return F.interpolate(
            x,
            size=x.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )


def test_dynamic_size_abstains_without_false_positive():
    x = torch.randn(2, 3, 5, 7)
    real = _DynamicSize()(x)
    assert tuple(real.shape) == tuple(x.shape)

    result = verify_module(
        _DynamicSize(),
        input_shapes={"x": tuple(x.shape)},
        backend="fx",
    )
    assert result.safe

    source = """
        import torch
        import torch.nn as nn
        import torch.nn.functional as F

        class M(nn.Module):
            def forward(self, x):
                return F.interpolate(x, size=x.shape[-2:], mode="bilinear", align_corners=False)
    """
    assert verify_model(textwrap.dedent(source), input_shapes={"x": tuple(x.shape)}).safe


@pytest.mark.parametrize(
    ("module", "shape", "torch_error", "tg_message"),
    [
        (
            lambda: type(
                "BothSizeAndScale",
                (nn.Module,),
                {
                    "forward": lambda self, x: F.interpolate(
                        x,
                        size=(8, 8),
                        scale_factor=2.0,
                        mode="nearest",
                    )
                },
            )(),
            (1, 3, 4, 4),
            "only one of size or scale_factor",
            "both size and scale_factor",
        ),
        (
            lambda: type(
                "NoSizeOrScale",
                (nn.Module,),
                {"forward": lambda self, x: F.interpolate(x)},
            )(),
            (1, 3, 4, 4),
            "either size or scale_factor",
            "requires either size or scale_factor",
        ),
        (
            lambda: type(
                "NearestAlignCorners",
                (nn.Module,),
                {
                    "forward": lambda self, x: F.interpolate(
                        x,
                        size=(8, 8),
                        mode="nearest",
                        align_corners=False,
                    )
                },
            )(),
            (1, 3, 4, 4),
            "align_corners option can only be set",
            "align_corners option can only be set",
        ),
        (
            lambda: type(
                "BadModeRank",
                (nn.Module,),
                {
                    "forward": lambda self, x: F.interpolate(
                        x,
                        size=(8, 8),
                        mode="linear",
                    )
                },
            )(),
            (1, 3, 4, 4),
            "Got 4D input, but linear mode needs 3D input",
            "mode 'linear' expects 1 spatial dims",
        ),
        (
            lambda: type(
                "BadAntialias",
                (nn.Module,),
                {
                    "forward": lambda self, x: F.interpolate(
                        x,
                        size=(8, 8),
                        mode="nearest",
                        antialias=True,
                    )
                },
            )(),
            (1, 3, 4, 4),
            "Anti-alias option is restricted",
            "antialias=True is only supported",
        ),
    ],
)
def test_invalid_literal_interpolate_contracts_match_torch_errors(
    module,
    shape,
    torch_error,
    tg_message,
):
    m = module()
    with pytest.raises((RuntimeError, ValueError)) as exc:
        m(torch.randn(*shape))
    assert torch_error in str(exc.value)

    result = verify_module(m, input_shapes={"x": shape}, backend="fx")
    assert not result.safe
    assert any(tg_message in message for message in _messages(result))
