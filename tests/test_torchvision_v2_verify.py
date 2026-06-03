"""Step 205 — source-level torchvision.transforms.v2 tensor contracts."""

from __future__ import annotations

import textwrap

import pytest

from src.model_checker import ConstraintVerifier, extract_computation_graph
from src.torchvision_v2_verify import TransformVerdict, verify_torchvision_v2_transform

torch = pytest.importorskip("torch")
v2 = pytest.importorskip("torchvision.transforms.v2")


def _dims(shape):
    return tuple(d.value for d in shape.dims)


def _source_state(source: str, input_shapes):
    graph = extract_computation_graph(textwrap.dedent(source))
    verifier = ConstraintVerifier(graph, input_shapes=input_shapes)
    violations, states, _ = verifier._bmc_base_case()
    return violations, states[-1], graph


def _source_output_shape(source: str, input_shapes):
    violations, state, graph = _source_state(source, input_shapes)
    assert violations == []
    return _dims(state.shape_env[graph.output_names[-1]])


def _messages(violations):
    return [v.message for v in violations]


def test_pure_transform_contracts_cover_shape_changes_and_abstention():
    resize = verify_torchvision_v2_transform("Resize", (3, 20, 30), size=(12, 16))
    assert resize.ok
    assert resize.output_shape == (3, 12, 16)

    center_crop_2d = verify_torchvision_v2_transform(
        "CenterCrop", (20, 30), size=(10, 14)
    )
    assert center_crop_2d.ok
    assert center_crop_2d.output_shape == (10, 14)

    pad = verify_torchvision_v2_transform("Pad", (3, 20, 30), padding=(1, 2, 3, 4))
    assert pad.ok
    assert pad.output_shape == (3, 26, 34)

    normalizes_channel = verify_torchvision_v2_transform(
        "Normalize", (1, 8, 8), mean=[0.5, 0.5, 0.5], std=[0.2, 0.2, 0.2]
    )
    assert normalizes_channel.ok
    assert normalizes_channel.output_shape == (3, 8, 8)

    pil_only = verify_torchvision_v2_transform("PILToTensor", None)
    assert pil_only.ok
    assert pil_only.unknown_reason
    assert isinstance(pil_only, TransformVerdict)


def test_pure_contracts_refute_real_tensor_path_errors():
    resize_rank = verify_torchvision_v2_transform("Resize", (20, 30), size=(12, 16))
    assert not resize_rank.ok
    assert resize_rank.error_kind == "rank"

    random_crop_oversize = verify_torchvision_v2_transform(
        "RandomCrop", (3, 8, 8), size=(12, 12)
    )
    assert not random_crop_oversize.ok
    assert random_crop_oversize.error_kind == "shape"

    random_crop_padded = verify_torchvision_v2_transform(
        "RandomCrop", (3, 8, 8), size=(12, 12), pad_if_needed=True
    )
    assert random_crop_padded.ok
    assert random_crop_padded.output_shape == (3, 12, 12)

    bad_normalize = verify_torchvision_v2_transform(
        "Normalize", (3, 8, 8), mean=[0.5, 0.5], std=[0.2, 0.2]
    )
    assert not bad_normalize.ok
    assert bad_normalize.error_kind == "shape"


@pytest.mark.parametrize(
    "transform,shape,kwargs",
    [
        (v2.Resize((12, 16)), (3, 20, 30), {"size": (12, 16)}),
        (v2.CenterCrop((10, 14)), (20, 30), {"size": (10, 14)}),
        (v2.RandomCrop((10, 14)), (3, 20, 30), {"size": (10, 14)}),
        (v2.Pad((1, 2, 3, 4)), (3, 20, 30), {"padding": (1, 2, 3, 4)}),
        (v2.RandomHorizontalFlip(p=1.0), (1,), {}),
        (
            v2.Normalize([0.5, 0.5, 0.5], [0.2, 0.2, 0.2]),
            (1, 8, 8),
            {"mean": [0.5, 0.5, 0.5], "std": [0.2, 0.2, 0.2]},
        ),
    ],
)
def test_static_contract_shapes_match_real_torchvision_v2(transform, shape, kwargs):
    x = torch.rand(*shape)
    expected = tuple(transform(x).shape)
    verdict = verify_torchvision_v2_transform(
        type(transform).__name__, shape, **kwargs
    )
    assert verdict.ok
    assert verdict.output_shape == expected


def test_source_compose_matches_real_torchvision_v2_shape():
    source = """
        import torch
        import torch.nn as nn
        from torchvision.transforms import v2

        class M(nn.Module):
            def __init__(self):
                super().__init__()
                self.t = v2.Compose([
                    v2.Resize((12, 16)),
                    v2.Pad((1, 2, 3, 4)),
                    v2.Normalize([0.5, 0.5, 0.5], [0.2, 0.2, 0.2]),
                ])

            def forward(self, x):
                return self.t(x)
    """
    x = torch.rand(3, 20, 30)
    expected = tuple(
        v2.Compose([
            v2.Resize((12, 16)),
            v2.Pad((1, 2, 3, 4)),
            v2.Normalize([0.5, 0.5, 0.5], [0.2, 0.2, 0.2]),
        ])(x).shape
    )
    assert expected == (3, 18, 20)
    assert _source_output_shape(source, {"x": tuple(x.shape)}) == expected


def test_source_inline_transform_constructor_is_modelled():
    source = """
        import torch
        import torch.nn as nn
        from torchvision.transforms import v2

        class M(nn.Module):
            def forward(self, x):
                return v2.CenterCrop((8, 9))(x)
    """
    assert _source_output_shape(source, {"x": (3, 16, 20)}) == (3, 8, 9)


def test_source_transform_errors_match_real_torchvision_failures():
    resize_source = """
        import torch
        import torch.nn as nn
        from torchvision.transforms import v2

        class M(nn.Module):
            def __init__(self):
                super().__init__()
                self.t = v2.Resize((8, 8))

            def forward(self, x):
                return self.t(x)
    """
    resize_violations, _, _ = _source_state(resize_source, {"x": (20, 30)})
    assert any("rank >= 3" in msg for msg in _messages(resize_violations))
    with pytest.raises(ValueError):
        v2.Resize((8, 8))(torch.rand(20, 30))

    normalize_source = """
        import torch
        import torch.nn as nn
        from torchvision.transforms import v2

        class M(nn.Module):
            def __init__(self):
                super().__init__()
                self.t = v2.Normalize([0.5, 0.5], [0.2, 0.2])

            def forward(self, x):
                return self.t(x)
    """
    normalize_violations, _, _ = _source_state(normalize_source, {"x": (3, 8, 8)})
    assert any("Normalize channel dim 3" in msg for msg in _messages(normalize_violations))
    with pytest.raises(RuntimeError):
        v2.Normalize([0.5, 0.5], [0.2, 0.2])(torch.rand(3, 8, 8))


def test_source_pil_only_transform_abstains_instead_of_preserving_shape():
    source = """
        import torch
        import torch.nn as nn
        from torchvision.transforms import v2

        class M(nn.Module):
            def __init__(self):
                super().__init__()
                self.t = v2.PILToTensor()

            def forward(self, x):
                return self.t(x)
    """
    violations, state, graph = _source_state(source, {"x": (3, 8, 8)})
    assert violations == []
    out = _source_output_shape(source, {"x": (3, 8, 8)})
    assert len(out) == 3
    assert all(str(dim).startswith("_tv2_") for dim in out)
    assert graph.output_names[-1] in state.shape_env


def test_public_exports_torchvision_v2_verifier():
    import src
    import tensorguard

    assert src.verify_torchvision_v2_transform is verify_torchvision_v2_transform
    assert tensorguard.verify_torchvision_v2_transform is verify_torchvision_v2_transform
