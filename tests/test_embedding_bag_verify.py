"""Step 207 — exact EmbeddingBag / TorchRec jagged contracts."""

from __future__ import annotations

import pytest

from src.embedding_bag_verify import (
    EmbeddingBagVerdict,
    TorchRecJaggedSpec,
    verify_embedding_bag,
    verify_torchrec_embedding_bag,
)

torch = pytest.importorskip("torch")
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from src.fx_extractor import verify_module  # noqa: E402


def _messages(result, kind=None):
    if result.counterexample is None:
        return []
    return [
        v.message for v in result.counterexample.violations
        if kind is None or v.kind == kind
    ]


def test_embedding_bag_shapes_match_real_torch_1d_and_2d_inputs():
    weight = torch.randn(7, 4)
    indices_2d = torch.tensor([[0, 1, 2], [3, 4, 5]], dtype=torch.long)
    actual_2d = F.embedding_bag(indices_2d, weight, mode="mean")
    static_2d = verify_embedding_bag(
        indices_2d.shape,
        weight_shape=weight.shape,
        input_dtype=indices_2d.dtype,
        weight_dtype=weight.dtype,
        mode="mean",
    )
    assert static_2d.ok
    assert static_2d.output_shape == tuple(actual_2d.shape) == (2, 4)

    indices_1d = torch.tensor([0, 1, 2, 3], dtype=torch.int32)
    offsets = torch.tensor([0, 2, 4], dtype=torch.int64)
    actual_1d = F.embedding_bag(
        indices_1d, weight, offsets, mode="sum", include_last_offset=True
    )
    static_1d = verify_embedding_bag(
        indices_1d.shape,
        weight_shape=weight.shape,
        offsets_shape=offsets.shape,
        input_dtype=indices_1d.dtype,
        offsets_dtype=offsets.dtype,
        weight_dtype=weight.dtype,
        offsets_values=tuple(offsets.tolist()),
        mode="sum",
        include_last_offset=True,
    )
    assert static_1d.ok
    assert static_1d.output_shape == tuple(actual_1d.shape) == (2, 4)
    assert static_1d.output_dtype == "float32"


def test_offsets_contracts_match_real_pytorch_failures():
    weight_shape = (5, 3)
    no_offsets = verify_embedding_bag((3,), weight_shape=weight_shape)
    assert not no_offsets.ok
    assert no_offsets.error_kind == "offsets"
    with pytest.raises(ValueError, match="offsets"):
        F.embedding_bag(torch.tensor([0, 1, 2]), torch.randn(5, 3))

    bad_first = verify_embedding_bag(
        (3,),
        weight_shape=weight_shape,
        offsets_shape=(1,),
        offsets_values=(1,),
    )
    assert not bad_first.ok
    assert "offsets[0] must be 0" in bad_first.message
    with pytest.raises(RuntimeError, match="offsets\\[0\\]"):
        F.embedding_bag(torch.tensor([0, 1, 2]), torch.randn(5, 3), torch.tensor([1]))

    bad_last = verify_embedding_bag(
        (3,),
        weight_shape=weight_shape,
        offsets_shape=(2,),
        offsets_values=(0, 4),
    )
    assert not bad_last.ok
    assert "cannot exceed input length" in bad_last.message
    with pytest.raises(RuntimeError, match="greater than input"):
        F.embedding_bag(torch.tensor([0, 1, 2]), torch.randn(5, 3), torch.tensor([0, 4]))


def test_per_sample_weight_pooling_shape_and_dtype_contracts_match_torch():
    ok = verify_embedding_bag(
        (4,),
        weight_shape=(8, 2),
        offsets_shape=(1,),
        per_sample_weights_shape=(4,),
        weight_dtype="float64",
        per_sample_weights_dtype="float64",
        mode="sum",
    )
    assert ok.ok
    F.embedding_bag(
        torch.tensor([0, 1, 2, 3]),
        torch.randn(8, 2, dtype=torch.float64),
        torch.tensor([0]),
        mode="sum",
        per_sample_weights=torch.ones(4, dtype=torch.float64),
    )

    bad_mode = verify_embedding_bag(
        (4,),
        weight_shape=(8, 2),
        offsets_shape=(1,),
        per_sample_weights_shape=(4,),
        mode="mean",
    )
    assert not bad_mode.ok
    assert bad_mode.error_kind == "per_sample_weights"
    with pytest.raises(NotImplementedError):
        F.embedding_bag(
            torch.tensor([0, 1, 2, 3]),
            torch.randn(8, 2),
            torch.tensor([0]),
            mode="mean",
            per_sample_weights=torch.ones(4),
        )

    bad_dtype = verify_embedding_bag(
        (4,),
        weight_shape=(8, 2),
        offsets_shape=(1,),
        per_sample_weights_shape=(4,),
        weight_dtype="float32",
        per_sample_weights_dtype="float64",
        mode="sum",
    )
    assert not bad_dtype.ok
    assert bad_dtype.error_kind == "dtype"


class _EmbeddingBagOffsets(nn.Module):
    def __init__(self, offsets, *, include_last_offset=False, out_features=2):
        super().__init__()
        self.bag = nn.EmbeddingBag(
            8, 4, mode="sum", include_last_offset=include_last_offset
        )
        self.proj = nn.Linear(4, out_features)
        self.register_buffer("offsets", torch.tensor(offsets, dtype=torch.long))

    def forward(self, idx):
        return self.proj(self.bag(idx, self.offsets))


def test_fx_module_embedding_bag_checks_static_offsets_and_output_shape():
    safe = verify_module(
        _EmbeddingBagOffsets([0, 2, 4], include_last_offset=True).eval(),
        input_shapes={"idx": (4,)},
        input_dtypes={"idx": "int64"},
        backend="fx",
    )
    assert safe.safe

    bad_offsets = verify_module(
        _EmbeddingBagOffsets([1]).eval(),
        input_shapes={"idx": (3,)},
        input_dtypes={"idx": "int64"},
        backend="fx",
    )
    assert not bad_offsets.safe
    assert any("offsets[0] must be 0" in m for m in _messages(bad_offsets))

    with pytest.raises(RuntimeError, match="offsets\\[0\\]"):
        _EmbeddingBagOffsets([1]).eval()(torch.tensor([0, 1, 2]))


class _EmbeddingBagPerSampleWeights(nn.Module):
    def __init__(self):
        super().__init__()
        self.bag = nn.EmbeddingBag(8, 4, mode="mean")

    def forward(self, idx, offsets, weights):
        return self.bag(idx, offsets, weights)


def test_fx_module_embedding_bag_rejects_per_sample_weights_outside_sum_mode():
    result = verify_module(
        _EmbeddingBagPerSampleWeights().eval(),
        input_shapes={"idx": (4,), "offsets": (1,), "weights": (4,)},
        input_dtypes={"idx": "int64", "offsets": "int64", "weights": "float32"},
        backend="fx",
    )
    assert not result.safe
    assert any("per_sample_weights is only supported" in m for m in _messages(result))

    with pytest.raises(NotImplementedError):
        _EmbeddingBagPerSampleWeights().eval()(
            torch.tensor([0, 1, 2, 3]), torch.tensor([0]), torch.ones(4)
        )


class _FunctionalEmbeddingBag(nn.Module):
    def __init__(self, out_features=2):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(6, 3))
        self.proj = nn.Linear(3, out_features)

    def forward(self, idx, offsets):
        pooled = F.embedding_bag(idx, self.weight, offsets, mode="sum")
        return self.proj(pooled)


def test_fx_functional_embedding_bag_uses_weight_shape_and_feeds_downstream_layers():
    safe = verify_module(
        _FunctionalEmbeddingBag(out_features=2).eval(),
        input_shapes={"idx": (4,), "offsets": (2,)},
        input_dtypes={"idx": "int64", "offsets": "int64"},
        backend="fx",
    )
    assert safe.safe

    class BadFunctional(_FunctionalEmbeddingBag):
        def __init__(self):
            super().__init__(out_features=2)
            self.proj = nn.Linear(4, 2)

    bad = verify_module(
        BadFunctional().eval(),
        input_shapes={"idx": (4,), "offsets": (2,)},
        input_dtypes={"idx": "int64", "offsets": "int64"},
        backend="fx",
    )
    assert not bad.safe
    assert any("Linear" in m or "in_features" in m for m in _messages(bad))


def test_torchrec_jagged_contract_abstains_on_ragged_boundaries():
    spec = TorchRecJaggedSpec(
        values_shape=("nnz",),
        offsets_shape=("batch_plus_1",),
        values_dtype="int64",
        offsets_dtype="int64",
    )
    verdict = verify_torchrec_embedding_bag(spec, embedding_dim=16, pooling="sum")
    assert verdict.ok
    assert verdict.output_shape == ("(batch_plus_1-1)", 16)
    assert verdict.unknown_reason


def test_public_exports_embedding_bag_contract():
    import src
    import tensorguard

    assert src.verify_embedding_bag is verify_embedding_bag
    assert tensorguard.verify_torchrec_embedding_bag is verify_torchrec_embedding_bag
    assert isinstance(verify_embedding_bag((2, 3), weight_shape=(5, 7)), EmbeddingBagVerdict)

