"""Step 204 — PyTorch loss-function shape, reduction, and dtype contracts."""

from __future__ import annotations

import pytest

from src.loss_verify import LossVerdict, verify_loss
from src.model_checker import verify_model


torch = pytest.importorskip("torch")
nn = pytest.importorskip("torch.nn")
F = pytest.importorskip("torch.nn.functional")


def _messages(result) -> str:
    if result.counterexample is None:
        return ""
    return "\n".join(v.message for v in result.counterexample.violations)


def test_cross_entropy_class_and_probability_targets_match_pytorch_shapes():
    class_mode = verify_loss(
        "CrossEntropyLoss",
        (4, 7, 8, 8),
        (4, 8, 8),
        input_dtype="float32",
        target_dtype="int64",
        reduction="none",
        weight_shape=(7,),
    )
    assert class_mode.ok
    assert class_mode.output_shape == (4, 8, 8)
    assert class_mode.output_dtype == "float32"

    probability_mode = verify_loss(
        "cross_entropy",
        (3, 5),
        (3, 5),
        input_dtype=torch.float64,
        target_dtype=torch.float64,
        reduction="none",
    )
    assert probability_mode.ok
    assert probability_mode.output_shape == (3,)
    assert probability_mode.output_dtype == "float64"

    reduced = verify_loss("cross_entropy", (5,), (), target_dtype="int64")
    assert reduced.ok
    assert reduced.output_shape == ()


def test_cross_entropy_rejects_bad_target_dtype_and_weight_shape():
    bad_target = verify_loss(
        "cross_entropy",
        (4, 7),
        (4,),
        input_dtype="float32",
        target_dtype="float32",
    )
    assert not bad_target.ok
    assert bad_target.error_kind == "dtype"
    assert "int64" in bad_target.message

    bad_weight = verify_loss(
        "CrossEntropyLoss",
        (4, 7),
        (4,),
        input_dtype="float32",
        target_dtype="int64",
        weight_shape=(6,),
    )
    assert not bad_weight.ok
    assert bad_weight.error_kind == "shape"
    assert "weight has 6 classes" in bad_weight.message


def test_nll_loss_requires_class_index_targets():
    ok = verify_loss(
        "NLLLoss",
        (2, 3, 4),
        (2, 4),
        input_dtype="float32",
        target_dtype="int64",
        reduction="none",
    )
    assert ok.ok
    assert ok.output_shape == (2, 4)

    bad = verify_loss(
        "nll_loss",
        (2, 3, 4),
        (2, 3, 4),
        input_dtype="float32",
        target_dtype="int64",
    )
    assert not bad.ok
    assert bad.error_kind == "shape"


def test_mse_loss_broadcasting_and_dtype_contracts():
    broadcast = verify_loss(
        "MSELoss",
        (3, 5),
        (1, 5),
        input_dtype="int64",
        target_dtype="float32",
        reduction="none",
    )
    assert broadcast.ok
    assert broadcast.output_shape == (3, 5)
    assert broadcast.output_dtype == "float32"

    bad_shape = verify_loss("mse_loss", (3, 5), (4, 5))
    assert not bad_shape.ok
    assert bad_shape.error_kind == "shape"

    bad_dtype = verify_loss(
        "mse_loss",
        (3, 5),
        (3, 5),
        input_dtype="int64",
        target_dtype="int64",
    )
    assert not bad_dtype.ok
    assert bad_dtype.error_kind == "dtype"


def test_bce_with_logits_exact_target_shape_dtype_and_pos_weight():
    ok = verify_loss(
        "BCEWithLogitsLoss",
        (3, 5),
        (3, 5),
        input_dtype="float32",
        target_dtype="float32",
        pos_weight_shape=(5,),
        reduction="none",
    )
    assert ok.ok
    assert ok.output_shape == (3, 5)

    no_target_broadcast = verify_loss(
        "binary_cross_entropy_with_logits",
        (3, 5),
        (1, 5),
        input_dtype="float32",
        target_dtype="float32",
    )
    assert not no_target_broadcast.ok
    assert no_target_broadcast.error_kind == "shape"
    assert "exactly the same shape" in no_target_broadcast.message

    bad_dtype = verify_loss(
        "binary_cross_entropy_with_logits",
        (3, 5),
        (3, 5),
        input_dtype="float32",
        target_dtype="int64",
    )
    assert not bad_dtype.ok
    assert bad_dtype.error_kind == "dtype"

    bad_pos_weight = verify_loss(
        "BCEWithLogitsLoss",
        (3, 5),
        (3, 5),
        input_dtype="float32",
        target_dtype="float32",
        pos_weight_shape=(4,),
    )
    assert not bad_pos_weight.ok
    assert bad_pos_weight.error_kind == "shape"


def test_kl_div_broadcast_batchmean_and_dtype_contracts():
    ok = verify_loss(
        "KLDivLoss",
        (3, 5),
        (1, 5),
        input_dtype="float32",
        target_dtype="float32",
        reduction="batchmean",
    )
    assert ok.ok
    assert ok.output_shape == ()

    bad_dtype = verify_loss(
        "kl_div",
        (3, 5),
        (3, 5),
        input_dtype="float32",
        target_dtype="int64",
    )
    assert not bad_dtype.ok
    assert bad_dtype.error_kind == "dtype"

    bad_reduction = verify_loss("cross_entropy", (3, 5), (3,), reduction="batchmean")
    assert not bad_reduction.ok
    assert bad_reduction.error_kind == "reduction"


def test_symbolic_dims_abstain_without_false_positive():
    verdict = verify_loss(
        "mse_loss",
        ("batch", 5),
        ("other_batch", 5),
        input_dtype="float32",
        target_dtype="float32",
        reduction="none",
    )
    assert verdict.ok
    assert verdict.output_shape == ("batch", 5)
    assert verdict.unknown_reason


def test_public_exports_loss_verifier():
    import src
    import tensorguard

    assert src.verify_loss is verify_loss
    assert tensorguard.verify_loss is verify_loss
    assert isinstance(verify_loss("mse_loss", (2,), (2,)), LossVerdict)


def test_verify_model_catches_cross_entropy_target_dtype_in_training_source():
    source = """
import torch
import torch.nn as nn

class TrainStep(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss = nn.CrossEntropyLoss()

    def forward(self, logits, target):
        return self.loss(logits, target)
"""
    result = verify_model(
        source,
        input_shapes={"logits": (4, 5), "target": (4,)},
        input_dtypes={"logits": "float32", "target": "float32"},
        infer_inputs=False,
    )
    assert not result.safe
    assert "class-index targets must be int64" in _messages(result)

    with pytest.raises(RuntimeError):
        nn.CrossEntropyLoss()(torch.randn(4, 5), torch.rand(4))


def test_verify_model_catches_functional_bce_shape_in_training_source():
    source = """
import torch
import torch.nn as nn
import torch.nn.functional as F

class TrainStep(nn.Module):
    def forward(self, logits, target):
        return F.binary_cross_entropy_with_logits(logits, target, reduction="none")
"""
    result = verify_model(
        source,
        input_shapes={"logits": (3, 5), "target": (1, 5)},
        input_dtypes={"logits": "float32", "target": "float32"},
        infer_inputs=False,
    )
    assert not result.safe
    assert "exactly the same shape" in _messages(result)

    with pytest.raises(ValueError):
        F.binary_cross_entropy_with_logits(torch.randn(3, 5), torch.rand(1, 5))


def test_verify_module_catches_fx_loss_module_shape_contract():
    from src.fx_extractor import verify_module

    class TrainStep(nn.Module):
        def __init__(self):
            super().__init__()
            self.loss = nn.BCEWithLogitsLoss()

        def forward(self, logits, target):
            return self.loss(logits, target)

    result = verify_module(
        TrainStep(),
        input_shapes={"logits": (3, 5), "target": (1, 5)},
        input_dtypes={"logits": "float32", "target": "float32"},
    )
    assert not result.safe
    assert "exactly the same shape" in _messages(result)
