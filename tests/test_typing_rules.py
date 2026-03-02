"""Tests for formal typing rules (src/typing_rules.py).

Covers all eight rules: T-LINEAR, T-CONV2D, T-BROADCAST, T-RESHAPE,
T-CAT, T-MATMUL, T-REDUCE, T-EMBED.  Includes positive, negative,
soundness (runtime match), and property-based tests.
"""

import math
import random

import pytest

from src.typing_rules import (
    TensorType,
    Judgement,
    TypingRuleError,
    apply_t_linear,
    apply_t_conv2d,
    apply_t_broadcast,
    apply_t_reshape,
    apply_t_cat,
    apply_t_matmul,
    apply_t_reduce,
    apply_t_embed,
    verify_rule,
    generate_random_judgement,
)


# ═══════════════════════════════════════════════════════════════════════════
# T-LINEAR
# ═══════════════════════════════════════════════════════════════════════════

class TestTLinear:
    def test_basic(self):
        x = TensorType(shape=(32, 784))
        out = apply_t_linear(x, 784, 256)
        assert out.shape == (32, 256)

    def test_preserves_device_dtype(self):
        x = TensorType(shape=(4, 10), device="cuda:0", dtype="float16")
        out = apply_t_linear(x, 10, 20)
        assert out.device == "cuda:0"
        assert out.dtype == "float16"

    def test_batched_3d(self):
        x = TensorType(shape=(2, 8, 64))
        out = apply_t_linear(x, 64, 32)
        assert out.shape == (2, 8, 32)

    def test_wrong_in_features(self):
        x = TensorType(shape=(4, 10))
        with pytest.raises(TypingRuleError, match="in_features"):
            apply_t_linear(x, 20, 5)

    def test_scalar_rejected(self):
        x = TensorType(shape=())
        with pytest.raises(TypingRuleError, match="ndim"):
            apply_t_linear(x, 1, 1)


# ═══════════════════════════════════════════════════════════════════════════
# T-CONV2D
# ═══════════════════════════════════════════════════════════════════════════

class TestTConv2d:
    def test_basic_no_padding(self):
        x = TensorType(shape=(1, 3, 32, 32))
        out = apply_t_conv2d(x, out_channels=16, kernel_size=(3, 3))
        assert out.shape == (1, 16, 30, 30)

    def test_with_padding(self):
        x = TensorType(shape=(2, 3, 28, 28))
        out = apply_t_conv2d(x, 16, kernel_size=(5, 5), padding=(2, 2))
        assert out.shape == (2, 16, 28, 28)

    def test_stride(self):
        x = TensorType(shape=(1, 3, 64, 64))
        out = apply_t_conv2d(x, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        assert out.shape == (1, 32, 32, 32)

    def test_wrong_ndim(self):
        x = TensorType(shape=(3, 32, 32))
        with pytest.raises(TypingRuleError, match="ndim"):
            apply_t_conv2d(x, 16, (3, 3))


# ═══════════════════════════════════════════════════════════════════════════
# T-BROADCAST
# ═══════════════════════════════════════════════════════════════════════════

class TestTBroadcast:
    def test_same_shape(self):
        a = TensorType(shape=(3, 4))
        b = TensorType(shape=(3, 4))
        out = apply_t_broadcast(a, b)
        assert out.shape == (3, 4)

    def test_scalar_broadcast(self):
        a = TensorType(shape=(3, 4))
        b = TensorType(shape=(1,))
        out = apply_t_broadcast(a, b)
        assert out.shape == (3, 4)

    def test_expand_ones(self):
        a = TensorType(shape=(1, 4))
        b = TensorType(shape=(3, 1))
        out = apply_t_broadcast(a, b)
        assert out.shape == (3, 4)

    def test_incompatible(self):
        a = TensorType(shape=(3,))
        b = TensorType(shape=(4,))
        with pytest.raises(TypingRuleError, match="incompatible"):
            apply_t_broadcast(a, b)


# ═══════════════════════════════════════════════════════════════════════════
# T-RESHAPE
# ═══════════════════════════════════════════════════════════════════════════

class TestTReshape:
    def test_flatten(self):
        x = TensorType(shape=(2, 3, 4))
        out = apply_t_reshape(x, (24,))
        assert out.shape == (24,)

    def test_infer_minus_one(self):
        x = TensorType(shape=(2, 3, 4))
        out = apply_t_reshape(x, (6, -1))
        assert out.shape == (6, 4)

    def test_numel_mismatch(self):
        x = TensorType(shape=(2, 3))
        with pytest.raises(TypingRuleError, match="numel"):
            apply_t_reshape(x, (5,))

    def test_two_minus_ones_rejected(self):
        x = TensorType(shape=(2, 3))
        with pytest.raises(TypingRuleError, match="-1"):
            apply_t_reshape(x, (-1, -1))


# ═══════════════════════════════════════════════════════════════════════════
# T-CAT
# ═══════════════════════════════════════════════════════════════════════════

class TestTCat:
    def test_basic_cat_dim0(self):
        a = TensorType(shape=(2, 4))
        b = TensorType(shape=(3, 4))
        out = apply_t_cat([a, b], dim=0)
        assert out.shape == (5, 4)

    def test_cat_dim1(self):
        a = TensorType(shape=(2, 3))
        b = TensorType(shape=(2, 5))
        out = apply_t_cat([a, b], dim=1)
        assert out.shape == (2, 8)

    def test_dim_mismatch(self):
        a = TensorType(shape=(2, 4))
        b = TensorType(shape=(3, 5))
        with pytest.raises(TypingRuleError, match="mismatch"):
            apply_t_cat([a, b], dim=0)

    def test_empty_list(self):
        with pytest.raises(TypingRuleError, match="at least one"):
            apply_t_cat([], dim=0)


# ═══════════════════════════════════════════════════════════════════════════
# T-MATMUL
# ═══════════════════════════════════════════════════════════════════════════

class TestTMatmul:
    def test_2d(self):
        a = TensorType(shape=(3, 4))
        b = TensorType(shape=(4, 5))
        out = apply_t_matmul(a, b)
        assert out.shape == (3, 5)

    def test_1d_dot(self):
        a = TensorType(shape=(4,))
        b = TensorType(shape=(4,))
        out = apply_t_matmul(a, b)
        assert out.shape == ()

    def test_batched(self):
        a = TensorType(shape=(2, 3, 4))
        b = TensorType(shape=(2, 4, 5))
        out = apply_t_matmul(a, b)
        assert out.shape == (2, 3, 5)

    def test_inner_dim_mismatch(self):
        a = TensorType(shape=(3, 4))
        b = TensorType(shape=(5, 6))
        with pytest.raises(TypingRuleError, match="inner dims"):
            apply_t_matmul(a, b)

    def test_1d_by_2d(self):
        a = TensorType(shape=(4,))
        b = TensorType(shape=(4, 5))
        out = apply_t_matmul(a, b)
        assert out.shape == (5,)


# ═══════════════════════════════════════════════════════════════════════════
# T-REDUCE
# ═══════════════════════════════════════════════════════════════════════════

class TestTReduce:
    def test_remove_dim(self):
        x = TensorType(shape=(2, 3, 4))
        out = apply_t_reduce(x, dim=1)
        assert out.shape == (2, 4)

    def test_keepdim(self):
        x = TensorType(shape=(2, 3, 4))
        out = apply_t_reduce(x, dim=1, keepdim=True)
        assert out.shape == (2, 1, 4)

    def test_negative_dim(self):
        x = TensorType(shape=(2, 3, 4))
        out = apply_t_reduce(x, dim=-1)
        assert out.shape == (2, 3)

    def test_scalar_rejected(self):
        x = TensorType(shape=())
        with pytest.raises(TypingRuleError, match="scalar"):
            apply_t_reduce(x, dim=0)


# ═══════════════════════════════════════════════════════════════════════════
# T-EMBED
# ═══════════════════════════════════════════════════════════════════════════

class TestTEmbed:
    def test_basic(self):
        x = TensorType(shape=(4, 10), dtype="int64")
        out = apply_t_embed(x, num_embeddings=1000, embedding_dim=128)
        assert out.shape == (4, 10, 128)
        assert out.dtype == "float32"

    def test_1d_indices(self):
        x = TensorType(shape=(5,), dtype="int64")
        out = apply_t_embed(x, num_embeddings=500, embedding_dim=64)
        assert out.shape == (5, 64)


# ═══════════════════════════════════════════════════════════════════════════
# verify_rule
# ═══════════════════════════════════════════════════════════════════════════

class TestVerifyRule:
    def test_linear_ok(self):
        ok, ty = verify_rule(
            "T-LINEAR",
            {"x": TensorType(shape=(8, 16))},
            {"in_features": 16, "out_features": 32},
        )
        assert ok
        assert ty is not None
        assert ty.shape == (8, 32)

    def test_linear_fail(self):
        ok, ty = verify_rule(
            "T-LINEAR",
            {"x": TensorType(shape=(8, 16))},
            {"in_features": 99, "out_features": 32},
        )
        assert not ok
        assert ty is None

    def test_unknown_rule(self):
        with pytest.raises(ValueError, match="Unknown rule"):
            verify_rule("T-UNKNOWN", {}, {})


# ═══════════════════════════════════════════════════════════════════════════
# Soundness: static types match PyTorch runtime (when available)
# ═══════════════════════════════════════════════════════════════════════════

_torch_available = False
try:
    import torch
    _torch_available = True
except ImportError:
    pass


@pytest.mark.skipif(not _torch_available, reason="PyTorch not installed")
class TestRuntimeSoundness:
    """Each static rule result matches the actual PyTorch output shape."""

    def test_linear_runtime(self):
        import torch
        x = torch.randn(4, 10)
        layer = torch.nn.Linear(10, 20)
        y = layer(x)
        static = apply_t_linear(TensorType(shape=(4, 10)), 10, 20)
        assert static.shape == tuple(y.shape)

    def test_conv2d_runtime(self):
        import torch
        x = torch.randn(1, 3, 32, 32)
        layer = torch.nn.Conv2d(3, 16, kernel_size=3)
        y = layer(x)
        static = apply_t_conv2d(TensorType(shape=(1, 3, 32, 32)), 16, (3, 3))
        assert static.shape == tuple(y.shape)

    def test_matmul_runtime(self):
        import torch
        a = torch.randn(2, 3, 4)
        b = torch.randn(2, 4, 5)
        y = a @ b
        static = apply_t_matmul(
            TensorType(shape=(2, 3, 4)), TensorType(shape=(2, 4, 5))
        )
        assert static.shape == tuple(y.shape)


# ═══════════════════════════════════════════════════════════════════════════
# Property-based: random judgement generation
# ═══════════════════════════════════════════════════════════════════════════

class TestRandomJudgement:
    def test_generates_valid_judgement(self):
        rng = random.Random(42)
        for _ in range(50):
            j = generate_random_judgement(rng)
            assert isinstance(j, Judgement)
            assert isinstance(j.type, TensorType)
            assert len(j.context) > 0

    def test_all_dims_positive(self):
        rng = random.Random(123)
        for _ in range(50):
            j = generate_random_judgement(rng)
            for d in j.type.shape:
                if isinstance(d, int):
                    assert d > 0, f"Non-positive dim {d} in {j}"
