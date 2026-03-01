"""
Tests for graph compiler (arbitrary computation graph support)
and composition soundness verification.
"""

import pytest
import ast
from src.graph_compiler import (
    compile_model,
    verify_arbitrary_model,
    analyze_moe_shapes,
    analyze_torch_cond,
    MoEConfig,
    ShapeConstraint,
    DynamicControlFlow,
    count_registered_transfers,
    _detect_moe_patterns,
    _detect_dynamic_patterns,
    _compute_coverage,
    TransferFunction,
    register_transfer,
    get_transfer,
)
from src.model_checker import TensorShape, ShapeDim


# ═══════════════════════════════════════════════════════════════════════════════
# Transfer function registry tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestTransferRegistry:
    def test_registry_has_many_ops(self):
        """Registry should have 100+ ops from initialization."""
        count = count_registered_transfers()
        assert count >= 100, f"Expected 100+ transfers, got {count}"

    def test_activation_registered(self):
        tf = get_transfer("F.relu")
        assert tf is not None
        assert tf.preserves_shape

    def test_gelu_registered(self):
        tf = get_transfer("F.gelu")
        assert tf is not None
        assert tf.preserves_shape

    def test_silu_registered(self):
        tf = get_transfer("F.silu")
        assert tf is not None
        assert tf.preserves_shape

    def test_elementwise_registered(self):
        tf = get_transfer("torch.abs")
        assert tf is not None
        assert tf.preserves_shape

    def test_reduction_registered(self):
        tf = get_transfer("torch.sum")
        assert tf is not None
        assert not tf.preserves_shape

    def test_matmul_registered(self):
        tf = get_transfer("torch.matmul")
        assert tf is not None

    def test_comparison_registered(self):
        tf = get_transfer("torch.eq")
        assert tf is not None
        assert tf.preserves_shape

    def test_fft_registered(self):
        tf = get_transfer("torch.fft.fft")
        assert tf is not None

    def test_custom_registration(self):
        register_transfer("custom.op", TransferFunction(
            name="custom_op", preserves_shape=True))
        tf = get_transfer("custom.op")
        assert tf is not None
        assert tf.preserves_shape

    def test_unknown_returns_none(self):
        tf = get_transfer("nonexistent.op")
        assert tf is None


# ═══════════════════════════════════════════════════════════════════════════════
# MoE analysis tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestMoEAnalysis:
    def test_basic_moe_shapes(self):
        inp = TensorShape.from_tuple((32, 128, 512))
        config = MoEConfig(num_experts=8, top_k=2)
        out, constraints, err = analyze_moe_shapes(inp, config)
        assert err is None
        assert out is not None
        assert out.dims == inp.dims  # MoE preserves shape

    def test_moe_expert_dim_mismatch(self):
        inp = TensorShape.from_tuple((32, 128, 512))
        config = MoEConfig(num_experts=4, top_k=1)
        expert_shapes = [TensorShape.from_tuple((32, 128, 256))]
        out, constraints, err = analyze_moe_shapes(inp, config, expert_shapes)
        assert err is not None
        assert "256" in err and "512" in err

    def test_moe_1d_input_fails(self):
        inp = TensorShape.from_tuple((512,))
        config = MoEConfig(num_experts=8)
        out, constraints, err = analyze_moe_shapes(inp, config)
        assert err is not None

    def test_moe_capacity_constraint(self):
        inp = TensorShape.from_tuple((32, 128, 512))
        config = MoEConfig(num_experts=8, top_k=2, expert_capacity=64)
        out, constraints, err = analyze_moe_shapes(inp, config)
        assert err is None
        assert len(constraints) >= 1


# ═══════════════════════════════════════════════════════════════════════════════
# Dynamic control flow tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestDynamicControlFlow:
    def test_torch_cond_matching_shapes(self):
        true_out = TensorShape.from_tuple((32, 512))
        false_out = TensorShape.from_tuple((32, 512))
        result = analyze_torch_cond(None, true_out, false_out)
        assert result.pattern == "torch_cond"
        assert result.is_shape_preserving
        assert result.merged_shape is not None

    def test_torch_cond_mismatching_shapes(self):
        true_out = TensorShape.from_tuple((32, 512))
        false_out = TensorShape.from_tuple((32, 256))
        result = analyze_torch_cond(None, true_out, false_out)
        assert not result.is_shape_preserving

    def test_torch_cond_mismatching_ranks(self):
        true_out = TensorShape.from_tuple((32, 512))
        false_out = TensorShape.from_tuple((32, 16, 32))
        result = analyze_torch_cond(None, true_out, false_out)
        assert not result.is_shape_preserving

    def test_torch_cond_symbolic_dims(self):
        true_out = TensorShape((ShapeDim("batch"), ShapeDim(512)))
        false_out = TensorShape((ShapeDim("batch"), ShapeDim(512)))
        result = analyze_torch_cond(None, true_out, false_out)
        assert result.is_shape_preserving
        # Should have constraint for symbolic batch dim
        assert len(result.constraints) >= 1


# ═══════════════════════════════════════════════════════════════════════════════
# Graph compiler tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestGraphCompiler:
    def test_compile_simple_mlp(self):
        source = '''
import torch.nn as nn
class SimpleMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 10)
    def forward(self, x):
        x = self.fc1(x)
        return self.fc2(x)
'''
        result = compile_model(source, {"x": ("batch", 784)})
        assert result.strategy == "ast"
        assert result.graph.steps
        assert result.coverage_ratio > 0.9

    def test_compile_transformer(self):
        source = '''
import torch.nn as nn
class TransformerModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(10000, 512)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=6)
        self.fc = nn.Linear(512, 10000)
    def forward(self, x):
        x = self.embedding(x)
        x = self.encoder(x)
        return self.fc(x)
'''
        result = compile_model(source, {"x": ("batch", "seq_len")})
        assert result.strategy == "ast"
        assert result.graph.steps

    def test_compile_resnet_block(self):
        source = '''
import torch
import torch.nn as nn
class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = torch.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        return out + residual
'''
        result = compile_model(source, {"x": ("batch", 64, 32, 32)})
        assert result.strategy == "ast"
        assert result.coverage_ratio > 0.9

    def test_compile_moe_model(self):
        source = '''
import torch
import torch.nn as nn
class MoEModel(nn.Module):
    def __init__(self, d_model=512, num_experts=8):
        super().__init__()
        self.gate = nn.Linear(d_model, num_experts)
        self.experts = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(num_experts)])
    def forward(self, x):
        gate_logits = self.gate(x)
        return x
'''
        result = compile_model(source, {"x": ("batch", "seq", 512)}, detect_moe=True)
        assert result.strategy == "ast"
        assert len(result.warnings) >= 1

    def test_compile_dynamic_model(self):
        source = '''
import torch
import torch.nn as nn
class DynModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(512, 256)
    def forward(self, x):
        if self.training:
            x = torch.dropout(x, 0.5, True)
        return self.fc(x)
'''
        result = compile_model(source, {"x": ("batch", 512)})
        assert result.strategy == "ast"

    def test_compile_empty_source(self):
        result = compile_model("x = 1", {})
        assert result.graph is not None

    def test_compilation_time_recorded(self):
        source = '''
import torch.nn as nn
class M(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 5)
    def forward(self, x):
        return self.fc(x)
'''
        result = compile_model(source)
        assert result.compilation_time_ms >= 0

    def test_coverage_ratio_perfect(self):
        source = '''
import torch.nn as nn
class M(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 5)
    def forward(self, x):
        return self.fc(x)
'''
        result = compile_model(source)
        assert result.coverage_ratio >= 0.8

    def test_syntax_error_handled(self):
        result = compile_model("def (broken syntax", {})
        assert result.strategy == "error"
        assert len(result.warnings) >= 1


# ═══════════════════════════════════════════════════════════════════════════════
# Arbitrary model verification tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestArbitraryModelVerification:
    def test_verify_safe_mlp(self):
        source = '''
import torch.nn as nn
class SafeMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 10)
    def forward(self, x):
        return self.fc2(self.fc1(x))
'''
        result = verify_arbitrary_model(source, {"x": ("batch", 784)})
        assert result.safe

    def test_verify_buggy_mlp(self):
        source = '''
import torch.nn as nn
class BuggyMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(128, 10)
    def forward(self, x):
        x = self.fc1(x)
        return self.fc2(x)
'''
        result = verify_arbitrary_model(source, {"x": ("batch", 784)})
        assert not result.safe

    def test_verify_transformer(self):
        source = '''
import torch.nn as nn
class TModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(1000, 512)
        self.layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
        self.fc = nn.Linear(512, 1000)
    def forward(self, x):
        x = self.emb(x)
        x = self.layer(x)
        return self.fc(x)
'''
        result = verify_arbitrary_model(source, {"x": ("batch", "seq")})
        assert result.safe


# ═══════════════════════════════════════════════════════════════════════════════
# Pattern detection tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestPatternDetection:
    def test_detect_moe_pattern(self):
        source = '''
import torch.nn as nn
class MoE(nn.Module):
    def __init__(self):
        super().__init__()
        self.experts = nn.ModuleList([nn.Linear(512, 512) for _ in range(8)])
'''
        tree = ast.parse(source)
        configs = _detect_moe_patterns(tree)
        assert len(configs) >= 1
        assert configs[0].num_experts == 8

    def test_detect_no_moe(self):
        source = '''
import torch.nn as nn
class Plain(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 5)
'''
        tree = ast.parse(source)
        configs = _detect_moe_patterns(tree)
        assert len(configs) == 0


# ═══════════════════════════════════════════════════════════════════════════════
# Shape constraint tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestShapeConstraints:
    def test_equality_constraint(self):
        c = ShapeConstraint(kind="eq", lhs="dim_0", rhs="dim_1")
        assert c.kind == "eq"

    def test_product_constraint(self):
        c = ShapeConstraint(
            kind="product",
            lhs=("batch", "seq"),
            rhs="total",
        )
        assert c.kind == "product"

    def test_constraint_to_z3(self):
        try:
            import z3
            c = ShapeConstraint(kind="eq", lhs="a", rhs="b")
            ctx = {"a": z3.Int("a"), "b": z3.Int("b")}
            expr = c.to_z3(ctx)
            assert expr is not None
        except ImportError:
            pytest.skip("Z3 not available")

    def test_product_constraint_to_z3(self):
        try:
            import z3
            c = ShapeConstraint(kind="product", lhs=("a", "b"), rhs="c")
            ctx = {
                "a": z3.Int("a"),
                "b": z3.Int("b"),
                "c": z3.Int("c"),
            }
            expr = c.to_z3(ctx)
            assert expr is not None
        except ImportError:
            pytest.skip("Z3 not available")
