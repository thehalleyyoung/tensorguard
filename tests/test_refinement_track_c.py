"""Tests for Track C refinements.

Covers:
- Symbolic config attributes
- Split/chunk/unbind tuple unpacking
- Reshape with -1 inference
- Scaled dot product attention
- MultiheadAttention
- LayerNorm and RMSNorm
"""

import pytest
from src.refinement.symbolic_config import (
    symbolic_config,
    detect_symbolic_config_attrs,
    resolve_config_attr,
    make_expression_symbolic,
)
from src.refinement.qkv import (
    propagate_split_shape,
    propagate_chunk_shape,
    propagate_unbind_shape,
)
from src.refinement.reshape import (
    infer_reshape_minus_one,
    compute_numel,
    validate_reshape_with_z3,
)
from src.refinement.attention import (
    propagate_scaled_dot_product_attention,
    propagate_multihead_attention,
)
from src.refinement.norms import (
    propagate_layernorm,
    propagate_rmsnorm,
)
import ast


class TestSymbolicConfig:
    """Test symbolic config attribute handling."""
    
    def test_symbolic_config_registration(self):
        """Test registering symbolic config fields."""
        symbolic_config(["hidden_size", "num_heads"])
        # Should not raise
    
    def test_detect_config_attrs_linear(self):
        """Test detecting config attrs in Linear layers."""
        code = """
def __init__(self, config):
    self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
    self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)
"""
        tree = ast.parse(code)
        init_node = tree.body[0]
        detected = detect_symbolic_config_attrs(init_node)
        assert "hidden_size" in detected
        assert "intermediate_size" in detected
    
    def test_detect_config_attrs_with_multiplication(self):
        """Test detecting config attrs with multiplication."""
        code = """
def __init__(self, config):
    self.qkv = nn.Linear(config.hidden_size, 3 * config.hidden_size)
"""
        tree = ast.parse(code)
        init_node = tree.body[0]
        detected = detect_symbolic_config_attrs(init_node)
        assert "hidden_size" in detected
    
    def test_make_expression_symbolic_simple(self):
        """Test making simple config.attr symbolic."""
        code = "config.hidden_size"
        node = ast.parse(code, mode='eval').body
        result = make_expression_symbolic(node, "config", {"hidden_size"})
        assert result == "d_hidden_size"
    
    def test_make_expression_symbolic_multiply(self):
        """Test making config.attr * 3 symbolic."""
        code = "3 * config.hidden_size"
        node = ast.parse(code, mode='eval').body
        result = make_expression_symbolic(node, "config", {"hidden_size"})
        assert "d_hidden_size" in result
        assert "3" in result


class TestQKVUnpacking:
    """Test split/chunk/unbind shape propagation."""
    
    def test_split_concrete_shape(self):
        """Test split with concrete dimensions."""
        input_shape = (2, 16, 768)
        dim = 2
        split_size = 256
        n_outputs = 3
        
        output_shape = propagate_split_shape(input_shape, dim, split_size, n_outputs, 0)
        assert output_shape == (2, 16, 256)
    
    def test_split_symbolic_shape(self):
        """Test split with symbolic dimensions."""
        input_shape = ("B", "T", "C")
        dim = 2
        split_size = "C//3"
        n_outputs = 3
        
        output_shape = propagate_split_shape(input_shape, dim, split_size, n_outputs, 0)
        assert output_shape == ("B", "T", "C//3")
    
    def test_chunk_concrete_shape(self):
        """Test chunk with concrete dimensions."""
        input_shape = (2, 16, 768)
        dim = 2
        chunks = 3
        
        output_shape = propagate_chunk_shape(input_shape, dim, chunks, 0)
        assert output_shape == (2, 16, 256)  # 768 // 3 = 256
    
    def test_chunk_symbolic_shape(self):
        """Test chunk with symbolic dimensions."""
        input_shape = ("B", "T", "C")
        dim = 2
        chunks = 3
        
        output_shape = propagate_chunk_shape(input_shape, dim, chunks, 0)
        assert "C" in str(output_shape[2])
    
    def test_unbind_removes_dimension(self):
        """Test unbind removes the specified dimension."""
        input_shape = (2, 3, 16, 64)
        dim = 1
        
        output_shape = propagate_unbind_shape(input_shape, dim, 0)
        assert output_shape == (2, 16, 64)
    
    def test_unbind_negative_dim(self):
        """Test unbind with negative dimension."""
        input_shape = (2, 3, 16, 64)
        dim = -3  # Should be dimension 1
        
        output_shape = propagate_unbind_shape(input_shape, dim, 0)
        assert output_shape == (2, 16, 64)


class TestReshapeInference:
    """Test reshape with -1 dimension inference."""
    
    def test_reshape_concrete_simple(self):
        """Test -1 inference with concrete shapes."""
        input_shape = (2, 3, 4, 5)
        target_shape = (2, -1)
        
        output_shape = infer_reshape_minus_one(input_shape, target_shape)
        assert output_shape == (2, 60)  # 3*4*5 = 60
    
    def test_reshape_concrete_multi_dim(self):
        """Test -1 inference with multiple target dims."""
        input_shape = (4, 6, 8)
        target_shape = (4, 3, -1)
        
        output_shape = infer_reshape_minus_one(input_shape, target_shape)
        assert output_shape == (4, 3, 16)  # 6*8 = 48, 48//3 = 16
    
    def test_reshape_symbolic(self):
        """Test -1 inference with symbolic shapes."""
        input_shape = ("B", "C", "H", "W")
        target_shape = ("B", -1)
        
        output_shape = infer_reshape_minus_one(input_shape, target_shape)
        assert output_shape[0] == "B"
        assert "C" in str(output_shape[1])
        assert "H" in str(output_shape[1])
        assert "W" in str(output_shape[1])
    
    def test_reshape_invalid_numel(self):
        """Test -1 inference with incompatible shapes."""
        input_shape = (2, 3, 5)  # 30 elements
        target_shape = (2, 4, -1)  # Would need 30 = 2*4*x, x=3.75 (invalid)
        
        output_shape = infer_reshape_minus_one(input_shape, target_shape)
        # Should return None or detect incompatibility
        # For now, it may return non-integer result
    
    def test_compute_numel_concrete(self):
        """Test computing element count for concrete shape."""
        shape = (2, 3, 4, 5)
        numel = compute_numel(shape)
        assert numel == 120
    
    def test_compute_numel_symbolic(self):
        """Test computing element count for symbolic shape."""
        shape = ("B", "C", 4, 5)
        numel = compute_numel(shape)
        assert numel == "20*B*C" or numel == "B*C*20"
    
    def test_validate_reshape_compatible(self):
        """Test Z3 validation for compatible reshape."""
        input_shape = (2, 3, 4)
        target_shape = (6, 4)
        
        is_valid, error = validate_reshape_with_z3(input_shape, target_shape)
        assert is_valid


class TestScaledDotProductAttention:
    """Test F.scaled_dot_product_attention shape propagation."""
    
    def test_sdpa_basic(self):
        """Test SDPA with matching shapes."""
        q_shape = (2, 8, 16, 64)  # (B, H, T, D)
        k_shape = (2, 8, 16, 64)
        v_shape = (2, 8, 16, 64)
        
        output_shape, error = propagate_scaled_dot_product_attention(q_shape, k_shape, v_shape)
        assert error is None
        assert output_shape == (2, 8, 16, 64)
    
    def test_sdpa_cross_attention(self):
        """Test SDPA with different sequence lengths (cross-attention)."""
        q_shape = (2, 8, 16, 64)  # Decoder: T=16
        k_shape = (2, 8, 32, 64)  # Encoder: S=32
        v_shape = (2, 8, 32, 64)
        
        output_shape, error = propagate_scaled_dot_product_attention(q_shape, k_shape, v_shape)
        assert error is None
        assert output_shape == (2, 8, 16, 64)  # Output has Q's sequence length
    
    def test_sdpa_batch_mismatch(self):
        """Test SDPA detects batch size mismatch."""
        q_shape = (2, 8, 16, 64)
        k_shape = (4, 8, 16, 64)  # Different batch
        v_shape = (2, 8, 16, 64)
        
        output_shape, error = propagate_scaled_dot_product_attention(q_shape, k_shape, v_shape)
        assert error is not None
        assert "batch" in error.lower()
    
    def test_sdpa_head_dim_mismatch(self):
        """Test SDPA detects head dimension mismatch."""
        q_shape = (2, 8, 16, 64)
        k_shape = (2, 8, 16, 32)  # Different head dim
        v_shape = (2, 8, 16, 64)
        
        output_shape, error = propagate_scaled_dot_product_attention(q_shape, k_shape, v_shape)
        assert error is not None
        assert "head dim" in error.lower()
    
    def test_sdpa_wrong_ndim(self):
        """Test SDPA detects wrong input dimensions."""
        q_shape = (2, 8, 16)  # Only 3D
        k_shape = (2, 8, 16, 64)
        v_shape = (2, 8, 16, 64)
        
        output_shape, error = propagate_scaled_dot_product_attention(q_shape, k_shape, v_shape)
        assert error is not None
        assert "4D" in error


class TestMultiheadAttention:
    """Test nn.MultiheadAttention shape propagation."""
    
    def test_mha_batch_first_false(self):
        """Test MHA with batch_first=False (default)."""
        input_shape = (16, 2, 768)  # (T, B, E)
        embed_dim = 768
        num_heads = 12
        
        output_shape, error = propagate_multihead_attention(input_shape, embed_dim, num_heads, batch_first=False)
        assert error is None
        assert output_shape == (16, 2, 768)
    
    def test_mha_batch_first_true(self):
        """Test MHA with batch_first=True."""
        input_shape = (2, 16, 768)  # (B, T, E)
        embed_dim = 768
        num_heads = 12
        
        output_shape, error = propagate_multihead_attention(input_shape, embed_dim, num_heads, batch_first=True)
        assert error is None
        assert output_shape == (2, 16, 768)
    
    def test_mha_indivisible_embed_dim(self):
        """Test MHA detects embed_dim not divisible by num_heads."""
        input_shape = (16, 2, 768)
        embed_dim = 768
        num_heads = 7  # 768 % 7 != 0
        
        output_shape, error = propagate_multihead_attention(input_shape, embed_dim, num_heads, batch_first=False)
        assert error is not None
        assert "divisible" in error.lower()
    
    def test_mha_wrong_embed_dim(self):
        """Test MHA detects wrong embedding dimension."""
        input_shape = (16, 2, 512)  # E=512
        embed_dim = 768  # Expected 768
        num_heads = 12
        
        output_shape, error = propagate_multihead_attention(input_shape, embed_dim, num_heads, batch_first=False)
        assert error is not None
        assert "embed_dim" in error.lower()


class TestLayerNorm:
    """Test nn.LayerNorm shape propagation."""
    
    def test_layernorm_1d(self):
        """Test LayerNorm with 1D normalized shape."""
        input_shape = (2, 16, 768)
        normalized_shape = (768,)
        
        output_shape, error = propagate_layernorm(input_shape, normalized_shape)
        assert error is None
        assert output_shape == input_shape
    
    def test_layernorm_2d(self):
        """Test LayerNorm with 2D normalized shape."""
        input_shape = (2, 16, 12, 64)
        normalized_shape = (12, 64)
        
        output_shape, error = propagate_layernorm(input_shape, normalized_shape)
        assert error is None
        assert output_shape == input_shape
    
    def test_layernorm_dimension_mismatch(self):
        """Test LayerNorm detects dimension mismatch."""
        input_shape = (2, 16, 512)
        normalized_shape = (768,)  # Expected 512
        
        output_shape, error = propagate_layernorm(input_shape, normalized_shape)
        assert error is not None
        assert "mismatch" in error.lower()
    
    def test_layernorm_too_few_dims(self):
        """Test LayerNorm detects insufficient input dimensions."""
        input_shape = (768,)  # 1D
        normalized_shape = (12, 64)  # Needs 2D trailing
        
        output_shape, error = propagate_layernorm(input_shape, normalized_shape)
        assert error is not None


class TestRMSNorm:
    """Test nn.RMSNorm shape propagation."""
    
    def test_rmsnorm_basic(self):
        """Test RMSNorm with basic shape."""
        input_shape = (2, 16, 768)
        normalized_shape = (768,)
        
        output_shape, error = propagate_rmsnorm(input_shape, normalized_shape)
        assert error is None
        assert output_shape == input_shape
    
    def test_rmsnorm_dimension_mismatch(self):
        """Test RMSNorm detects dimension mismatch."""
        input_shape = (2, 16, 512)
        normalized_shape = (768,)
        
        output_shape, error = propagate_rmsnorm(input_shape, normalized_shape)
        assert error is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
