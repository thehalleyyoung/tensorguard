"""Attention mechanism shape propagation.

Handles:
- F.scaled_dot_product_attention (SDPA)
- nn.MultiheadAttention (MHA)
"""

from __future__ import annotations
from typing import Tuple, Optional, Any, Dict


def propagate_scaled_dot_product_attention(
    q_shape: Tuple[Any, ...],
    k_shape: Tuple[Any, ...],
    v_shape: Tuple[Any, ...],
    attn_mask_shape: Optional[Tuple[Any, ...]] = None,
) -> Tuple[Optional[Tuple[Any, ...]], Optional[str]]:
    """Propagate shape through F.scaled_dot_product_attention.
    
    SDPA signature:
        Q: (B, H, T, D)
        K: (B, H, T, D)  [or (B, H, S, D) for cross-attention]
        V: (B, H, S, D)
        Output: (B, H, T, D)
    
    Args:
        q_shape: Query shape
        k_shape: Key shape
        v_shape: Value shape
        attn_mask_shape: Optional attention mask shape
        
    Returns:
        (output_shape, error_message) tuple
    """
    # Check all are 4D
    if len(q_shape) != 4:
        return None, f"SDPA expects 4D query, got {len(q_shape)}D"
    if len(k_shape) != 4:
        return None, f"SDPA expects 4D key, got {len(k_shape)}D"
    if len(v_shape) != 4:
        return None, f"SDPA expects 4D value, got {len(v_shape)}D"
    
    B_q, H_q, T_q, D_q = q_shape
    B_k, H_k, T_k, D_k = k_shape
    B_v, H_v, T_v, D_v = v_shape
    
    # Validate batch and head dims match
    if not dims_compatible(B_q, B_k) or not dims_compatible(B_q, B_v):
        return None, f"SDPA batch size mismatch: Q={B_q}, K={B_k}, V={B_v}"
    if not dims_compatible(H_q, H_k) or not dims_compatible(H_q, H_v):
        return None, f"SDPA head count mismatch: Q={H_q}, K={H_k}, V={H_v}"
    
    # K and V sequence length must match
    if not dims_compatible(T_k, T_v):
        return None, f"SDPA sequence length mismatch: K={T_k}, V={T_v}"
    
    # Q and K head dimension must match
    if not dims_compatible(D_q, D_k):
        return None, f"SDPA head dim mismatch: Q={D_q}, K={D_k}"
    
    # Output has Q's sequence length and V's value dimension
    output_shape = (B_q, H_q, T_q, D_v)
    return output_shape, None


def propagate_multihead_attention(
    input_shape: Tuple[Any, ...],
    embed_dim: int,
    num_heads: int,
    batch_first: bool = False,
) -> Tuple[Optional[Tuple[Any, ...]], Optional[str]]:
    """Propagate shape through nn.MultiheadAttention.
    
    MHA signatures:
        batch_first=False: (T, B, E) → (T, B, E)
        batch_first=True:  (B, T, E) → (B, T, E)
    
    Constraints:
        - embed_dim must be divisible by num_heads
        - Input last dim must equal embed_dim
    
    Args:
        input_shape: Input tensor shape
        embed_dim: Embedding dimension
        num_heads: Number of attention heads
        batch_first: Whether batch is first dimension
        
    Returns:
        (output_shape, error_message) tuple
    """
    # Check embed_dim divisibility
    if embed_dim % num_heads != 0:
        return None, f"MHA embed_dim={embed_dim} must be divisible by num_heads={num_heads}"
    
    # Check input shape
    if len(input_shape) != 3:
        return None, f"MHA expects 3D input, got {len(input_shape)}D"
    
    if batch_first:
        B, T, E = input_shape
        expected_e = embed_dim
    else:
        T, B, E = input_shape
        expected_e = embed_dim
    
    # Validate embedding dimension
    if not dim_equals(E, expected_e):
        return None, f"MHA expects embed_dim={embed_dim}, got {E}"
    
    # Output shape is same as input shape
    output_shape = input_shape
    return output_shape, None


def propagate_mha_with_separate_kv(
    q_shape: Tuple[Any, ...],
    k_shape: Tuple[Any, ...],
    v_shape: Tuple[Any, ...],
    embed_dim: int,
    num_heads: int,
    batch_first: bool = False,
) -> Tuple[Optional[Tuple[Any, ...]], Optional[str]]:
    """Propagate shape through MHA with separate key/value.
    
    Used for cross-attention where Q comes from decoder, K/V from encoder.
    
    Args:
        q_shape: Query shape
        k_shape: Key shape
        v_shape: Value shape
        embed_dim: Embedding dimension
        num_heads: Number of heads
        batch_first: Whether batch is first
        
    Returns:
        (output_shape, error_message) tuple
    """
    # Check all are 3D
    if len(q_shape) != 3 or len(k_shape) != 3 or len(v_shape) != 3:
        return None, "MHA cross-attention expects 3D inputs"
    
    # Check embed_dim divisibility
    if embed_dim % num_heads != 0:
        return None, f"MHA embed_dim={embed_dim} must be divisible by num_heads={num_heads}"
    
    if batch_first:
        B_q, T_q, E_q = q_shape
        B_k, T_k, E_k = k_shape
        B_v, T_v, E_v = v_shape
    else:
        T_q, B_q, E_q = q_shape
        T_k, B_k, E_k = k_shape
        T_v, B_v, E_v = v_shape
    
    # Validate batch sizes
    if not dims_compatible(B_q, B_k) or not dims_compatible(B_q, B_v):
        return None, "MHA batch size mismatch"
    
    # Validate K/V sequence lengths match
    if not dims_compatible(T_k, T_v):
        return None, "MHA K/V sequence length mismatch"
    
    # Validate embedding dimensions
    if not dim_equals(E_q, embed_dim):
        return None, f"MHA Q embed_dim mismatch: expected {embed_dim}, got {E_q}"
    if not dim_equals(E_k, embed_dim):
        return None, f"MHA K embed_dim mismatch: expected {embed_dim}, got {E_k}"
    if not dim_equals(E_v, embed_dim):
        return None, f"MHA V embed_dim mismatch: expected {embed_dim}, got {E_v}"
    
    # Output has Q's sequence length
    if batch_first:
        output_shape = (B_q, T_q, embed_dim)
    else:
        output_shape = (T_q, B_q, embed_dim)
    
    return output_shape, None


def dims_compatible(a: Any, b: Any) -> bool:
    """Check if two dimensions are compatible (equal or one/both symbolic).
    
    Args:
        a: First dimension
        b: Second dimension
        
    Returns:
        True if compatible
    """
    if isinstance(a, int) and isinstance(b, int):
        return a == b
    # Symbolic dimensions are optimistically compatible
    return True


def dim_equals(a: Any, expected: int) -> bool:
    """Check if dimension equals expected value.
    
    Args:
        a: Dimension to check
        expected: Expected value
        
    Returns:
        True if equal or a is symbolic
    """
    if isinstance(a, int):
        return a == expected
    # Symbolic: optimistically assume match
    return True
