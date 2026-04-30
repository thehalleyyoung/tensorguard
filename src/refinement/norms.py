"""Normalization layer shape propagation.

Handles:
- nn.LayerNorm
- nn.RMSNorm (relatively new, added in PyTorch 2.0+)
"""

from __future__ import annotations
from typing import Tuple, Optional, Any, List


def propagate_layernorm(
    input_shape: Tuple[Any, ...],
    normalized_shape: Tuple[int, ...],
) -> Tuple[Optional[Tuple[Any, ...]], Optional[str]]:
    """Propagate shape through nn.LayerNorm.
    
    LayerNorm normalizes over the last len(normalized_shape) dimensions.
    The output shape is identical to the input shape.
    
    Validation: input_shape[-N:] must match normalized_shape, where
    N = len(normalized_shape).
    
    Args:
        input_shape: Input tensor shape
        normalized_shape: Tuple of normalized dimensions
        
    Returns:
        (output_shape, error_message) tuple
        
    Examples:
        LayerNorm([768])  with input (B, T, 768) → (B, T, 768) ✓
        LayerNorm([12, 64]) with input (B, T, 12, 64) → (B, T, 12, 64) ✓
        LayerNorm([768])  with input (B, T, 512) → ERROR
    """
    n = len(normalized_shape)
    if n == 0:
        return None, "LayerNorm normalized_shape cannot be empty"
    
    if len(input_shape) < n:
        return None, (
            f"LayerNorm normalized_shape has {n} dims, "
            f"but input has only {len(input_shape)} dims"
        )
    
    # Check trailing dims match
    input_trailing = input_shape[-n:]
    for i, (inp_dim, norm_dim) in enumerate(zip(input_trailing, normalized_shape)):
        if isinstance(inp_dim, int) and inp_dim != norm_dim:
            return None, (
                f"LayerNorm dimension mismatch at position {-n+i}: "
                f"input has {inp_dim}, normalized_shape expects {norm_dim}"
            )
        # If symbolic, optimistically assume match
    
    # Output shape is same as input
    return input_shape, None


def propagate_rmsnorm(
    input_shape: Tuple[Any, ...],
    normalized_shape: Tuple[int, ...],
) -> Tuple[Optional[Tuple[Any, ...]], Optional[str]]:
    """Propagate shape through nn.RMSNorm.
    
    RMSNorm (Root Mean Square Normalization) is similar to LayerNorm but
    without mean centering. It normalizes over the last dimensions.
    
    The output shape is identical to the input shape.
    
    Args:
        input_shape: Input tensor shape
        normalized_shape: Tuple of normalized dimensions
        
    Returns:
        (output_shape, error_message) tuple
    """
    # RMSNorm has the same shape semantics as LayerNorm
    return propagate_layernorm(input_shape, normalized_shape)


def propagate_groupnorm(
    input_shape: Tuple[Any, ...],
    num_groups: int,
    num_channels: int,
) -> Tuple[Optional[Tuple[Any, ...]], Optional[str]]:
    """Propagate shape through nn.GroupNorm.
    
    GroupNorm expects input (N, C, *) where C = num_channels.
    Divides C into num_groups and normalizes within each group.
    
    Constraint: num_channels must be divisible by num_groups.
    
    Args:
        input_shape: Input tensor shape
        num_groups: Number of groups
        num_channels: Number of channels
        
    Returns:
        (output_shape, error_message) tuple
    """
    if len(input_shape) < 2:
        return None, "GroupNorm expects at least 2D input (N, C, ...)"
    
    # Check channel dimension
    C = input_shape[1]
    if isinstance(C, int):
        if C != num_channels:
            return None, (
                f"GroupNorm expects {num_channels} channels, got {C}"
            )
    
    # Check divisibility
    if num_channels % num_groups != 0:
        return None, (
            f"GroupNorm num_channels={num_channels} must be divisible "
            f"by num_groups={num_groups}"
        )
    
    # Output shape is same as input
    return input_shape, None


def propagate_instancenorm(
    input_shape: Tuple[Any, ...],
    num_features: int,
    ndim: int = 2,
) -> Tuple[Optional[Tuple[Any, ...]], Optional[str]]:
    """Propagate shape through nn.InstanceNorm{1d,2d,3d}.
    
    InstanceNorm normalizes over spatial dimensions per channel.
    
    Args:
        input_shape: Input tensor shape
        num_features: Number of channels
        ndim: Spatial dimensions (1, 2, or 3)
        
    Returns:
        (output_shape, error_message) tuple
    """
    expected_ndim = ndim + 2  # (N, C, *spatial)
    if len(input_shape) != expected_ndim:
        return None, (
            f"InstanceNorm{ndim}d expects {expected_ndim}D input, "
            f"got {len(input_shape)}D"
        )
    
    # Check channel dimension
    C = input_shape[1]
    if isinstance(C, int):
        if C != num_features:
            return None, (
                f"InstanceNorm{ndim}d expects {num_features} channels, got {C}"
            )
    
    # Output shape is same as input
    return input_shape, None


def propagate_batchnorm(
    input_shape: Tuple[Any, ...],
    num_features: int,
    ndim: int = 2,
) -> Tuple[Optional[Tuple[Any, ...]], Optional[str]]:
    """Propagate shape through nn.BatchNorm{1d,2d,3d}.
    
    BatchNorm normalizes over batch dimension per channel.
    
    Args:
        input_shape: Input tensor shape
        num_features: Number of channels
        ndim: Spatial dimensions (1, 2, or 3)
        
    Returns:
        (output_shape, error_message) tuple
    """
    expected_ndim = ndim + 2  # (N, C, *spatial)
    if ndim == 1:
        # BatchNorm1d can also accept 2D (N, C) or 3D (N, C, L)
        if len(input_shape) not in (2, 3):
            return None, (
                f"BatchNorm1d expects 2D or 3D input, got {len(input_shape)}D"
            )
    else:
        if len(input_shape) != expected_ndim:
            return None, (
                f"BatchNorm{ndim}d expects {expected_ndim}D input, "
                f"got {len(input_shape)}D"
            )
    
    # Check channel dimension
    C = input_shape[1]
    if isinstance(C, int):
        if C != num_features:
            return None, (
                f"BatchNorm{ndim}d expects {num_features} channels, got {C}"
            )
    
    # Output shape is same as input
    return input_shape, None
