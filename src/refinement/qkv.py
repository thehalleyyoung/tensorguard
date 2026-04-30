"""QKV unpacking patterns for attention mechanisms.

Handles tuple unpacking from:
- split(size, dim=-1) → multiple tensors
- chunk(n, dim=-1) → n tensors 
- view(..., 3, ...).unbind(dim) → 3 tensors
"""

from __future__ import annotations
from typing import List, Tuple, Optional, Any, Dict
import ast


def handle_split_unpack(
    target_elts: List[ast.expr],
    value: ast.Call,
    dim: int = -1
) -> Optional[List[Tuple[str, Dict[str, Any]]]]:
    """Handle q, k, v = tensor.split(size, dim=dim) unpacking.
    
    Args:
        target_elts: LHS tuple elements (q, k, v names)
        value: The split() Call node
        dim: Split dimension
        
    Returns:
        List of (output_name, params_dict) for each split output, or None
    """
    if not isinstance(value.func, ast.Attribute) or value.func.attr != "split":
        return None
    
    n = len(target_elts)
    if n == 0:
        return None
    
    results = []
    for i, elt in enumerate(target_elts):
        if isinstance(elt, ast.Name) and elt.id != "_":
            name = elt.id
        else:
            name = f"__split_{i}"
        
        params = {
            "dim": dim,
            "split_index": i,
            "n_outputs": n,
        }
        results.append((name, params))
    
    return results


def handle_chunk_unpack(
    target_elts: List[ast.expr],
    value: ast.Call,
    chunks: int,
    dim: int = -1
) -> Optional[List[Tuple[str, Dict[str, Any]]]]:
    """Handle q, k, v = tensor.chunk(3, dim=dim) unpacking.
    
    Args:
        target_elts: LHS tuple elements
        value: The chunk() Call node
        chunks: Number of chunks
        dim: Split dimension
        
    Returns:
        List of (output_name, params_dict) for each chunk output, or None
    """
    if not isinstance(value.func, ast.Attribute) or value.func.attr != "chunk":
        return None
    
    n = len(target_elts)
    if n == 0:
        return None
    
    results = []
    for i, elt in enumerate(target_elts):
        if isinstance(elt, ast.Name) and elt.id != "_":
            name = elt.id
        else:
            name = f"__chunk_{i}"
        
        params = {
            "dim": dim,
            "split_index": i,
            "n_outputs": n,
            "chunks": chunks,
        }
        results.append((name, params))
    
    return results


def handle_unbind_unpack(
    target_elts: List[ast.expr],
    value: ast.Call,
    dim: int
) -> Optional[List[Tuple[str, Dict[str, Any]]]]:
    """Handle q, k, v = tensor.view(..., 3, ...).unbind(dim) unpacking.
    
    This is the pattern: qkv.view(B, T, 3, H, D).unbind(2)
    which yields 3 tensors of shape (B, T, H, D).
    
    Args:
        target_elts: LHS tuple elements
        value: The unbind() Call node (might be chained after view())
        dim: Unbind dimension
        
    Returns:
        List of (output_name, params_dict) for each unbound output, or None
    """
    if not isinstance(value.func, ast.Attribute) or value.func.attr != "unbind":
        return None
    
    n = len(target_elts)
    if n == 0:
        return None
    
    results = []
    for i, elt in enumerate(target_elts):
        if isinstance(elt, ast.Name) and elt.id != "_":
            name = elt.id
        else:
            name = f"__unbind_{i}"
        
        params = {
            "dim": dim,
            "unbind_index": i,
            "n_outputs": n,
        }
        results.append((name, params))
    
    return results


def propagate_split_shape(
    input_shape: Tuple[Any, ...],
    dim: int,
    split_size: Any,
    n_outputs: int,
    split_index: int
) -> Optional[Tuple[Any, ...]]:
    """Compute output shape for one split output.
    
    Args:
        input_shape: Input tensor shape
        dim: Split dimension
        split_size: Size of each split (or total to divide)
        n_outputs: Number of output tensors
        split_index: Index of this output
        
    Returns:
        Output shape tuple, or None if invalid
    """
    ndim = len(input_shape)
    if dim < 0:
        dim = ndim + dim
    if dim < 0 or dim >= ndim:
        return None
    
    new_shape = list(input_shape)
    new_shape[dim] = split_size
    return tuple(new_shape)


def propagate_chunk_shape(
    input_shape: Tuple[Any, ...],
    dim: int,
    chunks: int,
    split_index: int
) -> Optional[Tuple[Any, ...]]:
    """Compute output shape for one chunk output.
    
    For chunk(n, dim), each output has size ceil(input_dim / n) along dim.
    Adds a divisibility constraint via Z3 if input_dim is symbolic.
    
    Args:
        input_shape: Input tensor shape
        dim: Split dimension
        chunks: Number of chunks
        split_index: Index of this output
        
    Returns:
        Output shape tuple, or None if invalid
    """
    ndim = len(input_shape)
    if dim < 0:
        dim = ndim + dim
    if dim < 0 or dim >= ndim:
        return None
    
    input_dim = input_shape[dim]
    
    # If concrete, compute chunk size
    if isinstance(input_dim, int):
        chunk_size = (input_dim + chunks - 1) // chunks
        new_shape = list(input_shape)
        new_shape[dim] = chunk_size
        return tuple(new_shape)
    
    # If symbolic, represent as division
    new_shape = list(input_shape)
    if isinstance(input_dim, str):
        # Symbolic: need divisibility constraint
        new_shape[dim] = f"({input_dim}+{chunks}-1)//{chunks}"
    else:
        # Unknown
        new_shape[dim] = input_dim
    return tuple(new_shape)


def propagate_unbind_shape(
    input_shape: Tuple[Any, ...],
    dim: int,
    unbind_index: int
) -> Optional[Tuple[Any, ...]]:
    """Compute output shape for one unbind output.
    
    Unbind removes dimension 'dim', so output is one dimension smaller.
    
    Args:
        input_shape: Input tensor shape
        dim: Unbind dimension
        unbind_index: Index of this output
        
    Returns:
        Output shape tuple with dim removed, or None if invalid
    """
    ndim = len(input_shape)
    if dim < 0:
        dim = ndim + dim
    if dim < 0 or dim >= ndim:
        return None
    
    # Remove dimension
    new_shape = list(input_shape)
    del new_shape[dim]
    return tuple(new_shape)
