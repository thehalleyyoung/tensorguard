"""
Refinement-typed shape calculus for TensorGuard.

This package implements the theoretical foundation for sound static shape
verification using refinement types over tensor shapes.

Track C refinements (NeurIPS 2026 revision):
- Symbolic config attributes (config.hidden_size patterns)
- Tuple unpacking from split/chunk/unbind operations
- Reshape with -1 inference using Z3
- Attention mechanisms (SDPA, MultiheadAttention)
- Normalization layers (LayerNorm, RMSNorm)
"""

# Track C refinements
try:
    from .symbolic_config import (
        symbolic_config,
        detect_symbolic_config_attrs,
        resolve_config_attr,
        make_expression_symbolic,
    )
    from .qkv import (
        handle_split_unpack,
        handle_chunk_unpack,
        handle_unbind_unpack,
    )
    from .reshape import (
        infer_reshape_minus_one,
        validate_reshape_with_z3,
    )
    from .attention import (
        propagate_scaled_dot_product_attention,
        propagate_multihead_attention,
    )
    from .norms import (
        propagate_layernorm,
        propagate_rmsnorm,
    )
    _HAS_TRACK_C = True
except ImportError as e:
    _HAS_TRACK_C = False
    print(f"Track C refinements not available: {e}")

# Try to import existing calculus module if it exists
try:
    from .calculus import (
        RefinementType,
        Gamma,
        entails,
        subject_reduction_check,
        type_matmul,
        type_conv2d,
        type_broadcast,
        type_view,
        type_reshape,
        type_split,
        type_cat,
        type_permute,
        type_transpose,
        type_einsum,
        type_indexing,
        type_scatter,
        type_gather,
    )
    _HAS_CALCULUS = True
except ImportError:
    _HAS_CALCULUS = False

__all__ = []

if _HAS_CALCULUS:
    __all__.extend([
        "RefinementType",
        "Gamma",
        "entails",
        "subject_reduction_check",
        "type_matmul",
        "type_conv2d",
        "type_broadcast",
        "type_view",
        "type_reshape",
        "type_split",
        "type_cat",
        "type_permute",
        "type_transpose",
        "type_einsum",
        "type_indexing",
        "type_scatter",
        "type_gather",
    ])

if _HAS_TRACK_C:
    __all__.extend([
        "symbolic_config",
        "detect_symbolic_config_attrs",
        "resolve_config_attr",
        "make_expression_symbolic",
        "handle_split_unpack",
        "handle_chunk_unpack",
        "handle_unbind_unpack",
        "infer_reshape_minus_one",
        "validate_reshape_with_z3",
        "propagate_scaled_dot_product_attention",
        "propagate_multihead_attention",
        "propagate_layernorm",
        "propagate_rmsnorm",
    ])
