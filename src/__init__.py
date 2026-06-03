"""
GuardHarvest: Find bugs in Python code with zero annotations.

Flow-sensitive abstract interpreter that harvests programmer-written guards
(isinstance, is not None, comparisons) as implicit refinement type predicates.
"""

__version__ = "0.1.0"

from .api import (
    analyze,
    analyze_file,
    analyze_directory,
    analyze_function,
    quick_check,
    verify_architecture,
    AnalysisResult,
    Bug,
    BugCategory,
    SourceLocation,
)
from .runtime_check import checked, TensorGuardCheckError
from .safe_loader import (
    verify_file_safely,
    verify_source_safely,
    is_static_only_source,
)
from .distributions_verify import verify_distribution, verify_log_prob
from .complex_verify import verify_fft, verify_view_as_complex, verify_view_as_real
from .embedding_bag_verify import (
    EmbeddingBagVerdict,
    TorchRecJaggedSpec,
    verify_embedding_bag,
    verify_torchrec_embedding_bag,
)
from .linalg_verify import (
    verify_linalg,
    verify_linalg_cholesky,
    verify_linalg_eig,
    verify_linalg_inv,
    verify_linalg_qr,
    verify_linalg_solve,
    verify_linalg_svd,
)
from .named_tensor_verify import verify_align_to, verify_named_tensor_source, verify_refine_names
from .quantization_verify import (
    QuantizationIssue,
    QuantizationVerdict,
    verify_quantization,
    verify_quantization_eager,
    verify_quantization_fx,
)
from .mixed_precision_verify import (
    AutocastTraceEntry,
    MixedPrecisionIssue,
    MixedPrecisionVerdict,
    verify_mixed_precision,
    verify_mixed_precision_fx,
)
from .sparse_verify import (
    verify_sparse_addmm,
    verify_sparse_bsc,
    verify_sparse_bsr,
    verify_sparse_coalesce,
    verify_sparse_coo,
    verify_sparse_csc,
    verify_sparse_csr,
    verify_sparse_layout_conversion,
    verify_sparse_mm,
    verify_sparse_sampled_addmm,
    verify_sparse_softmax,
    verify_sparse_to_dense,
)
from .grid_sample_verify import verify_affine_grid, verify_grid_sample
from .loss_verify import LossVerdict, verify_loss
from .mha_verify import verify_multihead_attention
from .torchvision_v2_verify import TransformVerdict, verify_torchvision_v2_transform
from .vmap_verify import verify_vmap
from .func_autodiff_verify import (
    verify_func_autodiff,
    verify_func_grad,
    verify_func_jacfwd,
    verify_func_jacrev,
    verify_func_jvp,
    verify_func_vjp,
)
from .distributed_verification import (
    DistributedPlacement,
    DTensorPlacement,
    DTensorSpec,
    DTensorVerifier,
    FSDP2Config,
    FSDP2Verifier,
    ParameterShardingSpec,
    ParameterShardingStrategy,
    ParameterShardingVerifier,
    verify_dtensor_specs,
    verify_parameter_sharding,
)

__all__ = [
    "analyze",
    "analyze_file",
    "analyze_directory",
    "analyze_function",
    "quick_check",
    "verify_architecture",
    "verify_file_safely",
    "verify_source_safely",
    "is_static_only_source",
    "AnalysisResult",
    "Bug",
    "BugCategory",
    "SourceLocation",
    "checked",
    "TensorGuardCheckError",
    "verify_distribution",
    "verify_log_prob",
    "verify_fft",
    "verify_view_as_real",
    "verify_view_as_complex",
    "EmbeddingBagVerdict",
    "TorchRecJaggedSpec",
    "verify_embedding_bag",
    "verify_torchrec_embedding_bag",
    "verify_linalg",
    "verify_linalg_inv",
    "verify_linalg_cholesky",
    "verify_linalg_eig",
    "verify_linalg_qr",
    "verify_linalg_svd",
    "verify_linalg_solve",
    "verify_refine_names",
    "verify_align_to",
    "verify_named_tensor_source",
    "QuantizationIssue",
    "QuantizationVerdict",
    "verify_quantization",
    "verify_quantization_eager",
    "verify_quantization_fx",
    "AutocastTraceEntry",
    "MixedPrecisionIssue",
    "MixedPrecisionVerdict",
    "verify_mixed_precision",
    "verify_mixed_precision_fx",
    "verify_sparse_coo",
    "verify_sparse_csr",
    "verify_sparse_csc",
    "verify_sparse_bsr",
    "verify_sparse_bsc",
    "verify_sparse_mm",
    "verify_sparse_addmm",
    "verify_sparse_sampled_addmm",
    "verify_sparse_softmax",
    "verify_sparse_coalesce",
    "verify_sparse_to_dense",
    "verify_sparse_layout_conversion",
    "verify_grid_sample",
    "verify_affine_grid",
    "LossVerdict",
    "verify_loss",
    "verify_multihead_attention",
    "TransformVerdict",
    "verify_torchvision_v2_transform",
    "verify_vmap",
    "verify_func_autodiff",
    "verify_func_grad",
    "verify_func_jacrev",
    "verify_func_jacfwd",
    "verify_func_jvp",
    "verify_func_vjp",
    "DistributedPlacement",
    "DTensorPlacement",
    "DTensorSpec",
    "DTensorVerifier",
    "FSDP2Config",
    "FSDP2Verifier",
    "ParameterShardingSpec",
    "ParameterShardingStrategy",
    "ParameterShardingVerifier",
    "verify_dtensor_specs",
    "verify_parameter_sharding",
    "__version__",
]
