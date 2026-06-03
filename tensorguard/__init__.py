"""TensorGuard — the importable top-level package (Step 161).

The verifier's implementation lives in the ``src`` package (kept stable for the
console-script and ``pytest11`` entry points). This module is the *public import
surface* a ``pip install tensorguard`` user reaches for::

    from tensorguard import verify_architecture, analyze, AnalysisResult

It re-exports exactly the stability-guaranteed names from :mod:`src` (the set
pinned by ``tests/test_api_stability.py`` and documented in
``DEPRECATION_POLICY.md``), plus the Phase-7 integration entry points
(``guarded_compile``, ``make_tensorguard_backend``, ``verify_module``,
``TensorGuardViolation``) so the adoption paths in the README work against the
real package, not a private module path.

Submodules remain reachable as ``tensorguard.api``, ``tensorguard.torch`` etc.
for callers that want the full surface.
"""

from __future__ import annotations

from src import (  # noqa: F401  (re-export)
    AnalysisResult,
    Bug,
    BugCategory,
    SourceLocation,
    TensorGuardCheckError,
    __version__,
    analyze,
    analyze_directory,
    analyze_file,
    analyze_function,
    checked,
    is_static_only_source,
    quick_check,
    verify_architecture,
    verify_file_safely,
    verify_source_safely,
)
from src.torch_integration import (  # noqa: F401  (re-export)
    AOTPackageGateResult,
    AOTPackageIssue,
    ONNXExportGateResult,
    ONNXExportIssue,
    ONNXLoweredOp,
    TensorGuardAOTPackageError,
    TensorGuardONNXExportError,
    TensorGuardViolation,
    guarded_aot_package,
    guarded_compile,
    guarded_onnx_export,
    make_tensorguard_backend,
    verify_onnx_export_contract,
    verify_aot_package_contract,
    verify_exported_program,
    verify_module,
)
from src.gguf_export import (  # noqa: F401  (re-export)
    GGUFExportGateResult,
    GGUFExportIssue,
    GGUFTensorInfo,
    TensorGuardGGUFExportError,
    guarded_gguf_export,
    verify_gguf_export_contract,
)

from src.einops_verify import verify_einops  # noqa: F401  (re-export)
from src.einops_source import verify_einops_source  # noqa: F401  (re-export)
from src.distributions_verify import (  # noqa: F401  (re-export)
    verify_distribution,
    verify_log_prob,
)
from src.complex_verify import (  # noqa: F401  (re-export)
    verify_fft,
    verify_view_as_complex,
    verify_view_as_real,
)
from src.embedding_bag_verify import (  # noqa: F401  (re-export)
    EmbeddingBagVerdict,
    TorchRecJaggedSpec,
    verify_embedding_bag,
    verify_torchrec_embedding_bag,
)
from src.linalg_verify import (  # noqa: F401  (re-export)
    verify_linalg,
    verify_linalg_cholesky,
    verify_linalg_eig,
    verify_linalg_inv,
    verify_linalg_qr,
    verify_linalg_solve,
    verify_linalg_svd,
)
from src.named_tensor_verify import (  # noqa: F401  (re-export)
    verify_align_to,
    verify_named_tensor_source,
    verify_refine_names,
)
from src.quantization_verify import (  # noqa: F401  (re-export)
    QuantizationIssue,
    QuantizationVerdict,
    verify_quantization,
    verify_quantization_eager,
    verify_quantization_fx,
)
from src.mixed_precision_verify import (  # noqa: F401  (re-export)
    AutocastTraceEntry,
    MixedPrecisionIssue,
    MixedPrecisionVerdict,
    verify_mixed_precision,
    verify_mixed_precision_fx,
)
from src.sparse_verify import (  # noqa: F401  (re-export)
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
from src.grid_sample_verify import (  # noqa: F401  (re-export)
    verify_affine_grid,
    verify_grid_sample,
)
from src.loss_verify import LossVerdict, verify_loss  # noqa: F401  (re-export)
from src.mha_verify import verify_multihead_attention  # noqa: F401  (re-export)
from src.torchvision_v2_verify import (  # noqa: F401  (re-export)
    TransformVerdict,
    verify_torchvision_v2_transform,
)
from src.vmap_verify import verify_vmap  # noqa: F401  (re-export)
from src.func_autodiff_verify import (  # noqa: F401  (re-export)
    verify_func_autodiff,
    verify_func_grad,
    verify_func_jacfwd,
    verify_func_jacrev,
    verify_func_jvp,
    verify_func_vjp,
)
from src.distributed_verification import (  # noqa: F401  (re-export)
    DistributedPlacement,
    DTensorPlacement,
    DTensorSpec,
    DTensorVerifier,
    FSDP2Config,
    FSDP2Verifier,
    ParameterShardingSpec,
    ParameterShardingStrategy,
    ParameterShardingVerifier,
    PipelineBoundarySpec,
    PipelineParallelVerifier,
    PipelineStageSpec,
    verify_dtensor_specs,
    verify_parameter_sharding,
    verify_pipeline_boundaries,
)
from src.optimizer_state_verify import (  # noqa: F401  (re-export)
    OptimizerStateIssue,
    OptimizerStateShard,
    OptimizerStateVerificationResult,
    TensorGuardOptimizerStateError,
    guarded_optimizer_load_state_dict,
    verify_optimizer_state,
)
from src.checkpoint_verify import (  # noqa: F401  (re-export)
    CheckpointIssue,
    CheckpointVerificationResult,
    TensorGuardCheckpointError,
    TensorParallelCheckpointShard,
    guarded_load_state_dict,
    verify_checkpoint_state_dict,
)

# Lazily importable submodule aliases (``import tensorguard.api`` etc.).
from src import api as api  # noqa: F401
from . import torch as torch  # noqa: F401

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
    "verify_module",
    "guarded_compile",
    "make_tensorguard_backend",
    "guarded_onnx_export",
    "verify_exported_program",
    "guarded_gguf_export",
    "verify_gguf_export_contract",
    "verify_aot_package_contract",
    "verify_onnx_export_contract",
    "guarded_aot_package",
    "ONNXExportGateResult",
    "ONNXExportIssue",
    "ONNXLoweredOp",
    "GGUFExportGateResult",
    "GGUFExportIssue",
    "GGUFTensorInfo",
    "TensorGuardONNXExportError",
    "TensorGuardGGUFExportError",
    "verify_einops",
    "verify_einops_source",
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
    "PipelineBoundarySpec",
    "PipelineParallelVerifier",
    "PipelineStageSpec",
    "verify_dtensor_specs",
    "verify_parameter_sharding",
    "verify_pipeline_boundaries",
    "OptimizerStateIssue",
    "OptimizerStateShard",
    "OptimizerStateVerificationResult",
    "TensorGuardOptimizerStateError",
    "guarded_optimizer_load_state_dict",
    "verify_optimizer_state",
    "CheckpointIssue",
    "CheckpointVerificationResult",
    "TensorGuardCheckpointError",
    "TensorParallelCheckpointShard",
    "guarded_load_state_dict",
    "verify_checkpoint_state_dict",
    "TensorGuardViolation",
    "TensorGuardAOTPackageError",
    "AOTPackageIssue",
    "AOTPackageGateResult",
    "api",
    "torch",
    "__version__",
]
