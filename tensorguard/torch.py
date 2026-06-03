"""PyTorch lifecycle gates exposed as ``tensorguard.torch``."""

from __future__ import annotations

from src.torch_integration import (
    AOTPackageGateResult,
    AOTPackageIssue,
    TensorGuardAOTPackageError,
    TensorGuardDynamicShapeError,
    TensorGuardViolation,
    guarded_aot_package,
    guarded_compile,
    guarded_onnx_export,
    make_tensorguard_backend,
    verify_aot_package_contract,
    verify_exported_program,
    verify_module,
)

__all__ = [
    "AOTPackageGateResult",
    "AOTPackageIssue",
    "TensorGuardAOTPackageError",
    "TensorGuardDynamicShapeError",
    "TensorGuardViolation",
    "guarded_aot_package",
    "guarded_compile",
    "guarded_onnx_export",
    "make_tensorguard_backend",
    "verify_aot_package_contract",
    "verify_exported_program",
    "verify_module",
]
