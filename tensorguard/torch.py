"""PyTorch lifecycle gates exposed as ``tensorguard.torch``."""

from __future__ import annotations

from src.torch_integration import (
    AOTPackageGateResult,
    AOTPackageIssue,
    ONNXExportGateResult,
    ONNXExportIssue,
    ONNXLoweredOp,
    ONNXShapeRoundTripCheck,
    TensorGuardAOTPackageError,
    TensorGuardDynamicShapeError,
    TensorGuardONNXExportError,
    TensorGuardONNXShapeInferenceError,
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

__all__ = [
    "AOTPackageGateResult",
    "AOTPackageIssue",
    "ONNXExportGateResult",
    "ONNXExportIssue",
    "ONNXLoweredOp",
    "ONNXShapeRoundTripCheck",
    "TensorGuardAOTPackageError",
    "TensorGuardDynamicShapeError",
    "TensorGuardONNXExportError",
    "TensorGuardONNXShapeInferenceError",
    "TensorGuardViolation",
    "guarded_aot_package",
    "guarded_compile",
    "guarded_onnx_export",
    "make_tensorguard_backend",
    "verify_onnx_export_contract",
    "verify_aot_package_contract",
    "verify_exported_program",
    "verify_module",
]
