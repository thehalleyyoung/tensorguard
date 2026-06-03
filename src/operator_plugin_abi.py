"""Stable ABI for trusted third-party operator-theory plugins.

The community stub registry is intentionally declarative and safe for untrusted
pull requests.  This module covers the other extension path: a library author can
explicitly import a trusted Python package that provides executable operator
theories, validate its contracts against conformance cases, and install the
resulting transfers into TensorGuard's shape-stub registry.

Security boundary: TensorGuard never auto-discovers or auto-imports plugins.
Importing a plugin is the caller's trust decision; this ABI only validates the
shape-transfer contract after that import has already happened.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Tuple

from src.shape_stub_registry import TransferFn, register_shape_stub
from src.tensor_shapes import ShapeDim, TensorShape

ABI_VERSION = "1.0"
SUPPORTED_ABI_MAJOR = 1


@dataclass(frozen=True)
class PluginProvenance:
    """Auditable identity for the package that owns an operator theory."""

    package: str
    version: str
    source_url: str
    license: str
    author: str


@dataclass(frozen=True)
class SecurityReview:
    """Human-review attestations for trusted executable plugin code.

    These booleans are not a sandbox.  They document the review checklist that a
    maintainer or integrator completed before explicitly importing the plugin.
    """

    reviewed_by: str
    reviewed_on: str
    no_import_side_effects: bool
    no_network: bool
    no_filesystem_writes: bool
    deterministic: bool
    no_model_execution: bool
    notes: str = ""


@dataclass(frozen=True)
class ConformanceCase:
    """One executable example proving a contract behaves as advertised."""

    input_shape: Tuple[Any, ...]
    ctor_args: Tuple[Any, ...] = ()
    ctor_kwargs: Dict[str, Any] = field(default_factory=dict)
    expected_output: Optional[Tuple[Any, ...]] = None
    expected_error_contains: Optional[str] = None


@dataclass(frozen=True)
class OperatorTheoryContract:
    """Versioned shape-transfer contract supplied by a trusted plugin."""

    class_name: str
    transfer: TransferFn
    arg_names: Tuple[str, ...] = ()
    defaults: Dict[str, Any] = field(default_factory=dict)
    conformance: Tuple[ConformanceCase, ...] = ()
    provenance: Optional[PluginProvenance] = None
    security_review: Optional[SecurityReview] = None
    abi_version: str = ABI_VERSION
    summary: str = ""


@dataclass(frozen=True)
class PluginValidationReport:
    """Validation result for one operator-theory contract."""

    class_name: str
    ok: bool
    errors: Tuple[str, ...] = ()
    cases_checked: int = 0

    def __bool__(self) -> bool:
        return self.ok


def _parse_major(version: str) -> Optional[int]:
    parts = str(version).split(".")
    if not parts or not parts[0].isdigit():
        return None
    return int(parts[0])


def is_abi_compatible(version: str) -> bool:
    """Return whether *version* is compatible with this runtime's ABI major."""

    return _parse_major(version) == SUPPORTED_ABI_MAJOR


def _dim_from_spec(spec: Any) -> ShapeDim:
    if isinstance(spec, bool):
        raise ValueError(f"invalid dimension {spec!r}")
    if isinstance(spec, int):
        return ShapeDim(spec)
    if isinstance(spec, str) and spec:
        return ShapeDim(spec)
    raise ValueError(f"invalid dimension {spec!r}; expected int or non-empty str")


def _shape_from_spec(spec: Tuple[Any, ...]) -> TensorShape:
    if not isinstance(spec, tuple) or not spec:
        raise ValueError(f"shape must be a non-empty tuple, got {spec!r}")
    return TensorShape(tuple(_dim_from_spec(d) for d in spec))


def _bind_params(contract: OperatorTheoryContract, case: ConformanceCase) -> Dict[str, Any]:
    params: Dict[str, Any] = dict(contract.defaults)
    for i, value in enumerate(case.ctor_args):
        if i < len(contract.arg_names):
            params[contract.arg_names[i]] = value
    params.update(dict(case.ctor_kwargs))
    return params


def _shape_values(shape: Optional[TensorShape]) -> Tuple[Any, ...]:
    if shape is None:
        return ()
    return tuple(dim.value for dim in shape.dims)


def _validate_conformance(contract: OperatorTheoryContract, errors: List[str]) -> int:
    if not contract.conformance:
        errors.append("at least one conformance case is required")
        return 0

    checked = 0
    for idx, case in enumerate(contract.conformance):
        if case.expected_output is None and case.expected_error_contains is None:
            errors.append(
                f"conformance[{idx}] must set expected_output or expected_error_contains"
            )
            continue
        if case.expected_output is not None and case.expected_error_contains is not None:
            errors.append(
                f"conformance[{idx}] must not set both output and error expectations"
            )
            continue
        try:
            input_shape = _shape_from_spec(case.input_shape)
            params = _bind_params(contract, case)
            output, err = contract.transfer(input_shape, params)
        except Exception as exc:
            errors.append(f"conformance[{idx}] transfer raised: {exc}")
            continue

        if case.expected_error_contains is not None:
            needle = case.expected_error_contains
            if not (err and needle in err):
                errors.append(
                    f"conformance[{idx}]: expected error containing {needle!r}, "
                    f"got {err!r}"
                )
                continue
            checked += 1
            continue

        try:
            expected = _shape_from_spec(case.expected_output or ())
        except Exception as exc:
            errors.append(f"conformance[{idx}].expected_output invalid: {exc}")
            continue
        if err is not None:
            errors.append(f"conformance[{idx}]: unexpected error {err!r}")
            continue
        got = _shape_values(output)
        want = _shape_values(expected)
        if got != want:
            errors.append(
                f"conformance[{idx}]: expected output {want}, got {got}"
            )
            continue
        checked += 1
    return checked


def validate_operator_theory(contract: OperatorTheoryContract) -> PluginValidationReport:
    """Validate one executable operator theory without touching global state."""

    errors: List[str] = []
    class_name = contract.class_name

    if not is_abi_compatible(contract.abi_version):
        errors.append(
            f"unsupported ABI version {contract.abi_version!r}; "
            f"runtime supports major {SUPPORTED_ABI_MAJOR}"
        )
    if not isinstance(class_name, str) or not class_name.strip():
        errors.append("class_name must be a non-empty string")
    if not callable(contract.transfer):
        errors.append("transfer must be callable")

    prov = contract.provenance
    if prov is None:
        errors.append("provenance is required")
    else:
        for field_name in ("package", "version", "source_url", "license", "author"):
            if not str(getattr(prov, field_name, "")).strip():
                errors.append(f"provenance.{field_name} is required")

    review = contract.security_review
    if review is None:
        errors.append("security_review is required")
    else:
        if not review.reviewed_by.strip():
            errors.append("security_review.reviewed_by is required")
        if not review.reviewed_on.strip():
            errors.append("security_review.reviewed_on is required")
        for field_name in (
            "no_import_side_effects",
            "no_network",
            "no_filesystem_writes",
            "deterministic",
            "no_model_execution",
        ):
            if getattr(review, field_name) is not True:
                errors.append(f"security_review.{field_name} must be attested true")

    cases_checked = 0
    if callable(contract.transfer):
        cases_checked = _validate_conformance(contract, errors)

    return PluginValidationReport(
        class_name=class_name,
        ok=not errors,
        errors=tuple(errors),
        cases_checked=cases_checked,
    )


def validate_operator_theories(
    contracts: Iterable[OperatorTheoryContract],
) -> List[PluginValidationReport]:
    """Validate several contracts in deterministic order."""

    return [validate_operator_theory(contract) for contract in contracts]


def install_operator_theories(
    contracts: Iterable[OperatorTheoryContract],
    *,
    fail_on_invalid: bool = True,
) -> List[PluginValidationReport]:
    """Validate and install plugin transfers into the shared shape-stub registry.

    The function returns one report per contract.  By default any invalid
    contract aborts installation of the whole batch; callers may set
    ``fail_on_invalid=False`` to install only the valid contracts after inspecting
    the returned reports.
    """

    contracts_list = list(contracts)
    reports = validate_operator_theories(contracts_list)
    invalid = [report for report in reports if not report.ok]
    if invalid and fail_on_invalid:
        messages = "; ".join(
            f"{report.class_name}: {', '.join(report.errors)}" for report in invalid
        )
        raise ValueError(f"invalid TensorGuard operator plugin contract(s): {messages}")

    for contract, report in zip(contracts_list, reports):
        if not report.ok:
            continue
        register_shape_stub(
            contract.class_name,
            contract.transfer,
            arg_names=contract.arg_names,
            defaults=contract.defaults,
        )
    return reports


__all__ = [
    "ABI_VERSION",
    "SUPPORTED_ABI_MAJOR",
    "ConformanceCase",
    "OperatorTheoryContract",
    "PluginProvenance",
    "PluginValidationReport",
    "SecurityReview",
    "install_operator_theories",
    "is_abi_compatible",
    "validate_operator_theories",
    "validate_operator_theory",
]
