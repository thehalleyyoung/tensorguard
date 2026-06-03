"""Third-party stub/plugin certification suite.

Library authors can run this module from their own tests to prove that a
TensorGuard community stub or trusted operator plugin behaves correctly at the
same public boundary users rely on: ``verify_architecture`` and its
sound/balanced/heuristic verdicts.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import json
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Tuple

from src.api import verify_architecture
from src.operator_plugin_abi import (
    OperatorTheoryContract,
    PluginValidationReport,
    install_operator_theories,
    validate_operator_theories,
)
from src.shape_stub_registry import _STUB_REGISTRY  # intentionally scoped snapshot
from src.stub_governance import ValidationReport, validate_manifest

_MODES: Tuple[str, ...] = ("sound", "balanced", "heuristic")
_VERDICTS = frozenset({"SAFE", "UNSAFE", "UNKNOWN"})


@dataclass(frozen=True)
class ThirdPartyConformanceScenario:
    """One real verifier scenario for an extension.

    ``source`` should be a minimal ``nn.Module`` using the third-party class name
    exactly as users write it.  ``expected_verdicts`` may specify one verdict for
    all modes with ``{"*": "SAFE"}``, or individual entries for
    ``sound``/``balanced``/``heuristic``.
    """

    name: str
    source: str
    input_shapes: Mapping[str, Tuple[Any, ...]]
    expected_verdicts: Mapping[str, str]
    expected_bug_substrings: Tuple[str, ...] = ()
    expected_unknown_substrings: Tuple[str, ...] = ()
    max_cegar_iterations: int = 0


@dataclass(frozen=True)
class ThirdPartyScenarioResult:
    scenario: str
    mode: str
    expected: str
    verdict: str
    passed: bool
    bug_count: int = 0
    unknown_reasons: Tuple[str, ...] = ()
    errors: Tuple[str, ...] = ()

    def to_json_dict(self) -> Dict[str, Any]:
        return {
            "scenario": self.scenario,
            "mode": self.mode,
            "expected": self.expected,
            "verdict": self.verdict,
            "passed": self.passed,
            "bug_count": self.bug_count,
            "unknown_reasons": list(self.unknown_reasons),
            "errors": list(self.errors),
        }


@dataclass(frozen=True)
class ThirdPartyConformanceReport:
    extension_kind: str
    extension_name: str
    validation_reports: Tuple[Any, ...] = ()
    scenarios: Tuple[ThirdPartyScenarioResult, ...] = ()
    errors: Tuple[str, ...] = ()

    @property
    def passed(self) -> bool:
        return not self.errors and all(_report_ok(r) for r in self.validation_reports) and all(
            s.passed for s in self.scenarios
        )

    @property
    def cases_checked(self) -> int:
        total = 0
        for report in self.validation_reports:
            total += int(getattr(report, "cases_checked", 0))
        return total

    def to_json_dict(self) -> Dict[str, Any]:
        return {
            "extension_kind": self.extension_kind,
            "extension_name": self.extension_name,
            "passed": self.passed,
            "cases_checked": self.cases_checked,
            "validation_reports": [_validation_to_json(r) for r in self.validation_reports],
            "scenarios": [s.to_json_dict() for s in self.scenarios],
            "errors": list(self.errors),
        }

    def to_json(self) -> str:
        return json.dumps(self.to_json_dict(), indent=2, sort_keys=True) + "\n"

    def to_markdown(self) -> str:
        status = "PASS" if self.passed else "FAIL"
        lines = [
            f"# TensorGuard third-party conformance: {self.extension_name}",
            "",
            f"- Kind: `{self.extension_kind}`",
            f"- Status: **{status}**",
            f"- Transfer conformance cases checked: **{self.cases_checked}**",
            "",
            "| Scenario | Mode | Expected | Got | Result |",
            "| --- | --- | --- | --- | --- |",
        ]
        for scenario in self.scenarios:
            result = "PASS" if scenario.passed else "FAIL"
            lines.append(
                f"| {scenario.scenario} | {scenario.mode} | {scenario.expected} | "
                f"{scenario.verdict} | {result} |"
            )
        if self.errors:
            lines.extend(["", "## Errors"])
            lines.extend(f"- {err}" for err in self.errors)
        return "\n".join(lines) + "\n"


def certify_plugin_contracts(
    contracts: Iterable[OperatorTheoryContract],
    scenarios: Iterable[ThirdPartyConformanceScenario],
    *,
    extension_name: str = "operator-plugin",
    modes: Iterable[str] = _MODES,
) -> ThirdPartyConformanceReport:
    """Certify trusted executable plugin contracts against real verifier modes."""

    contracts_list = list(contracts)
    scenarios_list = list(scenarios)
    modes_tuple = _normalize_modes(modes)
    reports = tuple(validate_operator_theories(contracts_list))
    errors: List[str] = []
    if any(not report.ok for report in reports):
        errors.append("plugin validation failed; verifier scenarios were not run")
        return ThirdPartyConformanceReport(
            extension_kind="plugin",
            extension_name=extension_name,
            validation_reports=reports,
            errors=tuple(errors),
        )

    with _preserved_registry():
        install_operator_theories(contracts_list)
        scenario_results = _run_scenarios(scenarios_list, modes_tuple)
    return ThirdPartyConformanceReport(
        extension_kind="plugin",
        extension_name=extension_name,
        validation_reports=reports,
        scenarios=tuple(scenario_results),
        errors=tuple(errors),
    )


def certify_stub_manifests(
    manifests: Iterable[Mapping[str, Any]],
    scenarios: Iterable[ThirdPartyConformanceScenario],
    *,
    extension_name: str = "community-stubs",
    modes: Iterable[str] = _MODES,
) -> ThirdPartyConformanceReport:
    """Certify declarative community-stub manifests against real verifier modes."""

    manifest_list = [dict(manifest) for manifest in manifests]
    scenarios_list = list(scenarios)
    modes_tuple = _normalize_modes(modes)
    validation = tuple(validate_manifest(manifest) for manifest in manifest_list)
    errors: List[str] = []
    if any(not report.ok for report in validation):
        errors.append("stub-manifest validation failed; verifier scenarios were not run")
        return ThirdPartyConformanceReport(
            extension_kind="stub",
            extension_name=extension_name,
            validation_reports=validation,
            errors=tuple(errors),
        )

    with _preserved_registry():
        for manifest in manifest_list:
            _install_valid_manifest(manifest)
        scenario_results = _run_scenarios(scenarios_list, modes_tuple)
    return ThirdPartyConformanceReport(
        extension_kind="stub",
        extension_name=extension_name,
        validation_reports=validation,
        scenarios=tuple(scenario_results),
        errors=tuple(errors),
    )


def assert_conformance_passed(report: ThirdPartyConformanceReport) -> None:
    """Raise ``AssertionError`` with a compact report when certification fails."""

    if not report.passed:
        raise AssertionError(report.to_markdown())


def _expected_for_mode(scenario: ThirdPartyConformanceScenario, mode: str) -> str:
    verdict = scenario.expected_verdicts.get(mode, scenario.expected_verdicts.get("*"))
    if verdict not in _VERDICTS:
        raise ValueError(
            f"scenario {scenario.name!r} has no valid expected verdict for {mode!r}"
        )
    return verdict


def _run_scenarios(
    scenarios: Iterable[ThirdPartyConformanceScenario],
    modes: Tuple[str, ...],
) -> List[ThirdPartyScenarioResult]:
    results: List[ThirdPartyScenarioResult] = []
    for scenario in scenarios:
        for mode in modes:
            try:
                expected = _expected_for_mode(scenario, mode)
                result = verify_architecture(
                    scenario.source,
                    input_shapes=dict(scenario.input_shapes),
                    max_cegar_iterations=scenario.max_cegar_iterations,
                    soundness_mode=mode,
                )
                errors = _scenario_errors(scenario, result, expected)
                results.append(
                    ThirdPartyScenarioResult(
                        scenario=scenario.name,
                        mode=mode,
                        expected=expected,
                        verdict=result.verdict,
                        passed=not errors,
                        bug_count=result.bug_count,
                        unknown_reasons=tuple(result.unknown_reasons),
                        errors=tuple(errors),
                    )
                )
            except Exception as exc:
                results.append(
                    ThirdPartyScenarioResult(
                        scenario=scenario.name,
                        mode=mode,
                        expected=scenario.expected_verdicts.get(mode, "?"),
                        verdict="ERROR",
                        passed=False,
                        errors=(f"verifier raised: {exc}",),
                    )
                )
    return results


def _scenario_errors(
    scenario: ThirdPartyConformanceScenario,
    result: Any,
    expected: str,
) -> List[str]:
    errors: List[str] = []
    if result.verdict != expected:
        errors.append(f"expected {expected}, got {result.verdict}")
    bug_messages = "\n".join(getattr(bug, "message", "") for bug in result.bugs)
    for needle in scenario.expected_bug_substrings:
        if needle not in bug_messages:
            errors.append(f"missing bug substring {needle!r}")
    unknown_text = "\n".join(result.unknown_reasons)
    for needle in scenario.expected_unknown_substrings:
        if needle not in unknown_text:
            errors.append(f"missing UNKNOWN reason substring {needle!r}")
    return errors


def _normalize_modes(modes: Iterable[str]) -> Tuple[str, ...]:
    out = tuple(str(mode).lower() for mode in modes)
    invalid = [mode for mode in out if mode not in _MODES]
    if invalid:
        raise ValueError(f"invalid soundness mode(s): {invalid}")
    if not out:
        raise ValueError("at least one soundness mode is required")
    return out


def _install_valid_manifest(manifest: Mapping[str, Any]) -> None:
    from src.stub_governance import _register_declared_stub

    class_name, error = _register_declared_stub(dict(manifest))
    if error:
        raise ValueError(f"valid manifest {class_name!r} failed to install: {error}")


@contextmanager
def _preserved_registry() -> Iterator[None]:
    snapshot = dict(_STUB_REGISTRY)
    try:
        yield
    finally:
        _STUB_REGISTRY.clear()
        _STUB_REGISTRY.update(snapshot)


def _report_ok(report: Any) -> bool:
    return bool(getattr(report, "ok", False))


def _validation_to_json(report: Any) -> Dict[str, Any]:
    if isinstance(report, PluginValidationReport):
        return {
            "class_name": report.class_name,
            "ok": report.ok,
            "errors": list(report.errors),
            "cases_checked": report.cases_checked,
        }
    if isinstance(report, ValidationReport):
        return {
            "class_name": report.class_name,
            "ok": report.ok,
            "errors": list(report.errors),
            "cases_checked": report.cases_checked,
            "source": report.source,
        }
    return {
        "class_name": getattr(report, "class_name", None),
        "ok": _report_ok(report),
        "errors": list(getattr(report, "errors", ())),
        "cases_checked": int(getattr(report, "cases_checked", 0)),
    }


__all__ = [
    "ThirdPartyConformanceReport",
    "ThirdPartyConformanceScenario",
    "ThirdPartyScenarioResult",
    "assert_conformance_passed",
    "certify_plugin_contracts",
    "certify_stub_manifests",
]
