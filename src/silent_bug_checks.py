"""Object-level gates for runtime-silent PyTorch bugs.

These checks complement the source/FX verifier.  They validate explicit runtime
contracts that are normally invisible to a single forward pass: trainability,
declared buffer values, optimizer-state fingerprints, expected train/eval modes,
and declared quantization parameters.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Mapping, Optional, Tuple


@dataclass(frozen=True)
class SilentBugIssue:
    kind: str
    message: str
    location: str = "module"
    confidence: str = "sound"
    evidence: str = ""


@dataclass(frozen=True)
class SilentBugVerdict:
    ok: bool
    issues: Tuple[SilentBugIssue, ...] = ()

    def has_issue(self, kind: str) -> bool:
        return any(issue.kind == kind for issue in self.issues)


def verify_frozen_parameters(model: Any) -> SilentBugVerdict:
    """Reject parameters declared trainable but currently frozen.

    Models opt in with ``tensorguard_expected_trainable``: an iterable of exact
    parameter names or module-name prefixes whose parameters must require grad.
    """

    expected = tuple(getattr(model, "tensorguard_expected_trainable", ()))
    issues = []
    for name, param in _named_parameters(model):
        if _matches_expected(name, expected) and not bool(getattr(param, "requires_grad", False)):
            issues.append(SilentBugIssue(
                "gradient_freeze",
                f"{name} is declared trainable but requires_grad=False",
                name,
                evidence="tensorguard_expected_trainable",
            ))
    return SilentBugVerdict(not issues, tuple(issues))


def verify_declared_buffers(model: Any, *, atol: float = 1e-6) -> SilentBugVerdict:
    """Compare registered buffers against declared expected values."""

    expected = getattr(model, "tensorguard_expected_buffers", {})
    actual = dict(_named_buffers(model))
    issues = []
    for name, want in expected.items():
        if name not in actual:
            issues.append(SilentBugIssue(
                "stale_buffer",
                f"expected buffer {name!r} is missing",
                name,
                evidence="missing",
            ))
            continue
        if not _tensor_close(actual[name], want, atol=atol):
            issues.append(SilentBugIssue(
                "stale_buffer",
                f"buffer {name!r} differs from its declared expected value",
                name,
                evidence=f"actual={_tensor_fingerprint(actual[name])[:16]} expected={_tensor_fingerprint(want)[:16]}",
            ))
    return SilentBugVerdict(not issues, tuple(issues))


def optimizer_state_fingerprints(optimizer: Any) -> Dict[str, str]:
    """Return deterministic fingerprints for tensor/scalar optimizer state."""

    out: Dict[str, str] = {}
    state = getattr(optimizer, "state", {})
    params = []
    for group in getattr(optimizer, "param_groups", ()):
        params.extend(group.get("params", ()))
    index_by_id = {id(param): i for i, param in enumerate(params)}
    for param, slots in state.items():
        pidx = index_by_id.get(id(param))
        if pidx is None or not isinstance(slots, Mapping):
            continue
        for key, value in sorted(slots.items(), key=lambda kv: str(kv[0])):
            if _is_fingerprintable(value):
                out[f"p{pidx}:{key}"] = _tensor_fingerprint(value)
    return out


def verify_optimizer_state_fingerprints(
    optimizer: Any,
    expected_fingerprints: Mapping[str, str],
) -> SilentBugVerdict:
    """Reject optimizer states whose declared fingerprints drifted."""

    actual = optimizer_state_fingerprints(optimizer)
    issues = []
    for key, want in sorted(expected_fingerprints.items()):
        got = actual.get(key)
        if got != want:
            issues.append(SilentBugIssue(
                "optimizer_state_drift",
                f"optimizer state {key} fingerprint drifted",
                key,
                evidence=f"actual={(got or 'missing')[:16]} expected={want[:16]}",
            ))
    return SilentBugVerdict(not issues, tuple(issues))


def verify_mode_contract(model: Any) -> SilentBugVerdict:
    """Validate declared train/eval mode contracts for a module tree."""

    issues = []
    if hasattr(model, "tensorguard_expected_training"):
        expected = bool(getattr(model, "tensorguard_expected_training"))
        actual = bool(getattr(model, "training", False))
        if actual != expected:
            issues.append(SilentBugIssue(
                "train_eval_mode_leakage",
                f"module.training is {actual}, expected {expected}",
                "module.training",
                evidence="tensorguard_expected_training",
            ))
    module_modes = getattr(model, "tensorguard_expected_module_modes", {})
    modules = dict(_named_modules(model))
    for name, expected in module_modes.items():
        if name not in modules:
            issues.append(SilentBugIssue(
                "train_eval_mode_leakage",
                f"expected submodule {name!r} is missing",
                name,
                evidence="missing",
            ))
            continue
        actual = bool(getattr(modules[name], "training", False))
        if actual != bool(expected):
            issues.append(SilentBugIssue(
                "train_eval_mode_leakage",
                f"{name}.training is {actual}, expected {bool(expected)}",
                name,
                evidence="tensorguard_expected_module_modes",
            ))
    return SilentBugVerdict(not issues, tuple(issues))


def verify_quantization_contract(model: Any) -> SilentBugVerdict:
    """Validate declared quantization parameters on module attributes."""

    contract = getattr(model, "tensorguard_quantization_contract", {})
    issues = []
    for attr, expected in sorted(contract.items()):
        actual = getattr(model, attr, None)
        if actual != expected:
            issues.append(SilentBugIssue(
                "quantization_wrong_output",
                f"quantization attribute {attr!r} is {actual!r}, expected {expected!r}",
                attr,
                evidence="tensorguard_quantization_contract",
            ))
    return SilentBugVerdict(not issues, tuple(issues))


def verify_silent_bug_contracts(
    model: Any,
    *,
    optimizer: Optional[Any] = None,
    optimizer_fingerprints: Optional[Mapping[str, str]] = None,
) -> SilentBugVerdict:
    """Run every object-level silent-bug gate relevant to the provided objects."""

    issues = []
    for verdict in (
        verify_frozen_parameters(model),
        verify_declared_buffers(model),
        verify_mode_contract(model),
        verify_quantization_contract(model),
    ):
        issues.extend(verdict.issues)
    if optimizer is not None and optimizer_fingerprints is not None:
        issues.extend(verify_optimizer_state_fingerprints(optimizer, optimizer_fingerprints).issues)
    return SilentBugVerdict(not issues, tuple(issues))


def _named_parameters(model: Any) -> Iterable[Tuple[str, Any]]:
    named = getattr(model, "named_parameters", None)
    return () if named is None else named()


def _named_buffers(model: Any) -> Iterable[Tuple[str, Any]]:
    named = getattr(model, "named_buffers", None)
    return () if named is None else named()


def _named_modules(model: Any) -> Iterable[Tuple[str, Any]]:
    named = getattr(model, "named_modules", None)
    return (("", model),) if named is None else named()


def _matches_expected(name: str, expected: Tuple[str, ...]) -> bool:
    return any(name == item or name.startswith(item + ".") for item in expected)


def _is_fingerprintable(value: Any) -> bool:
    return hasattr(value, "detach") or isinstance(value, (int, float, bool))


def _tensor_fingerprint(value: Any) -> str:
    h = hashlib.sha256()
    if hasattr(value, "detach"):
        tensor = value.detach().cpu().contiguous()
        h.update(str(tensor.dtype).encode())
        h.update(str(tuple(tensor.shape)).encode())
        h.update(tensor.numpy().tobytes())
    else:
        h.update(type(value).__name__.encode())
        h.update(repr(value).encode())
    return h.hexdigest()


def _tensor_close(actual: Any, expected: Any, *, atol: float) -> bool:
    if not (hasattr(actual, "detach") and hasattr(expected, "detach")):
        return actual == expected
    if tuple(actual.shape) != tuple(expected.shape):
        return False
    return bool((actual.detach().cpu() - expected.detach().cpu()).abs().le(atol).all().item())
