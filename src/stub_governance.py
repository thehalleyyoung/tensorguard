"""Step 178 — community stub-registry **governance** (review, provenance, CI).

The shape-stub registry (:mod:`src.shape_stub_registry`) lets third-party layers
be taught to TensorGuard. To open that registry to *community contributions*
without (a) executing untrusted code or (b) silently weakening soundness, this
module defines a **declarative, provenance-bearing manifest format** and a
**validator** that every submission must pass — locally and in CI.

Design constraints that make community stubs safe to accept:

* **No arbitrary code.** A manifest never carries a Python ``transfer``. It only
  selects a *vetted, built-in* transfer ``kind`` (``shape_preserving`` or
  ``last_dim_linear``) and its declarative parameters. The validator refuses any
  manifest containing code-bearing fields (``transfer``, ``code``, ``python``,
  ``eval``, ``exec``, ``import``), so a malicious PR cannot run on a reviewer's
  or CI machine.
* **Mandatory provenance.** Every manifest must name an ``author``, a
  ``source_url`` (the upstream layer it models), a ``license``, and a
  ``reviewed_by`` reviewer. Missing or empty provenance is a hard reject — the
  registry stays auditable.
* **Conformance is proof, not promise.** Each manifest ships ≥1 conformance case
  (constructor args + an input shape → an expected output shape *or* an expected
  error substring). The validator *actually registers* the declared stub into an
  isolated registry and runs every case, so a stub that doesn't behave as
  claimed is rejected before merge.

Public API
----------
``validate_manifest(obj) -> ValidationReport``
``validate_manifest_file(path) -> ValidationReport``
``validate_directory(dir) -> list[ValidationReport]``
``load_community_stubs(dir) -> list[str]``  (registers all *valid* manifests)
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from src.tensor_shapes import ShapeDim, TensorShape
from src.shape_stub_registry import (
    ShapeStub,
    get_shape_stub,
    register_last_dim_linear,
    register_shape_preserving,
    registered_stub_names,
    clear_user_stubs,
)

# Fields that would imply executable content — their presence is a hard reject.
_FORBIDDEN_FIELDS = frozenset({
    "transfer", "code", "python", "eval", "exec", "import", "lambda", "__",
})

_REQUIRED_PROVENANCE = ("author", "source_url", "license", "reviewed_by")

_SUPPORTED_KINDS = frozenset({"shape_preserving", "last_dim_linear"})


@dataclass
class ValidationReport:
    """Outcome of validating a single community stub manifest."""
    class_name: Optional[str]
    ok: bool
    errors: List[str] = field(default_factory=list)
    cases_checked: int = 0
    source: Optional[str] = None

    def __bool__(self) -> bool:  # truthy iff valid
        return self.ok


def _dim_from_spec(spec: Any) -> ShapeDim:
    """Map a JSON dim (``int`` → concrete, ``str`` → symbolic) to a ShapeDim."""
    if isinstance(spec, bool):  # guard: bool is an int subclass
        raise ValueError(f"invalid dim {spec!r}")
    if isinstance(spec, int):
        return ShapeDim(spec)
    if isinstance(spec, str) and spec:
        return ShapeDim(spec)
    raise ValueError(f"invalid dim {spec!r}; expected int or non-empty str")


def _shape_from_spec(spec: Any) -> TensorShape:
    if not isinstance(spec, (list, tuple)) or not spec:
        raise ValueError(f"shape must be a non-empty list, got {spec!r}")
    return TensorShape(tuple(_dim_from_spec(d) for d in spec))


def _check_no_forbidden(obj: Dict[str, Any], errors: List[str]) -> None:
    for key in obj:
        low = str(key).lower()
        if any(bad in low for bad in _FORBIDDEN_FIELDS):
            errors.append(
                f"forbidden field {key!r}: community stubs are declarative only "
                f"(no executable code is accepted)"
            )


def _check_provenance(obj: Dict[str, Any], errors: List[str]) -> None:
    prov = obj.get("provenance")
    if not isinstance(prov, dict):
        errors.append("missing 'provenance' object")
        return
    for fieldname in _REQUIRED_PROVENANCE:
        val = prov.get(fieldname)
        if not isinstance(val, str) or not val.strip():
            errors.append(f"provenance.{fieldname} is required and must be non-empty")


def _register_declared_stub(obj: Dict[str, Any]) -> Tuple[str, Optional[str]]:
    """Register the declared stub into the (user) registry. Returns (name, err)."""
    class_name = obj.get("class_name")
    kind = obj.get("kind")
    if not isinstance(class_name, str) or not class_name:
        return "", "missing/invalid 'class_name'"
    if kind not in _SUPPORTED_KINDS:
        return class_name, (
            f"unsupported kind {kind!r}; allowed: {sorted(_SUPPORTED_KINDS)}"
        )
    if kind == "shape_preserving":
        register_shape_preserving(class_name)
        return class_name, None
    # last_dim_linear
    spec = obj.get("spec", {})
    if not isinstance(spec, dict):
        return class_name, "'spec' must be an object for last_dim_linear"
    in_arg = spec.get("in_arg")
    out_arg = spec.get("out_arg")
    arg_names = spec.get("arg_names")
    if not (isinstance(in_arg, str) and isinstance(out_arg, str)):
        return class_name, "last_dim_linear requires string 'in_arg' and 'out_arg'"
    if not (isinstance(arg_names, list) and all(isinstance(a, str) for a in arg_names)):
        return class_name, "last_dim_linear requires 'arg_names' (list of strings)"
    register_last_dim_linear(
        class_name,
        in_arg=in_arg,
        out_arg=out_arg,
        arg_names=tuple(arg_names),
        defaults=spec.get("defaults") or {},
        out_defaults_to_in=bool(spec.get("out_defaults_to_in", False)),
    )
    return class_name, None


def _run_conformance(class_name: str, cases: Any, errors: List[str]) -> int:
    if not isinstance(cases, list) or not cases:
        errors.append("at least one 'conformance' case is required")
        return 0
    stub: Optional[ShapeStub] = get_shape_stub(class_name)
    if stub is None:
        errors.append(f"stub {class_name!r} did not register")
        return 0
    checked = 0
    for i, case in enumerate(cases):
        if not isinstance(case, dict):
            errors.append(f"conformance[{i}] must be an object")
            continue
        try:
            inp = _shape_from_spec(case["input"])
        except Exception as exc:
            errors.append(f"conformance[{i}].input invalid: {exc}")
            continue
        ctor_args = tuple(case.get("ctor_args", ()) or ())
        ctor_kwargs = dict(case.get("ctor_kwargs", {}) or {})
        params = stub.bind_params(ctor_args, ctor_kwargs)
        try:
            out, err = stub.transfer(inp, params)
        except Exception as exc:  # a stub must never crash
            errors.append(f"conformance[{i}] transfer raised: {exc}")
            continue
        expect = case.get("expect", {})
        if "error_contains" in expect:
            needle = expect["error_contains"]
            if not (err and needle in err):
                errors.append(
                    f"conformance[{i}]: expected error containing {needle!r}, "
                    f"got err={err!r}"
                )
            else:
                checked += 1
        elif "output" in expect:
            if err is not None:
                errors.append(f"conformance[{i}]: unexpected error {err!r}")
                continue
            try:
                want = _shape_from_spec(expect["output"])
            except Exception as exc:
                errors.append(f"conformance[{i}].expect.output invalid: {exc}")
                continue
            got = tuple(d.value for d in (out.dims if out else ()))
            wnt = tuple(d.value for d in want.dims)
            if got != wnt:
                errors.append(
                    f"conformance[{i}]: expected output {wnt}, got {got}"
                )
            else:
                checked += 1
        else:
            errors.append(
                f"conformance[{i}].expect must contain 'output' or 'error_contains'"
            )
    return checked


def validate_manifest(obj: Dict[str, Any], *, source: Optional[str] = None) -> ValidationReport:
    """Validate a single in-memory manifest object (no merge side effects)."""
    errors: List[str] = []
    class_name = obj.get("class_name") if isinstance(obj, dict) else None
    if not isinstance(obj, dict):
        return ValidationReport(class_name=None, ok=False,
                                errors=["manifest must be a JSON object"], source=source)

    _check_no_forbidden(obj, errors)
    _check_provenance(obj, errors)

    # Register + conformance-check in an isolated user-registry scope so a bad
    # submission can't pollute the live registry.
    pre = set(registered_stub_names())
    cases_checked = 0
    try:
        reg_name, reg_err = _register_declared_stub(obj)
        if reg_err:
            errors.append(reg_err)
        elif reg_name:
            cases_checked = _run_conformance(reg_name, obj.get("conformance"), errors)
    finally:
        # Remove any stub we just added, restoring the prior user set.
        post = set(registered_stub_names())
        for nm in post - pre:
            # only drop user (non-builtin) additions
            st = get_shape_stub(nm)
            if st is not None and not st.is_builtin:
                from src.shape_stub_registry import _STUB_REGISTRY  # local import
                _STUB_REGISTRY.pop(nm, None)

    return ValidationReport(
        class_name=class_name,
        ok=not errors,
        errors=errors,
        cases_checked=cases_checked,
        source=source,
    )


def validate_manifest_file(path: str) -> ValidationReport:
    try:
        with open(path, "r", encoding="utf-8") as fh:
            obj = json.load(fh)
    except Exception as exc:
        return ValidationReport(class_name=None, ok=False,
                                errors=[f"could not parse {path}: {exc}"], source=path)
    return validate_manifest(obj, source=path)


def validate_directory(directory: str) -> List[ValidationReport]:
    """Validate every ``*.json`` manifest in *directory* (sorted, non-recursive)."""
    reports: List[ValidationReport] = []
    if not os.path.isdir(directory):
        return reports
    for name in sorted(os.listdir(directory)):
        if name.endswith(".json"):
            reports.append(validate_manifest_file(os.path.join(directory, name)))
    return reports


def load_community_stubs(directory: str) -> List[str]:
    """Register every *valid* manifest in *directory*; return the class names.

    Invalid manifests are skipped (never partially registered). This is the
    runtime entry point an application calls to opt into the community set.
    """
    loaded: List[str] = []
    for name in sorted(os.listdir(directory)) if os.path.isdir(directory) else []:
        if not name.endswith(".json"):
            continue
        path = os.path.join(directory, name)
        report = validate_manifest_file(path)
        if not report.ok:
            continue
        with open(path, "r", encoding="utf-8") as fh:
            obj = json.load(fh)
        reg_name, reg_err = _register_declared_stub(obj)
        if reg_err is None and reg_name:
            loaded.append(reg_name)
    return loaded


__all__ = [
    "ValidationReport",
    "validate_manifest",
    "validate_manifest_file",
    "validate_directory",
    "load_community_stubs",
]
