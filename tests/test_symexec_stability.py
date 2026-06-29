"""Step 90 — pin the symbolic-execution engine's stability-guaranteed surface.

These tests fail if a stability-guaranteed ``src.symexec`` entry point, result
type, ``SymBugKind`` member, ``SymConfig`` field, or analysis mode is removed or
renamed without going through the deprecation process described in
``DEPRECATION_POLICY.md``. ``SymBugKind`` is *additive*: new members may be
appended, but the ones pinned here must not disappear.

Per ``DEPRECATION_POLICY.md`` the *preview* surface (telemetry, mutation, fuzz,
coverage, incremental/parallel drivers, editor/export integrations, and the
``Interpreter``/lattice internals) is deliberately NOT pinned here.
"""

import dataclasses as dc
import inspect

import src.symexec as symexec

# Stability-guaranteed entry points.
_STABLE_ENTRYPOINTS = {
    "analyze_source",
    "analyze_file",
    "analyze_package",
}

# Stability-guaranteed types and singletons.
_STABLE_TYPES = {
    "SymResult",
    "SymBug",
    "SymBugKind",
    "PackageResult",
    "SymConfig",
    "DEFAULT_CONFIG",
    "MODES",
}

# SymBugKind is additive: these members must remain present.
_PINNED_BUG_KINDS = {
    "MATMUL_DIM_MISMATCH",
    "BROADCAST_MISMATCH",
    "RESHAPE_SIZE_MISMATCH",
    "EINSUM_DIM_MISMATCH",
    "LAYER_DIM_MISMATCH",
    "AXIS_OUT_OF_RANGE",
    "RANK_INDEX_ERROR",
    "NEGATIVE_DIMENSION",
}

# Documented public fields of the result/finding/config types.
_SYMRESULT_FIELDS = {"bugs", "functions_analyzed", "ran_main", "abstentions"}
_SYMBUG_FIELDS = {"kind", "message", "line", "col", "confidence"}
_SYMCONFIG_FIELDS = {
    "mode",
    "min_confidence",
    "require_feasibility",
    "enable_heuristics",
    "budget_ms",
}


def test_stable_surface_exported():
    exported = set(symexec.__all__)
    for name in _STABLE_ENTRYPOINTS | _STABLE_TYPES:
        assert name in exported, f"symexec.{name} dropped from __all__"
        assert hasattr(symexec, name), f"symexec.{name} not importable"


def test_entrypoints_are_callable():
    for name in _STABLE_ENTRYPOINTS:
        assert callable(getattr(symexec, name)), f"symexec.{name} not callable"


def test_entrypoint_signatures_keep_stable_params():
    # source/root plus the config knob threaded through every entry point.
    assert {"source", "filename", "config"} <= set(
        inspect.signature(symexec.analyze_source).parameters
    )
    assert {"root", "config"} <= set(
        inspect.signature(symexec.analyze_package).parameters
    )
    assert "config" in inspect.signature(symexec.analyze_file).parameters


def test_bug_kinds_are_additive():
    members = {m.name for m in symexec.SymBugKind}
    missing = _PINNED_BUG_KINDS - members
    assert not missing, f"SymBugKind members removed: {missing}"


def test_result_and_bug_fields_present():
    assert _SYMRESULT_FIELDS <= {f.name for f in dc.fields(symexec.SymResult)}
    assert _SYMBUG_FIELDS <= {f.name for f in dc.fields(symexec.SymBug)}


def test_config_fields_and_modes():
    assert _SYMCONFIG_FIELDS <= {f.name for f in dc.fields(symexec.SymConfig)}
    assert symexec.MODES == ("sound", "balanced", "heuristic")
    # balanced remains the default.
    assert symexec.DEFAULT_CONFIG.mode == "balanced"


def test_documented_in_deprecation_policy():
    import os

    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    with open(os.path.join(repo, "DEPRECATION_POLICY.md"), encoding="utf-8") as fh:
        text = fh.read()
    assert "src.symexec" in text
    assert "Proof fingerprints are not a compatibility surface" in text
    for name in _STABLE_ENTRYPOINTS:
        assert name in text, f"{name} not documented in DEPRECATION_POLICY.md"
