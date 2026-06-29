"""Tests for configuration & soundness modes (roadmap Step 86).

The engine ships three named modes whose report sets nest:

    ``sound`` ⊆ ``balanced`` ⊆ ``heuristic``

``balanced`` is the default and must be byte-identical to the engine's historic
behaviour (so omitting a config changes nothing and the corpus fingerprint is
preserved).  ``sound`` is a strict subset (maximum precision: a confidence floor
plus a positive-feasibility requirement).  ``heuristic`` is a superset (maximum
recall: best-effort, low-confidence suspicions at sites where balanced abstains).
"""

import os

import pytest

from src.symexec import analyze_source, SymConfig, MODES, DEFAULT_CONFIG
from src.symexec.engine import analyze_file
from src.symexec.config import SymConfig as _SymConfig
from src.symexec.package import analyze_package


CORPUS = os.path.join(
    os.path.dirname(__file__), "symexec_corpus", "wild", "matmul_dim_mismatch.py"
)

# A program whose only finding is a concrete dim (7) aligned with a *symbolic*
# dim: balanced/sound soundly abstain; heuristic surfaces a low-confidence
# suspected broadcast mismatch.
_HEURISTIC_SRC = """
import torch
def f(n):
    a = torch.randn(7)
    b = torch.randn(n)
    return a + b
"""

# A program with a forced, fully-concrete matmul mismatch: reported in *every*
# mode (no path constraints, high confidence).
_PROVEN_SRC = """
import torch
def f():
    a = torch.randn(2, 3)
    b = torch.randn(4, 5)
    return a @ b
"""


# --------------------------------------------------------------------------- #
# SymConfig construction / validation                                         #
# --------------------------------------------------------------------------- #

def test_default_config_is_balanced():
    assert DEFAULT_CONFIG.mode == "balanced"
    assert DEFAULT_CONFIG.min_confidence == 0.0
    assert DEFAULT_CONFIG.require_feasibility is False
    assert DEFAULT_CONFIG.enable_heuristics is False
    assert DEFAULT_CONFIG.budget_ms is None


def test_modes_tuple_ordering():
    assert MODES == ("sound", "balanced", "heuristic")


def test_preset_sound():
    c = SymConfig.sound()
    assert c.mode == "sound"
    assert c.require_feasibility is True
    assert c.enable_heuristics is False
    assert c.min_confidence > 0.0


def test_preset_heuristic():
    c = SymConfig.heuristic()
    assert c.mode == "heuristic"
    assert c.enable_heuristics is True
    assert c.require_feasibility is False
    assert c.min_confidence == 0.0


def test_preset_balanced_equals_default():
    assert SymConfig.balanced() == DEFAULT_CONFIG


def test_for_mode_dispatch():
    for m in MODES:
        assert SymConfig.for_mode(m).mode == m


def test_for_mode_rejects_unknown():
    with pytest.raises(ValueError):
        SymConfig.for_mode("aggressive")


def test_constructor_rejects_unknown_mode():
    with pytest.raises(ValueError):
        SymConfig(mode="nope")


def test_constructor_rejects_out_of_range_confidence():
    with pytest.raises(ValueError):
        SymConfig(min_confidence=1.5)
    with pytest.raises(ValueError):
        SymConfig(min_confidence=-0.1)


def test_constructor_rejects_nonpositive_budget():
    with pytest.raises(ValueError):
        SymConfig(budget_ms=0)
    with pytest.raises(ValueError):
        SymConfig(budget_ms=-5)


def test_config_is_frozen():
    c = SymConfig()
    with pytest.raises(Exception):
        c.mode = "sound"  # type: ignore[misc]


def test_with_overrides_preserves_mode():
    c = SymConfig.sound().with_overrides(min_confidence=0.5)
    assert c.mode == "sound"
    assert c.min_confidence == 0.5
    # the original is untouched (immutability)
    assert SymConfig.sound().min_confidence != 0.5


def test_with_overrides_revalidates():
    with pytest.raises(ValueError):
        SymConfig.balanced().with_overrides(min_confidence=2.0)


def test_overrides_via_preset_kwargs():
    c = SymConfig.heuristic(min_confidence=0.6)
    assert c.mode == "heuristic"
    assert c.enable_heuristics is True
    assert c.min_confidence == 0.6


def test_allows_confidence():
    c = SymConfig.sound()
    assert not c.allows_confidence(0.5)
    assert c.allows_confidence(0.9)
    assert SymConfig.balanced().allows_confidence(0.0)


def test_module_export_aliases_same_class():
    assert SymConfig is _SymConfig


# --------------------------------------------------------------------------- #
# Default behaviour is byte-identical (the zero-regression contract)          #
# --------------------------------------------------------------------------- #

def test_default_matches_explicit_balanced():
    r_implicit = analyze_file(CORPUS)
    r_explicit = analyze_file(CORPUS, config=SymConfig.balanced())
    assert r_implicit.fingerprint() == r_explicit.fingerprint()


def test_corpus_fingerprint_unchanged():
    # The golden corpus matmul fingerprint must survive Step 86 in the default
    # (balanced) mode.
    r = analyze_file(CORPUS)
    assert r.fingerprint() == (
        "de466b6f54018384cb5b3c27b5b3f7be"
        "178001535d59bb785c46fdba83ead9e0"
    )


def test_sound_preserves_corpus_fingerprint():
    # The corpus fault is unconditional + high-confidence, so even ``sound``
    # (which drops weak / unprovable-feasibility findings) reports it unchanged.
    r = analyze_file(CORPUS, config=SymConfig.sound())
    assert r.fingerprint() == analyze_file(CORPUS).fingerprint()


# --------------------------------------------------------------------------- #
# Mode nesting: sound ⊆ balanced ⊆ heuristic                                  #
# --------------------------------------------------------------------------- #

def _bug_keys(result):
    return {(b.kind, b.line, b.col, b.message) for b in result.bugs}


def test_heuristic_is_superset_on_symbolic_broadcast():
    sound = analyze_source(_HEURISTIC_SRC, config=SymConfig.sound())
    balanced = analyze_source(_HEURISTIC_SRC, config=SymConfig.balanced())
    heuristic = analyze_source(_HEURISTIC_SRC, config=SymConfig.heuristic())

    assert _bug_keys(sound) == set()
    assert _bug_keys(balanced) == set()
    # heuristic surfaces a single suspected broadcast mismatch the sound/balanced
    # modes abstained on.
    assert len(heuristic.bugs) == 1
    (bug,) = heuristic.bugs
    assert bug.kind.value == "broadcast_mismatch"
    assert "suspected" in bug.message.lower()
    assert bug.confidence < 0.85  # well below the sound floor

    # nesting holds
    assert _bug_keys(sound) <= _bug_keys(balanced) <= _bug_keys(heuristic)


def test_proven_fault_reported_in_every_mode():
    keys = {}
    for m in MODES:
        r = analyze_source(_PROVEN_SRC, config=SymConfig.for_mode(m))
        keys[m] = _bug_keys(r)
        assert any(b.kind.value == "matmul_dim_mismatch" for b in r.bugs)
    # the proven fault is identical across modes
    assert keys["sound"] == keys["balanced"] == keys["heuristic"]


def test_balanced_equals_default_on_heuristic_program():
    # The default analysis must not surface heuristic suspicions.
    assert _bug_keys(analyze_source(_HEURISTIC_SRC)) == set()


# --------------------------------------------------------------------------- #
# Confidence floor + feasibility gate                                         #
# --------------------------------------------------------------------------- #

def test_min_confidence_floor_filters_suspicion():
    # A heuristic run with a high floor drops the low-confidence suspicion.
    cfg = SymConfig.heuristic(min_confidence=0.9)
    r = analyze_source(_HEURISTIC_SRC, config=cfg)
    assert r.bugs == []


def test_min_confidence_floor_keeps_high_confidence_proven():
    cfg = SymConfig.balanced(min_confidence=0.9)
    r = analyze_source(_PROVEN_SRC, config=cfg)
    assert any(b.kind.value == "matmul_dim_mismatch" for b in r.bugs)


# --------------------------------------------------------------------------- #
# Threading through the whole-package driver                                  #
# --------------------------------------------------------------------------- #

def _write(root, files):
    for rel, txt in files.items():
        p = os.path.join(str(root), rel)
        os.makedirs(os.path.dirname(p) or ".", exist_ok=True)
        with open(p, "w", encoding="utf-8") as fh:
            fh.write(txt)


def test_analyze_package_accepts_config(tmp_path):
    _write(tmp_path, {"m.py": _HEURISTIC_SRC})
    balanced = analyze_package(str(tmp_path))
    heuristic = analyze_package(str(tmp_path), config=SymConfig.heuristic())
    assert all(
        b.kind.value != "broadcast_mismatch" for _, b in balanced.all_bugs()
    )
    assert any(
        b.kind.value == "broadcast_mismatch" and "suspected" in b.message.lower()
        for _, b in heuristic.all_bugs()
    )


def test_analyze_package_default_unchanged(tmp_path):
    _write(tmp_path, {"m.py": _PROVEN_SRC})
    default = analyze_package(str(tmp_path))
    explicit = analyze_package(str(tmp_path), config=SymConfig.balanced())
    d = {p: r.fingerprint() for p, r in default.results.items()}
    e = {p: r.fingerprint() for p, r in explicit.results.items()}
    assert d == e


# --------------------------------------------------------------------------- #
# budget_ms threading                                                         #
# --------------------------------------------------------------------------- #

def test_config_budget_ms_is_used(tmp_path):
    # A config-supplied budget feeds analyze_source when no explicit budget is
    # passed; an enormous budget is a no-op (just exercises the plumbing).
    cfg = SymConfig.balanced(budget_ms=10_000)
    r = analyze_source(_PROVEN_SRC, config=cfg)
    assert any(b.kind.value == "matmul_dim_mismatch" for b in r.bugs)
