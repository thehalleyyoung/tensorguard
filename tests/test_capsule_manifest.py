"""Tests for the Step 122 reproducibility-capsule manifest + env verifier."""

from __future__ import annotations

from pathlib import Path

import pytest

import reproducibility.capsule_manifest as cm

REPO = Path(__file__).resolve().parent.parent

VOLATILE = ("time", "elapsed", "seconds", "date", "timestamp", "duration", "wall")


def _walk_keys(obj):
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield k
            yield from _walk_keys(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from _walk_keys(v)


def test_capsule_files_exist():
    assert (REPO / "capsule" / "Dockerfile.reproduce").exists()
    assert (REPO / "capsule" / "requirements.lock.txt").exists()
    assert (REPO / "capsule" / "reproduce.sh").exists()


def test_lock_parses_to_exact_pins():
    pins = cm.parse_lock()
    names = {n for n, _ in pins}
    # The evidence pipeline's real dependencies must be pinned.
    assert {"torch", "z3-solver", "numpy", "hypothesis", "pytest"} <= names
    # Every pin is an exact (==) version with a numeric release.
    for _, v in pins:
        assert cm._release_tuple(v)[0] >= 0


def test_satisfies_zero_pads_and_strips_local():
    assert cm.satisfies("2.9.1", "2.9.1")
    assert cm.satisfies("4.15.4.0", "4.15.4")  # trailing zero
    assert cm.satisfies("2.9.1+cpu", "2.9.1")  # local build tag stripped
    assert not cm.satisfies("2.9.2", "2.9.1")
    assert not cm.satisfies("2.10.0", "2.9.0")


def test_live_environment_satisfies_lock():
    # The "proven against real code" guarantee: the interpreter that will run
    # the evidence pipeline actually satisfies every pin.
    assert cm.verify_env() == 0


def test_manifest_counts_match_reproduce_all():
    import reproducibility.reproduce_all as ra
    d = cm.build_manifest()
    assert d["n_deterministic_artifacts_regenerated"] == len(
        ra.GENERATED_DETERMINISTIC)
    assert d["n_pinned_wheels"] == len(cm.parse_lock())


def test_manifest_hashes_track_files():
    d = cm.build_manifest()
    h = d["capsule_file_sha256"]
    assert h["reproduce.sh"] == cm._sha256(REPO / "capsule" / "reproduce.sh")
    assert h["Dockerfile.reproduce"] == cm._sha256(
        REPO / "capsule" / "Dockerfile.reproduce")


def test_reproduce_sh_invokes_one_command():
    text = (REPO / "capsule" / "reproduce.sh").read_text()
    assert "reproduce_all.py --check" in text
    assert "capsule_manifest.py --verify-env" in text


def test_no_volatile_keys_and_deterministic():
    d = cm.build_manifest()
    for k in _walk_keys(d):
        assert not any(tok in k.lower() for tok in VOLATILE), k
    assert cm.build_manifest() == d


def test_check_mode_byte_identical():
    assert cm.run(check=True) == 0


def test_capsule_dockerignore_keeps_evidence_dirs():
    # The capsule-specific ignore must NOT drop the dirs reproduce.sh needs.
    ig = (REPO / "capsule" / "Dockerfile.reproduce.dockerignore").read_text()
    lines = {ln.strip() for ln in ig.splitlines()
             if ln.strip() and not ln.startswith("#")}
    for needed in ("tests", "reproducibility", "evaluation", "corpus_extended"):
        assert needed not in lines
    # but it must still drop the gitignored roadmap and venv.
    assert "100_STEPS.md" in lines
    assert ".venv" in lines
