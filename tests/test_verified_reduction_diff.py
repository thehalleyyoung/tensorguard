"""Step 131 — tests for the Lean-extracted verified reduction differential audit."""

import json
import os
import re
import shutil
import subprocess

import pytest

from reproducibility import verified_reduction_diff as vrd

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
_LEAN = os.path.join(_ROOT, "lean")

_VOLATILE = ("time", "elapsed", "seconds", "date", "timestamp", "duration", "wall")


def _walk_keys(obj):
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield k
            yield from _walk_keys(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from _walk_keys(v)


def test_check_is_byte_identical():
    assert vrd.main.__module__  # importable
    rc = subprocess.run(
        ["python3", os.path.join("reproducibility", "verified_reduction_diff.py"),
         "--check"],
        cwd=_ROOT, capture_output=True, text=True,
    )
    assert rc.returncode == 0, rc.stdout + rc.stderr
    assert "OK" in rc.stdout


def test_no_volatile_keys():
    data = vrd.build()
    for k in _walk_keys(data):
        assert not any(v in k.lower() for v in _VOLATILE), f"volatile key {k}"


def test_sixteen_abstract_values():
    data = vrd.build()
    assert data["n_abstract_values"] == 16
    assert len(data["cases"]) == 16


def test_python_matches_verified_on_all_consistent_inputs():
    data = vrd.build()
    assert data["findings"]["python_matches_verified_on_all_consistent_inputs"]
    # every consistent input agrees
    for c in data["cases"]:
        if c["input_consistent"]:
            assert c["agree"], f"consistent input {c['input']} diverged"


def test_python_is_sound_overapproximation_everywhere():
    data = vrd.build()
    assert data["n_python_sound_overapproximation"] == data["n_abstract_values"]
    for c in data["cases"]:
        # gamma_python must contain gamma_lean
        assert set(c["gamma_lean"]) <= set(c["gamma_python"]), c["input"]


def test_divergences_only_on_contradictory_inputs():
    data = vrd.build()
    assert data["findings"]["divergences_only_on_contradictory_unreachable_inputs"]
    for code in data["divergent_inputs"]:
        assert not vrd._is_consistent(code)
        # a contradictory input has empty true (Lean) concretization
        assert vrd._gamma(code) == frozenset()


def test_known_divergences_are_the_two_contradictory_states():
    data = vrd.build()
    assert set(data["divergent_inputs"]) == {"01:null", "10:notnull"}


def test_reference_table_has_sixteen_rows():
    rows = vrd._load_reference()
    assert len(rows) == 16


@pytest.mark.slow
def test_reference_table_matches_live_lean_extraction(tmp_path):
    """The committed reference table is regenerable from the verified Lean model."""
    if shutil.which("lake") is None:
        pytest.skip("lake (Lean toolchain) not installed")
    build = subprocess.run(
        ["lake", "build", "TensorGuard.ReducedProduct"],
        cwd=_LEAN, capture_output=True, text=True, timeout=900,
    )
    assert build.returncode == 0, build.stderr[-3000:]
    script = os.path.join(_LEAN, "_extract_ref.lean")
    src = (
        "import TensorGuard.ReducedProduct\n"
        "open TensorGuard.RP\n"
        "def allNul : List Nullity := [.bot, .null, .notnull, .top]\n"
        "def allTag : List Tag := [\u27e8false,false\u27e9,\u27e8false,true\u27e9,"
        "\u27e8true,false\u27e9,\u27e8true,true\u27e9]\n"
        "def nulStr : Nullity \u2192 String\n"
        "  | .bot => \"bot\" | .null => \"null\" | .notnull => \"notnull\" | .top => \"top\"\n"
        "def bs : Bool \u2192 String | true => \"1\" | false => \"0\"\n"
        "def pvStr (p : PVal) : String := bs p.tag.mayNone ++ bs p.tag.mayOther "
        "++ \":\" ++ nulStr p.nul\n"
        "def gStr (p : PVal) : String := bs (mem .cnone p) ++ bs (mem .cobj p)\n"
        "def line (p : PVal) : String := pvStr p ++ \" -> \" ++ pvStr (reduce p) "
        "++ \" | gamma=\" ++ gStr p\n"
        "def table : String := String.intercalate \"\\n\" (allTag.flatMap "
        "(fun t => allNul.map (fun n => line \u27e8t,n\u27e9)))\n"
        "#eval IO.println table\n"
    )
    with open(script, "w") as fh:
        fh.write(src)
    try:
        proc = subprocess.run(
            ["lake", "env", "lean", "_extract_ref.lean"],
            cwd=_LEAN, capture_output=True, text=True, timeout=900,
        )
    finally:
        if os.path.exists(script):
            os.remove(script)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    live = [ln.strip() for ln in proc.stdout.strip().splitlines() if ln.strip()]
    with open(vrd._REFERENCE) as fh:
        committed = [ln.strip() for ln in fh if ln.strip()]
    assert live == committed, (
        "committed Lean-extracted reference table is stale:\n"
        f"live:\n{live}\ncommitted:\n{committed}")


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
