"""Tests for the Step 125 paper-evidence index + make target."""

from __future__ import annotations

from pathlib import Path

import pytest

import reproducibility.paper_evidence_index as pei

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


@pytest.fixture(scope="module")
def data():
    return pei.measure()


def test_make_target_exists_and_calls_index():
    mk = (REPO / "Makefile").read_text()
    assert "\npaper-evidence:" in mk
    assert "reproducibility/paper_evidence_index.py" in mk
    assert "paper-evidence" in mk.split(".PHONY")[1].split("\n")[0]


def test_index_covers_all_deterministic_md_json_artifacts(data):
    import reproducibility.reproduce_all as ra
    stems = {Path(p).stem for p in ra.GENERATED_DETERMINISTIC
             if Path(p).suffix in (".json", ".md")}
    indexed = {e["stem"] for e in data["evidence"]}
    assert indexed == stems


def test_all_artifacts_present(data):
    assert data["all_artifacts_present"]
    for e in data["evidence"]:
        assert e["json_present"] or e["md_present"]


def test_index_includes_itself_and_renders_table(data):
    me = next(e for e in data["evidence"] if e["stem"] == "paper_evidence_index")
    assert me["generator"] == "reproducibility/paper_evidence_index.py"
    assert me["renders_table"]


def test_table_count_is_meaningful(data):
    # The catalogue should contain many rendered tables, not a handful.
    assert data["n_with_table"] >= 20
    assert data["n_evidence_items"] >= data["n_with_table"]


def test_generators_resolve_for_harness_artifacts(data):
    # Every artifact whose stem matches a reproducibility/<stem>.py must resolve.
    for e in data["evidence"]:
        cand = REPO / "reproducibility" / f"{e['stem']}.py"
        if cand.exists():
            assert e["generator"] == f"reproducibility/{e['stem']}.py"
            assert e["generator_present"]


def test_no_volatile_keys_and_deterministic(data):
    for k in _walk_keys(data):
        assert not any(tok in k.lower() for tok in VOLATILE), k
    assert pei.measure() == data


def test_check_mode_byte_identical():
    assert pei.run(check=True) == 0
