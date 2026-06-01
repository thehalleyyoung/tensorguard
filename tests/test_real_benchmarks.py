"""Regression tests for the frozen ground-truth benchmark corpus.

These tests guarantee that:
  1. The corpus on disk matches its frozen manifest (content hashes).
  2. The manifest is internally consistent and well-formed.
  3. Re-running the generator is deterministic (no uncommitted drift).
  4. TensorGuard's verdict on every item matches its frozen ground-truth label.
"""

import os
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
RB_DIR = os.path.join(REPO, "real_benchmarks")
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from real_benchmarks import build_manifest, corpus_def, load  # noqa: E402


def test_manifest_loads_and_is_frozen():
    m = load.load_manifest()
    assert m["meta"]["frozen"] is True
    assert m["meta"]["version"] == corpus_def.CORPUS_VERSION
    assert m["meta"]["total"] == len(m["items"])
    assert m["meta"]["clean"] + m["meta"]["buggy"] == m["meta"]["total"]


def test_corpus_is_balanced_and_nonempty():
    items = load.load_manifest()["items"]
    clean = [i for i in items if i["label"] == "clean"]
    buggy = [i for i in items if i["label"] == "buggy"]
    assert len(clean) >= 8
    assert len(buggy) >= 8
    # ids are unique
    ids = [i["id"] for i in items]
    assert len(ids) == len(set(ids))


def test_integrity_hashes_match_frozen_manifest():
    # Raises CorpusIntegrityError on any drift/missing file.
    load.verify_integrity()


def test_every_item_has_required_fields():
    items = load.load_manifest()["items"]
    required = {
        "id", "label", "domain", "category", "provenance_type", "source_url",
        "note", "input_shapes", "expected_verdict", "check_devices",
        "check_gradients", "repro_file", "sha256",
    }
    for it in items:
        assert required <= set(it), f"{it['id']} missing {required - set(it)}"
        assert it["label"] in ("clean", "buggy")
        assert it["expected_verdict"] == ("SAFE" if it["label"] == "clean" else "UNSAFE")
        assert os.path.exists(os.path.join(RB_DIR, it["repro_file"]))


def test_buggy_items_have_provenance():
    items = load.load_manifest()["items"]
    for it in items:
        if it["label"] != "buggy":
            continue
        if it["provenance_type"] == "pytorch_issue":
            assert it["source_url"] and "github.com/pytorch/pytorch" in it["source_url"]
        else:
            assert it["provenance_type"] == "canonical_pattern"


def test_generator_is_deterministic():
    # Re-render in memory and compare to the frozen hashes -- must not drift.
    rebuilt = build_manifest.build(write_files=False)
    frozen = {i["id"]: i["sha256"] for i in load.load_manifest()["items"]}
    for it in rebuilt["items"]:
        assert it["sha256"] == frozen[it["id"]], (
            f"{it['id']} would change on regeneration; "
            f"bump CORPUS_VERSION and re-freeze."
        )


@pytest.mark.parametrize("item", load.load_manifest()["items"], ids=lambda i: i["id"])
def test_tensorguard_verdict_matches_label(item):
    result = load.verify_item(item)
    actual = "UNSAFE" if result.bug_count > 0 else "SAFE"
    assert actual == item["expected_verdict"], (
        f"{item['id']}: expected {item['expected_verdict']} but TensorGuard "
        f"returned {actual} ({result.bug_count} bugs)"
    )


def test_check_corpus_all_match():
    ok, rows = load.check_corpus()
    failures = [r for r in rows if not r["match"]]
    assert ok, f"label mismatches: {failures}"
    assert len(rows) == load.load_manifest()["meta"]["total"]
