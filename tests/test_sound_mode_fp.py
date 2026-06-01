"""Step 13 regression -- sound mode never false-alarms on clean code.

Locks in the committed `evaluation/sound_mode_fp.json` artifact and re-derives
the verdicts live: TensorGuard in strict `sound` mode must report zero bugs on
every clean, executing model, with non-trivial SAFE coverage so the result is
not a vacuous "always abstain".
"""

from __future__ import annotations

import json
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import pytest  # noqa: E402

from evaluation import sound_mode_fp as smfp  # noqa: E402


@pytest.fixture(scope="module")
def committed():
    with open(smfp.OUT_JSON, "r", encoding="utf-8") as fh:
        return json.load(fh)


def test_committed_zero_false_positives(committed):
    assert committed["summary"]["false_positives"] == 0
    assert committed["summary"]["false_positive_rate"] == 0.0
    assert committed["false_positive_ids"] == []


def test_committed_corpus_is_large_and_diverse(committed):
    s, m = committed["summary"], committed["meta"]
    assert s["total"] >= 50  # a real FP hunt, not a token sample
    assert m["n_real_clean"] == 8
    assert len(m["families"]) >= 6


def test_committed_coverage_is_non_trivial(committed):
    """Zero-FP must come with real verification, not blanket abstention."""
    s = committed["summary"]
    assert s["verified_safe"] >= s["total"] // 2
    assert s["coverage_safe"] >= 0.5


def test_every_family_has_zero_false_positives(committed):
    for family, counts in committed["by_family"].items():
        assert counts["REFUTED"] == 0, family


def test_generated_corpus_is_deterministic():
    a = smfp.generate_corpus()
    b = smfp.generate_corpus()
    assert [m["id"] for m in a] == [m["id"] for m in b]
    assert [m["source"] for m in a] == [m["source"] for m in b]


def test_all_admitted_models_actually_execute_clean():
    """The corpus invariant: every admitted model runs in eager PyTorch."""
    corpus = smfp.generate_corpus()
    assert len(corpus) >= 50
    # Spot-check a deterministic slice to keep the test fast.
    for m in corpus[::7]:
        shapes = {k: tuple(v) for k, v in m["input_shapes"].items()}
        assert smfp._executes_clean(m["source"], shapes), m["id"]


def test_live_sound_mode_has_no_false_positive_on_real_clean_half():
    """Direct re-derivation on the frozen hand-written clean models."""
    from real_benchmarks import load
    for item in load.load_items():
        if item["label"] != "clean":
            continue
        src = load.read_source(item)
        verdict, bug_count = smfp._sound_verdict(src, item["input_shapes"])
        assert bug_count == 0, item["id"]
        assert verdict in ("SAFE", "ABSTAIN"), (item["id"], verdict)


def test_artifact_regenerates_identically():
    with open(smfp.OUT_JSON, "r", encoding="utf-8") as fh:
        before = fh.read()
    smfp.run(check=True)  # raises SystemExit if stale
    with open(smfp.OUT_JSON, "r", encoding="utf-8") as fh:
        assert fh.read() == before
