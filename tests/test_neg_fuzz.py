"""Step 16 regression tests -- negative fuzzing (false-negative hunting).

Pins the invariants of the fault-injection recall hunt:

* injectors are pure functions of the source (deterministic);
* every *admitted* injected fault is genuine -- the mutated model really raises
  in eager PyTorch;
* TensorGuard's recall on genuine injected faults is perfect (no false
  negatives), and any miss would be root-cause tagged;
* the committed artifact regenerates byte-for-byte.

Live tests use a small base-seed budget; the full 120-seed run is pinned via
the committed artifact (`--check`).
"""

from __future__ import annotations

import os

from evaluation import neg_fuzz, diff_fuzz


def test_injectors_are_pure_and_deterministic():
    src, _ = diff_fuzz.build_model(0)
    for _, injector in neg_fuzz.INJECTORS:
        assert injector(src) == injector(src)


def test_each_family_can_produce_a_genuine_fault():
    seen = {name: False for name, _ in neg_fuzz.INJECTORS}
    for seed in range(60):
        src, shapes = diff_fuzz.build_model(seed)
        if not diff_fuzz.runtime_runs_clean(src, shapes):
            continue
        for name, injector in neg_fuzz.INJECTORS:
            mutated = injector(src)
            if mutated is None:
                continue
            if neg_fuzz.fault_is_genuine(mutated, shapes):
                seen[name] = True
    assert all(seen.values()), "families with no genuine fault: %s" % (
        [k for k, v in seen.items() if not v])


def test_admitted_faults_are_all_genuine():
    # Whatever is counted as 'genuine' must really fail at runtime.
    a = neg_fuzz.run(check=False, n_base=40, write=False)
    for fam, d in a["by_family"].items():
        assert d["genuine"] <= d["injected"]
    assert a["summary"]["genuine_faults"] >= 1


def test_recall_is_perfect_live_small_run():
    a = neg_fuzz.run(check=False, n_base=40, write=False)
    s = a["summary"]
    assert s["genuine_faults"] >= 1
    assert s["false_negatives"] == 0
    assert s["recall"] == 1.0
    assert not a["false_negatives"]


def test_every_false_negative_would_be_tagged():
    a = neg_fuzz.run(check=False, n_base=40, write=False)
    for m in a["false_negatives"]:
        assert m["root_cause"].strip()


def test_committed_artifact_is_up_to_date():
    assert os.path.exists(neg_fuzz.OUT_JSON)
    neg_fuzz.run(check=True)


def test_committed_artifact_reports_perfect_recall():
    import json
    with open(neg_fuzz.OUT_JSON, "r", encoding="utf-8") as fh:
        a = json.load(fh)
    s = a["summary"]
    assert s["genuine_faults"] >= 100
    assert s["false_negatives"] == 0
    assert s["recall"] == 1.0
    # all four families exercised with at least one genuine fault
    for name, _ in neg_fuzz.INJECTORS:
        assert a["by_family"][name]["genuine"] >= 1
