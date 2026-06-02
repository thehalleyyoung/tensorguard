"""Step 72 — baseline + inline suppression, proven against real torch findings."""

import json
import os

from src.baseline import (
    BASELINE_FILENAME,
    annotation_fingerprint,
    apply_baseline,
    baseline_payload,
    filter_inline,
    find_baseline_file,
    finding_fingerprint,
    is_suppressed_inline,
    load_baseline_fingerprints,
    parse_inline_suppressions,
    write_baseline,
)
from src.github_action import Annotation, run_action

# Real shape-mismatch model: fc1 outputs 20, fc2 expects 30 (line 8).
_BAD = (
    "import torch.nn as nn\n"
    "class Net(nn.Module):\n"
    "    def __init__(self):\n"
    "        super().__init__()\n"
    "        self.fc1 = nn.Linear(10, 20)\n"
    "        self.fc2 = nn.Linear(30, 5)\n"
    "    def forward(self, x):\n"
    "        return self.fc2(self.fc1(x))\n"
)


# --------------------------------------------------------------------------- #
# Fingerprint
# --------------------------------------------------------------------------- #
def test_fingerprint_is_line_independent():
    msg = "Layer fc2 (line 8) expects input dimension 30, but receives (batch, 20)"
    msg_moved = "Layer fc2 (line 42) expects input dimension 30, but receives (batch, 20)"
    fp1 = finding_fingerprint("m.py", msg)
    fp2 = finding_fingerprint("m.py", msg_moved)
    assert fp1 == fp2  # digits normalised away → stable across line shifts


def test_fingerprint_differs_by_file_and_message():
    a = finding_fingerprint("a.py", "Layer fc2 expects 30")
    b = finding_fingerprint("b.py", "Layer fc2 expects 30")
    c = finding_fingerprint("a.py", "Device mismatch on cuda")
    assert a != b
    assert a != c


def test_fingerprint_uses_relative_root():
    abs_path = "/tmp/proj/pkg/m.py"
    rel = finding_fingerprint(abs_path, "boom", root="/tmp/proj")
    # same finding written with the relative path directly must match
    assert rel == finding_fingerprint("pkg/m.py", "boom")


# --------------------------------------------------------------------------- #
# Inline suppression
# --------------------------------------------------------------------------- #
def test_parse_inline_suppress_all():
    src = "x = 1  # tensorguard: ignore\ny = 2\n"
    sup = parse_inline_suppressions(src)
    assert sup == {1: None}


def test_parse_inline_suppress_specific_rules():
    src = "z = 3  # tensorguard: ignore[shape, broadcast]\n"
    sup = parse_inline_suppressions(src)
    assert sup == {1: {"shape", "broadcast"}}


def test_parse_inline_tg_alias():
    src = "q = 4  # tg: ignore\n"
    assert parse_inline_suppressions(src) == {1: None}


def test_is_suppressed_inline_rule_specific():
    ann = Annotation("m.py", 3, None, "[SHAPE] dim mismatch")
    sup = {3: {"shape"}}
    assert is_suppressed_inline(ann, sup)
    # a different rule on the same line is NOT suppressed
    other = Annotation("m.py", 3, None, "[DEVICE] cuda vs cpu")
    assert not is_suppressed_inline(other, sup)


def test_filter_inline_splits():
    anns = [
        Annotation("m.py", 8, None, "[SHAPE] boom"),
        Annotation("m.py", 9, None, "[DEVICE] cuda"),
    ]
    src = "\n" * 7 + "bad = 1  # tensorguard: ignore\nok = 2\n"
    kept, suppressed = filter_inline(anns, src)
    assert [a.line for a in suppressed] == [8]
    assert [a.line for a in kept] == [9]


def test_run_action_inline_suppression_real_bug(tmp_path):
    # add a suppression comment to the failing forward line (line 8)
    suppressed_src = _BAD.replace(
        "        return self.fc2(self.fc1(x))\n",
        "        return self.fc2(self.fc1(x))  # tensorguard: ignore\n",
    )
    f = tmp_path / "bad.py"
    f.write_text(suppressed_src, encoding="utf-8")
    res = run_action([str(f)], input_shapes={"x": ("batch", 10)})
    assert res.total_issues == 0
    assert res.failed is False

    # without the comment the same model fails
    g = tmp_path / "bad2.py"
    g.write_text(_BAD, encoding="utf-8")
    res2 = run_action([str(g)], input_shapes={"x": ("batch", 10)})
    assert res2.total_issues == 1
    assert res2.failed is True


def test_inline_suppression_can_be_disabled(tmp_path):
    suppressed_src = _BAD.replace(
        "        return self.fc2(self.fc1(x))\n",
        "        return self.fc2(self.fc1(x))  # tensorguard: ignore\n",
    )
    f = tmp_path / "bad.py"
    f.write_text(suppressed_src, encoding="utf-8")
    res = run_action(
        [str(f)], input_shapes={"x": ("batch", 10)}, inline_suppression=False
    )
    assert res.total_issues == 1


# --------------------------------------------------------------------------- #
# Baseline
# --------------------------------------------------------------------------- #
def test_baseline_roundtrip_and_apply(tmp_path):
    f = tmp_path / "bad.py"
    f.write_text(_BAD, encoding="utf-8")
    res = run_action([str(f)], input_shapes={"x": ("batch", 10)})
    assert res.total_issues == 1 and res.failed

    bpath = tmp_path / BASELINE_FILENAME
    payload = write_baseline(str(bpath), res)
    assert payload["version"] == 1
    assert len(payload["fingerprints"]) == 1
    # file is valid JSON on disk
    on_disk = json.loads(bpath.read_text(encoding="utf-8"))
    assert on_disk["fingerprints"] == payload["fingerprints"]

    fps = load_baseline_fingerprints(str(bpath))
    suppressed = apply_baseline(res, fps)
    assert suppressed.total_issues == 0
    assert suppressed.failed is False  # baselined → no longer fails
    assert suppressed.files_checked == res.files_checked


def test_baseline_lets_new_findings_through(tmp_path):
    f = tmp_path / "bad.py"
    f.write_text(_BAD, encoding="utf-8")
    res = run_action([str(f)], input_shapes={"x": ("batch", 10)})
    # baseline contains an UNRELATED fingerprint, so the real bug is "new"
    suppressed = apply_baseline(res, {"deadbeefdeadbeef"})
    assert suppressed.total_issues == 1
    assert suppressed.failed is True


def test_run_action_with_baseline_path(tmp_path):
    f = tmp_path / "bad.py"
    f.write_text(_BAD, encoding="utf-8")
    res = run_action([str(f)], input_shapes={"x": ("batch", 10)})
    bpath = tmp_path / BASELINE_FILENAME
    write_baseline(str(bpath), res)

    # passing the baseline path suppresses the known finding
    res2 = run_action(
        [str(f)], input_shapes={"x": ("batch", 10)}, baseline=str(bpath)
    )
    assert res2.total_issues == 0
    assert res2.failed is False


def test_find_baseline_file_walks_up(tmp_path):
    (tmp_path / BASELINE_FILENAME).write_text("{}", encoding="utf-8")
    sub = tmp_path / "a" / "b"
    sub.mkdir(parents=True)
    found = find_baseline_file(str(sub))
    assert found == str(tmp_path / BASELINE_FILENAME)
    assert find_baseline_file(str(tmp_path / "nope")) is None or os.path.exists(
        find_baseline_file(str(tmp_path / "nope")) or ""
    )


def test_load_missing_baseline_is_empty():
    assert load_baseline_fingerprints("/nonexistent/path.json") == set()


def test_annotation_fingerprint_matches_finding():
    ann = Annotation("m.py", 8, None, "[SHAPE] dim 30 vs 20")
    assert annotation_fingerprint(ann) == finding_fingerprint(
        "m.py", "[SHAPE] dim 30 vs 20"
    )
