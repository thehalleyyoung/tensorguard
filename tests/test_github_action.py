"""Step 66 — production GitHub Action: annotation rendering + PR gating.

Proves the workflow-command escaping, the file→annotation mapping (preferring
diagnostics, falling back to bugs), de-duplication, the env-driven ``main``
(GITHUB_OUTPUT / GITHUB_STEP_SUMMARY / exit code), and an end-to-end run against
real torch source where a buggy file annotates the correct line and a clean file
yields no annotations and a zero exit.
"""

import os
import types

import torch  # noqa: F401

from src.github_action import (
    Annotation,
    annotations_for_result,
    escape_data,
    escape_property,
    format_annotation,
    main,
    run_action,
)

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
_GOOD = (
    "import torch.nn as nn\n"
    "class Net(nn.Module):\n"
    "    def __init__(self):\n"
    "        super().__init__()\n"
    "        self.conv1 = nn.Conv2d(3, 8, 3)\n"
    "        self.conv2 = nn.Conv2d(8, 16, 3)\n"
    "    def forward(self, x):\n"
    "        return self.conv2(self.conv1(x))\n"
)


def test_escaping():
    assert escape_data("a\nb\rc%d") == "a%0Ab%0Dc%25d"
    # property escaping also encodes : and ,
    assert escape_property("a:b,c") == "a%3Ab%2Cc"


def test_format_annotation_basic():
    s = format_annotation("m.py", 8, "bad shape", col=4, level="error", title="TG")
    assert s.startswith("::error ")
    assert "file=m.py" in s
    assert "line=8" in s
    assert "col=4" in s
    assert "title=TG" in s
    assert s.endswith("::bad shape")


def test_format_annotation_omits_zero_line_and_col():
    s = format_annotation("m.py", 0, "x", col=0)
    assert "line=" not in s
    assert "col=" not in s
    assert s == "::error file=m.py::x"


def test_format_annotation_multiline_message_escaped():
    s = format_annotation("m.py", 3, "line1\nline2")
    assert "\n" not in s
    assert "line1%0Aline2" in s


def test_annotations_prefer_diagnostics():
    Diag = types.SimpleNamespace
    result = types.SimpleNamespace(
        diagnostics=[
            Diag(source_line=8, source_col=8, message="fc2 mismatch", severity="error"),
            Diag(source_line=0, source_col=0, message="skip me", severity="error"),
        ],
        bugs=[types.SimpleNamespace(location=types.SimpleNamespace(line=8, column=8),
                                    message="[X] bug", severity="error")],
    )
    anns = annotations_for_result("m.py", result)
    assert len(anns) == 1  # the line-0 diagnostic is dropped
    assert anns[0].line == 8
    assert anns[0].message == "fc2 mismatch"  # came from diagnostics, not bugs


def test_annotations_fall_back_to_bugs_when_no_diagnostics():
    Loc = types.SimpleNamespace
    result = types.SimpleNamespace(
        diagnostics=[],
        bugs=[
            types.SimpleNamespace(location=Loc(line=8, column=8),
                                  message="[X] first\nsecond", severity="error"),
            types.SimpleNamespace(location=Loc(line=0, column=0),
                                  message="[X] no-loc", severity="error"),
        ],
    )
    anns = annotations_for_result("m.py", result)
    assert len(anns) == 1
    assert anns[0].line == 8
    assert anns[0].message == "[X] first"  # only the first line is used


def test_annotations_dedupe():
    Diag = types.SimpleNamespace
    d = Diag(source_line=8, source_col=8, message="dup", severity="error")
    result = types.SimpleNamespace(diagnostics=[d, d], bugs=[])
    assert len(annotations_for_result("m.py", result)) == 1


def test_run_action_end_to_end_bug(tmp_path):
    bad = tmp_path / "bad.py"
    bad.write_text(_BAD, encoding="utf-8")
    res = run_action([str(bad)], input_shapes={"x": ("batch", 10)})
    assert res.files_checked == 1
    assert res.files_with_issues == 1
    assert res.total_issues >= 1
    assert res.failed
    rendered = res.render_annotations()
    assert "::error " in rendered
    assert f"file={bad}" in rendered
    assert "line=8" in rendered


def test_run_action_end_to_end_clean(tmp_path):
    good = tmp_path / "convnet.py"
    good.write_text(_GOOD, encoding="utf-8")
    # Conv model: rank auto-inferred, no input_shapes needed.
    res = run_action([str(good)])
    assert res.files_checked == 1
    assert res.total_issues == 0
    assert not res.failed
    assert res.render_annotations() == ""


def test_run_action_fail_on_never(tmp_path):
    bad = tmp_path / "bad.py"
    bad.write_text(_BAD, encoding="utf-8")
    res = run_action([str(bad)], input_shapes={"x": ("batch", 10)}, fail_on="never")
    assert res.total_issues >= 1
    assert not res.failed  # annotate-only mode never fails the gate


def test_run_action_honors_config_ignore(tmp_path):
    bad = tmp_path / "bad.py"
    bad.write_text(_BAD, encoding="utf-8")
    (tmp_path / "tensorguard.toml").write_text(
        '[tensorguard]\nignore = ["bad.py"]\n', encoding="utf-8"
    )
    res = run_action([str(bad)], input_shapes={"x": ("batch", 10)})
    assert res.files_checked == 0  # ignored before verification
    assert res.total_issues == 0


def test_run_action_directory_recursion(tmp_path):
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "bad.py").write_text(_BAD, encoding="utf-8")
    # A safe Linear model that verifies clean under the same x=batch,10 shape.
    safe_linear = (
        "import torch.nn as nn\n"
        "class Net(nn.Module):\n"
        "    def __init__(self):\n"
        "        super().__init__()\n"
        "        self.fc1 = nn.Linear(10, 20)\n"
        "        self.fc2 = nn.Linear(20, 5)\n"
        "    def forward(self, x):\n"
        "        return self.fc2(self.fc1(x))\n"
    )
    (tmp_path / "pkg" / "ok.py").write_text(safe_linear, encoding="utf-8")
    res = run_action([str(tmp_path)], input_shapes={"x": ("batch", 10)})
    assert res.files_checked == 2
    assert res.files_with_issues == 1


def test_main_env_driven(tmp_path, monkeypatch, capsys):
    bad = tmp_path / "bad.py"
    bad.write_text(_BAD, encoding="utf-8")
    out_file = tmp_path / "gh_output"
    summary_file = tmp_path / "gh_summary"
    monkeypatch.setenv("INPUT_PATHS", str(bad))
    monkeypatch.setenv("INPUT_INPUT_SHAPES", "x=batch,10")
    monkeypatch.setenv("INPUT_SOUNDNESS_MODE", "balanced")
    monkeypatch.setenv("INPUT_FAIL_ON", "any")
    monkeypatch.setenv("GITHUB_OUTPUT", str(out_file))
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(summary_file))

    rc = main()
    assert rc == 1  # gate fails on the bug
    printed = capsys.readouterr().out
    assert "::error " in printed

    outputs = out_file.read_text(encoding="utf-8")
    assert "issues=" in outputs
    assert "files-checked=1" in outputs
    summary = summary_file.read_text(encoding="utf-8")
    assert "TensorGuard" in summary
    assert "issue" in summary


def test_main_clean_exit_zero(tmp_path, monkeypatch, capsys):
    good = tmp_path / "ok.py"
    good.write_text(_GOOD, encoding="utf-8")
    monkeypatch.setenv("INPUT_PATHS", str(good))
    monkeypatch.delenv("GITHUB_OUTPUT", raising=False)
    monkeypatch.delenv("GITHUB_STEP_SUMMARY", raising=False)
    rc = main()
    assert rc == 0
    assert capsys.readouterr().out.strip() == ""
