"""Step 64 — per-repo configuration (`tensorguard.toml`).

Covers the pure loader/precedence/ignore logic and an end-to-end check that a
config-driven ignore_rules entry suppresses a real bug via the CLI verify path.
"""

import sys
import types

import torch  # noqa: F401

from src.tg_config import (
    TGConfig,
    filter_result,
    find_config_file,
    is_ignored_file,
    is_ignored_rule,
    load_tg_config,
    parse_config,
    rule_tag,
)


def _write(p, text):
    p.write_text(text, encoding="utf-8")
    return p


def test_parse_config_full():
    data = {
        "soundness_mode": "sound",
        "infer_inputs": False,
        "high_confidence": True,
        "cegar_iterations": 12,
        "ignore": ["experiments/**", "legacy/old.py"],
        "ignore_rules": ["CEGAR-Real-Bug", "shape-incompatible"],
        "checks": {"devices": False, "phases": True, "gradients": False},
    }
    cfg = parse_config(data, source_path="/repo/tensorguard.toml")
    assert cfg.soundness_mode == "sound"
    assert cfg.infer_inputs is False
    assert cfg.high_confidence is True
    assert cfg.cegar_iterations == 12
    assert cfg.check_devices is False
    assert cfg.check_phases is True
    assert cfg.check_gradients is False
    assert cfg.ignore_files == ["experiments/**", "legacy/old.py"]
    # rule tags are lowercased
    assert cfg.ignore_rules == ["cegar-real-bug", "shape-incompatible"]
    assert cfg.project_root == "/repo"


def test_parse_config_rejects_bad_mode_and_defaults():
    cfg = parse_config({"soundness_mode": "bogus"})
    assert cfg.soundness_mode is None  # invalid mode ignored
    # all checks default to enabled
    assert cfg.check_devices and cfg.check_phases and cfg.check_gradients
    assert cfg.cegar_iterations is None


def test_find_and_load_standalone(tmp_path):
    sub = tmp_path / "pkg" / "models"
    sub.mkdir(parents=True)
    _write(tmp_path / "tensorguard.toml", '[tensorguard]\nsoundness_mode = "heuristic"\n')
    model = _write(sub / "m.py", "x=1\n")
    found = find_config_file(model)
    assert found is not None and found.name == "tensorguard.toml"
    cfg = load_tg_config(str(model))
    assert cfg.soundness_mode == "heuristic"


def test_load_from_pyproject(tmp_path):
    _write(
        tmp_path / "pyproject.toml",
        "[tool.tensorguard]\ncegar_iterations = 7\nignore_rules = ['x']\n",
    )
    model = _write(tmp_path / "m.py", "x=1\n")
    found = find_config_file(model)
    assert found is not None and found.name == "pyproject.toml"
    cfg = load_tg_config(str(model))
    assert cfg.cegar_iterations == 7
    assert cfg.ignore_rules == ["x"]


def test_pyproject_without_table_is_skipped(tmp_path):
    _write(tmp_path / "pyproject.toml", "[tool.black]\nline-length = 88\n")
    model = _write(tmp_path / "m.py", "x=1\n")
    assert find_config_file(model) is None
    cfg = load_tg_config(str(model))
    assert cfg == TGConfig()  # all defaults


def test_standalone_wins_over_pyproject(tmp_path):
    _write(tmp_path / "pyproject.toml", '[tool.tensorguard]\nsoundness_mode = "sound"\n')
    _write(tmp_path / "tensorguard.toml", '[tensorguard]\nsoundness_mode = "heuristic"\n')
    model = _write(tmp_path / "m.py", "x=1\n")
    cfg = load_tg_config(str(model))
    assert cfg.soundness_mode == "heuristic"


def test_is_ignored_file(tmp_path):
    cfg = TGConfig(
        ignore_files=["experiments/**", "scratch.py"],
        project_root=str(tmp_path),
    )
    assert is_ignored_file(cfg, str(tmp_path / "experiments" / "a" / "m.py"))
    assert is_ignored_file(cfg, str(tmp_path / "scratch.py"))
    assert not is_ignored_file(cfg, str(tmp_path / "src" / "model.py"))
    # empty ignore list -> never ignored
    assert not is_ignored_file(TGConfig(), str(tmp_path / "anything.py"))


def test_rule_tag_and_is_ignored_rule():
    assert rule_tag("[SHAPE-INCOMPATIBLE] Linear expects ...") == "shape-incompatible"
    assert rule_tag("no tag here") == ""
    cfg = TGConfig(ignore_rules=["shape-incompatible"])
    assert is_ignored_rule(cfg, "[SHAPE-INCOMPATIBLE] foo")
    assert not is_ignored_rule(cfg, "[CEGAR-REAL-BUG] bar")


def test_filter_result_drops_ignored_bugs():
    Loc = types.SimpleNamespace
    Bug = types.SimpleNamespace
    Diag = types.SimpleNamespace
    result = types.SimpleNamespace(
        bugs=[
            Bug(message="[SHAPE-INCOMPATIBLE] a", location=Loc(line=8), severity="error"),
            Bug(message="[CEGAR-REAL-BUG] b", location=Loc(line=8), severity="error"),
            Bug(message="[OTHER] c", location=Loc(line=12), severity="error"),
        ],
        diagnostics=[Diag(source_line=8), Diag(source_line=12)],
    )
    cfg = TGConfig(ignore_rules=["shape-incompatible", "cegar-real-bug"])
    filter_result(cfg, result)
    assert len(result.bugs) == 1
    assert result.bugs[0].message.startswith("[OTHER]")
    # the line-8 diagnostic is dropped, line-12 kept
    assert [d.source_line for d in result.diagnostics] == [12]


def test_end_to_end_ignore_rules_via_cli(tmp_path, monkeypatch, capsys):
    model = _write(
        tmp_path / "m.py",
        "import torch.nn as nn\n"
        "class Net(nn.Module):\n"
        "    def __init__(self):\n"
        "        super().__init__()\n"
        "        self.fc1 = nn.Linear(10, 20)\n"
        "        self.fc2 = nn.Linear(30, 5)\n"
        "    def forward(self, x):\n"
        "        return self.fc2(self.fc1(x))\n",
    )
    _write(
        tmp_path / "tensorguard.toml",
        '[tensorguard]\nignore_rules = ["shape-incompatible", "cegar-real-bug"]\n',
    )
    from src.cli.main import VerifyCommand
    import argparse

    parser = argparse.ArgumentParser()
    cmd = VerifyCommand()
    cmd.register(parser)
    args = parser.parse_args([str(model), "-s", "x=batch,10", "--no-color"])
    rc = cmd.execute(args)
    out = capsys.readouterr().out
    assert rc == 0
    assert "verified safe" in out
