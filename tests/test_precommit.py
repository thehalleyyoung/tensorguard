"""Step 68 — pre-commit hook entry point."""

import torch  # noqa: F401

from src.precommit import main

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
_GOOD_CONV = (
    "import torch.nn as nn\n"
    "class Net(nn.Module):\n"
    "    def __init__(self):\n"
    "        super().__init__()\n"
    "        self.conv1 = nn.Conv2d(3, 8, 3)\n"
    "        self.conv2 = nn.Conv2d(8, 16, 3)\n"
    "    def forward(self, x):\n"
    "        return self.conv2(self.conv1(x))\n"
)


def test_blocks_commit_on_bug(tmp_path, capsys):
    bad = tmp_path / "bad.py"
    bad.write_text(_BAD, encoding="utf-8")
    rc = main([str(bad), "--input-shapes", "x=batch,10"])
    out = capsys.readouterr().out
    assert rc == 1
    assert "TensorGuard" in out
    assert f"{bad}:8" in out


def test_allows_commit_when_clean(tmp_path, capsys):
    good = tmp_path / "ok.py"
    good.write_text(_GOOD_CONV, encoding="utf-8")  # conv: rank auto-inferred
    rc = main([str(good)])
    out = capsys.readouterr().out
    assert rc == 0
    assert "no issues" in out


def test_respects_config_ignore(tmp_path, capsys):
    bad = tmp_path / "bad.py"
    bad.write_text(_BAD, encoding="utf-8")
    (tmp_path / "tensorguard.toml").write_text(
        '[tensorguard]\nignore = ["bad.py"]\n', encoding="utf-8"
    )
    rc = main([str(bad), "--input-shapes", "x=batch,10"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "verified 0 file(s)" in out


def test_no_args_defaults_to_cwd_clean(tmp_path, monkeypatch, capsys):
    # An empty directory has nothing to verify -> clean pass.
    monkeypatch.chdir(tmp_path)
    rc = main([])
    assert rc == 0
    assert "no issues" in capsys.readouterr().out
