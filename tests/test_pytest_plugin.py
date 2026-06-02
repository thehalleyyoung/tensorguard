"""Step 68 — ``pytest --tensorguard`` plugin.

Unit-tests the session-gate helper with a stand-in config, then runs the plugin
end-to-end as a real subprocess pytest invocation to prove it fails the session
when a model has a bug and passes when models are clean.
"""

import os
import subprocess
import sys

import torch  # noqa: F401

from src.pytest_tensorguard import run_session_check

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

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


class _DummyConfig:
    def __init__(self, options, rootpath):
        self._options = options
        self.rootpath = rootpath

    def getoption(self, name):
        return self._options.get(name)


def _opts(paths=None, shapes="", mode="balanced"):
    return {
        "tensorguard": True,
        "tensorguard_paths": paths or [],
        "tensorguard_input_shapes": shapes,
        "tensorguard_soundness_mode": mode,
    }


def test_run_session_check_flags_bug(tmp_path):
    (tmp_path / "bad.py").write_text(_BAD, encoding="utf-8")
    config = _DummyConfig(_opts(paths=[str(tmp_path)], shapes="x=batch,10"), tmp_path)
    failed, lines = run_session_check(config)
    assert failed
    assert any("issue(s)" in ln for ln in lines)
    assert any("bad.py:8" in ln for ln in lines)


def test_run_session_check_clean(tmp_path):
    (tmp_path / "ok.py").write_text(_GOOD_CONV, encoding="utf-8")
    config = _DummyConfig(_opts(paths=[str(tmp_path)]), tmp_path)
    failed, lines = run_session_check(config)
    assert not failed
    assert any("no issues" in ln for ln in lines)


def test_run_session_check_defaults_to_rootpath(tmp_path):
    (tmp_path / "bad.py").write_text(_BAD, encoding="utf-8")
    # No explicit --tensorguard-path: should fall back to rootpath (tmp_path).
    config = _DummyConfig(_opts(shapes="x=batch,10"), tmp_path)
    failed, _ = run_session_check(config)
    assert failed


def _run_pytest(workdir, extra_args):
    env = dict(os.environ)
    env["PYTHONPATH"] = _REPO + os.pathsep + env.get("PYTHONPATH", "")
    # Disable entry-point autoload so an editable install of TensorGuard does not
    # register the pytest11 plugin a second time (we load it explicitly via -p).
    env["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
    # A trivial passing test so pytest has something to collect.
    (workdir / "test_trivial.py").write_text(
        "def test_ok():\n    assert True\n", encoding="utf-8"
    )
    return subprocess.run(
        [sys.executable, "-m", "pytest", "-p", "src.pytest_tensorguard",
         "-q", "-p", "no:cacheprovider", str(workdir / "test_trivial.py"),
         *extra_args],
        cwd=str(workdir), env=env, capture_output=True, text=True, timeout=300,
    )


def test_plugin_subprocess_fails_session_on_bug(tmp_path):
    (tmp_path / "bad.py").write_text(_BAD, encoding="utf-8")
    proc = _run_pytest(
        tmp_path,
        ["--tensorguard", f"--tensorguard-path={tmp_path}",
         "--tensorguard-input-shapes=x=batch,10"],
    )
    combined = proc.stdout + proc.stderr
    assert proc.returncode != 0, combined
    assert "TensorGuard" in combined
    assert "bad.py" in combined


def test_plugin_subprocess_passes_without_flag(tmp_path):
    # Same buggy file present, but without --tensorguard the gate is inactive.
    (tmp_path / "bad.py").write_text(_BAD, encoding="utf-8")
    proc = _run_pytest(tmp_path, [])
    combined = proc.stdout + proc.stderr
    assert proc.returncode == 0, combined
