"""Step 163 — ``pytest --tensorguard`` verifies the *modules under test*.

Proves the import-discovery behaviour against a real subprocess pytest run, not
just the helper. A throwaway project contains two buggy models:

* ``imported_buggy.py`` — imported by the test (i.e. exercised by the suite);
* ``orphan_buggy.py`` — present in the tree but never imported.

With the default ``--tensorguard`` (modules-under-test mode) the gate flags only
the *imported* model and reports its filename — orphan code is not scanned. With
``--tensorguard-rootdir`` the whole tree is scanned and the orphan is flagged too.
A control run where the test imports a *clean* model passes even though the
orphan buggy file still sits in the tree, proving the gate follows imports rather
than blindly walking the directory.
"""

from __future__ import annotations

import os
import subprocess
import sys

import pytest

pytest.importorskip("torch")

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

_BUGGY = (
    "import torch.nn as nn\n"
    "class Net(nn.Module):\n"
    "    def __init__(self):\n"
    "        super().__init__()\n"
    "        self.fc1 = nn.Linear(10, 20)\n"
    "        self.fc2 = nn.Linear(30, 5)\n"
    "    def forward(self, x):\n"
    "        return self.fc2(self.fc1(x))\n"
)
_CLEAN = (
    "import torch.nn as nn\n"
    "class Net(nn.Module):\n"
    "    def __init__(self):\n"
    "        super().__init__()\n"
    "        self.fc1 = nn.Linear(10, 20)\n"
    "        self.fc2 = nn.Linear(20, 5)\n"
    "    def forward(self, x):\n"
    "        return self.fc2(self.fc1(x))\n"
)


def _project(tmp_path, imported_src, imported_name):
    proj = tmp_path / "proj"
    proj.mkdir()
    # conftest puts the project root on sys.path so the test can import the model.
    (proj / "conftest.py").write_text(
        "import os, sys\nsys.path.insert(0, os.path.dirname(__file__))\n",
        encoding="utf-8",
    )
    (proj / f"{imported_name}.py").write_text(imported_src, encoding="utf-8")
    (proj / "orphan_buggy.py").write_text(_BUGGY, encoding="utf-8")
    (proj / "test_model.py").write_text(
        f"import {imported_name}\n"
        "def test_constructs():\n"
        f"    assert {imported_name}.Net() is not None\n",
        encoding="utf-8",
    )
    return proj


def _run(proj, extra):
    env = dict(os.environ)
    env["PYTHONPATH"] = _REPO + os.pathsep + env.get("PYTHONPATH", "")
    # Load the plugin explicitly and disable entry-point autoload so an editable
    # install of TensorGuard (which also exposes the pytest11 entry point) cannot
    # double-register the same module under two names.
    env["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
    return subprocess.run(
        [sys.executable, "-m", "pytest", "-p", "src.pytest_tensorguard",
         "-q", "-p", "no:cacheprovider", "--tensorguard",
         "--tensorguard-input-shapes=x=batch,10", *extra],
        cwd=str(proj), env=env, capture_output=True, text=True, timeout=300,
    )


def test_only_imported_model_is_verified(tmp_path):
    proj = _project(tmp_path, _BUGGY, "imported_buggy")
    proc = _run(proj, [])
    out = proc.stdout + proc.stderr
    assert proc.returncode != 0, out
    assert "imported_buggy.py" in out, out
    # The orphan buggy file was never imported, so the modules-under-test gate
    # must NOT report it.
    assert "orphan_buggy.py" not in out, out


def test_rootdir_mode_also_flags_orphan(tmp_path):
    proj = _project(tmp_path, _BUGGY, "imported_buggy")
    proc = _run(proj, ["--tensorguard-rootdir"])
    out = proc.stdout + proc.stderr
    assert proc.returncode != 0, out
    # Whole-tree scan: both buggy files are reported.
    assert "imported_buggy.py" in out
    assert "orphan_buggy.py" in out


def test_clean_imported_model_passes_despite_orphan(tmp_path):
    proj = _project(tmp_path, _CLEAN, "imported_clean")
    proc = _run(proj, [])
    out = proc.stdout + proc.stderr
    # The exercised model is clean; the orphan buggy file is not imported, so the
    # modules-under-test gate passes.
    assert proc.returncode == 0, out
    assert "no issues" in out, out
