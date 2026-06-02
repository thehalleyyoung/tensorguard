"""Step 161 — `pip install tensorguard` ships a real, importable public package.

The distribution is named ``tensorguard``; a user expects ``import tensorguard``
(not ``import src``) to give the stability-guaranteed API. These tests prove that
contract against a *real built wheel*:

* a fast, offline test builds the wheel with ``python -m build --no-isolation``
  (build deps are already present) and asserts the archive ships
  ``tensorguard/__init__.py`` and lists ``tensorguard`` in ``top_level.txt``;
* a ``slow`` test installs that wheel with ``pip install --no-deps`` into an
  isolated target and, in a *fresh* interpreter whose only added path is that
  target, imports ``tensorguard`` and verifies a real shape bug end-to-end.

No network is used (``--no-isolation`` / ``--no-deps``; z3 comes from the
ambient environment), so the test is deterministic in CI.
"""

from __future__ import annotations

import glob
import os
import subprocess
import sys
import sysconfig

import pytest

import tensorguard

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_top_level_import_exposes_public_api():
    # The documented adoption path from the README works against the package.
    for name in (
        "verify_architecture",
        "analyze",
        "AnalysisResult",
        "guarded_compile",
        "make_tensorguard_backend",
        "TensorGuardViolation",
        "__version__",
    ):
        assert name in tensorguard.__all__, f"{name} missing from tensorguard.__all__"
        assert hasattr(tensorguard, name), f"tensorguard.{name} not importable"


def test_changelog_exists_and_is_semver_keepachangelog():
    path = os.path.join(_REPO, "CHANGELOG.md")
    assert os.path.exists(path)
    text = open(path, encoding="utf-8").read()
    assert "Keep a Changelog" in text
    assert "Semantic Versioning" in text
    assert "[0.1.0]" in text
    assert tensorguard.__version__ in text


def _build_wheel(outdir: str) -> str:
    proc = subprocess.run(
        [sys.executable, "-m", "build", "--no-isolation", "--wheel", "--outdir", outdir],
        cwd=_REPO,
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    wheels = glob.glob(os.path.join(outdir, "tensorguard-*.whl"))
    assert wheels, f"no wheel built: {proc.stdout}\n{proc.stderr}"
    return wheels[0]


def _have_build() -> bool:
    try:
        import build  # noqa: F401

        return True
    except Exception:
        return False


@pytest.mark.skipif(not _have_build(), reason="`build` not installed")
def test_built_wheel_ships_importable_tensorguard_package(tmp_path):
    import zipfile

    wheel = _build_wheel(str(tmp_path))
    with zipfile.ZipFile(wheel) as zf:
        names = zf.namelist()
        assert "tensorguard/__init__.py" in names, names
        top_level = next(n for n in names if n.endswith("top_level.txt"))
        assert "tensorguard" in zf.read(top_level).decode().split()


@pytest.mark.slow
@pytest.mark.skipif(not _have_build(), reason="`build` not installed")
def test_clean_install_imports_and_verifies(tmp_path):
    wheel = _build_wheel(str(tmp_path / "dist"))
    target = tmp_path / "site"
    proc = subprocess.run(
        [sys.executable, "-m", "pip", "install", "--no-deps", "--target",
         str(target), wheel],
        capture_output=True, text=True, timeout=600,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr

    # Fresh interpreter: only the install target plus the ambient stdlib/site
    # (for z3/torch) are visible — crucially NOT the repo checkout.
    snippet = (
        "import tensorguard\n"
        "assert %r in tensorguard.__file__, tensorguard.__file__\n"
        "r = tensorguard.verify_architecture('''\n"
        "import torch.nn as nn\n"
        "class Net(nn.Module):\n"
        "    def __init__(self):\n"
        "        super().__init__()\n"
        "        self.a = nn.Linear(10, 20)\n"
        "        self.b = nn.Linear(30, 5)\n"
        "    def forward(self, x):\n"
        "        return self.b(self.a(x))\n"
        "''', input_shapes={'x': ('bb', 10)})\n"
        "assert r.verdict == 'UNSAFE', r.verdict\n"
        "assert r.bugs, 'no bugs reported'\n"
        "print('OK', tensorguard.__version__)\n"
    ) % str(target)

    env = dict(os.environ)
    # PYTHONPATH = install target + the ambient site-packages (z3/torch), but not
    # the repo. cwd is somewhere neutral so `src`/`tensorguard` cannot shadow.
    site = sysconfig.get_paths()["purelib"]
    env["PYTHONPATH"] = os.pathsep.join([str(target), site])
    proc = subprocess.run(
        [sys.executable, "-c", snippet],
        cwd=str(tmp_path), capture_output=True, text=True, timeout=300, env=env,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "OK" in proc.stdout, proc.stdout + proc.stderr
