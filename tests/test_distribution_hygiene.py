"""Steps 76-77 — license clarity + third-party redistribution hygiene.

Asserts the shipped distribution contains only the MIT-licensed ``src`` package
plus docs/license, and that the large development-time third-party trees (the
vendored PyTea checkout and its ``node_modules``) never leak into an sdist. A
real sdist is built and inspected so this is proven, not just asserted.
"""

import glob
import os
import subprocess
import sys
import tarfile
import tomllib

import pytest

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _pyproject():
    with open(os.path.join(_REPO, "pyproject.toml"), "rb") as fh:
        return tomllib.load(fh)


def test_packaging_restricted_to_src():
    find = _pyproject()["tool"]["setuptools"]["packages"]["find"]
    assert find["include"] == ["src*"]


def test_license_is_mit_bsd_compatible():
    lic = _pyproject()["project"]["license"]
    text = lic["text"] if isinstance(lic, dict) else lic
    assert text == "MIT"  # permissive, compatible with PyTorch BSD-3-Clause


def test_manifest_prunes_third_party_trees():
    manifest = open(os.path.join(_REPO, "MANIFEST.in"), encoding="utf-8").read()
    assert "prune experiments_v5" in manifest
    assert "global-exclude node_modules" in manifest


def test_third_party_notices_documented():
    path = os.path.join(_REPO, "THIRD_PARTY_NOTICES.md")
    assert os.path.exists(path)
    text = open(path, encoding="utf-8").read()
    assert "MIT" in text
    assert "BSD-3" in text
    assert "_pytea_src" in text


def test_src_contains_no_vendored_node_modules():
    # the shipped package tree must not embed a node_modules directory
    assert not glob.glob(os.path.join(_REPO, "src", "**", "node_modules"), recursive=True)


@pytest.mark.timeout(300)
def test_built_sdist_excludes_third_party():
    out = os.path.join(_REPO, ".tmp_dist_hygiene")
    env = dict(os.environ, SOURCE_DATE_EPOCH="1700000000")
    try:
        proc = subprocess.run(
            [sys.executable, "-m", "build", "--sdist", "--no-isolation",
             "--outdir", out],
            cwd=_REPO, capture_output=True, text=True, env=env, timeout=300,
        )
        assert proc.returncode == 0, proc.stderr[-2000:]
        tarballs = glob.glob(os.path.join(out, "*.tar.gz"))
        assert tarballs, "no sdist produced"
        with tarfile.open(tarballs[0]) as tf:
            names = tf.getnames()
        offenders = [
            n for n in names
            if "_pytea_src" in n or "node_modules" in n or "experiments_v5" in n
        ]
        assert offenders == [], offenders[:5]
        # sanity: the src package and LICENSE ARE present
        assert any(n.endswith("/LICENSE") for n in names)
        assert any("/src/" in n for n in names)
    finally:
        import shutil

        shutil.rmtree(out, ignore_errors=True)
        for junk in ("build", "tensorguard.egg-info", "src/tensorguard.egg-info"):
            shutil.rmtree(os.path.join(_REPO, junk), ignore_errors=True)
