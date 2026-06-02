"""Step 69 — packaging contract for a reproducible PyPI release.

Locks the distribution metadata that matters for a clean, reproducible PyPI
publish: canonical project URLs, a pinned ``z3-solver`` range, the MIT license
and LICENSE file, the console-script and pytest entry points, and a MANIFEST
that ships the docs while excluding the gitignored roadmap and the test tree.

The end-to-end build (``python -m build``), wheel byte-for-byte reproducibility
under a fixed ``SOURCE_DATE_EPOCH``, and ``twine check`` were verified manually
when this step was implemented; this test guards the inputs that make those
hold so a regression in the metadata is caught in CI.
"""

import os
import tomllib

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_CANONICAL = "https://github.com/thehalleyyoung/tensorguard"


def _pyproject():
    with open(os.path.join(_REPO, "pyproject.toml"), "rb") as fh:
        return tomllib.load(fh)


def test_canonical_urls():
    urls = _pyproject()["project"]["urls"]
    assert urls["Homepage"] == _CANONICAL
    assert urls["Repository"] == _CANONICAL
    # no stale placeholder org remains
    for v in urls.values():
        assert "tensorguard/tensorguard" not in v


def test_z3_pinned_range():
    deps = _pyproject()["project"]["dependencies"]
    z3 = [d for d in deps if d.replace(" ", "").startswith("z3-solver")]
    assert z3, deps
    spec = z3[0].replace(" ", "")
    assert ">=4.12" in spec
    assert "<5" in spec  # upper bound caps the major version for reproducibility


def test_license_is_mit_and_file_present():
    proj = _pyproject()["project"]
    lic = proj["license"]
    text = lic["text"] if isinstance(lic, dict) else lic
    assert text == "MIT"
    license_path = os.path.join(_REPO, "LICENSE")
    assert os.path.exists(license_path)
    assert "MIT License" in open(license_path, encoding="utf-8").read()


def test_entry_points_present():
    proj = _pyproject()["project"]
    scripts = proj["scripts"]
    assert scripts["tensorguard"] == "src.cli.main:main"
    assert scripts["tensorguard-precommit"] == "src.precommit:main"
    p11 = proj["entry-points"]["pytest11"]
    assert p11["tensorguard"] == "src.pytest_tensorguard"


def test_manifest_ships_docs_and_excludes_roadmap():
    manifest = open(os.path.join(_REPO, "MANIFEST.in"), encoding="utf-8").read()
    assert "include LICENSE" in manifest
    assert "include README.md" in manifest
    assert "include GETTING_STARTED.md" in manifest
    # the gitignored roadmap must never ship in a release
    assert "exclude 100_STEPS.md" in manifest
    # the test tree is pruned from the sdist
    assert "prune tests" in manifest


def test_license_files_configured():
    cfg = _pyproject().get("tool", {}).get("setuptools", {})
    assert "LICENSE" in cfg.get("license-files", [])
