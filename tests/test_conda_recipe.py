"""Step 70 — conda-forge recipe stays consistent with the Python packaging.

A conda-forge ``meta.yaml`` is a jinja2-templated document. This test renders
it (so a template typo fails loudly), parses the resulting YAML, and asserts the
recipe agrees with ``pyproject.toml`` on the things a drifting recipe would get
wrong: version, the pinned z3 range, the Python floor, the console-script entry
points, the MIT license, and the canonical home URL.
"""

import os
import re
import tomllib

import jinja2
import yaml

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_RECIPE = os.path.join(_REPO, "conda-recipe", "meta.yaml")


def _render():
    raw = open(_RECIPE, encoding="utf-8").read()
    # Provide the conda-build jinja globals the template references.
    env = jinja2.Environment(undefined=jinja2.StrictUndefined)
    rendered = env.from_string(raw).render(PYTHON="$PYTHON", os=os)
    return yaml.safe_load(rendered)


def _pyproject():
    with open(os.path.join(_REPO, "pyproject.toml"), "rb") as fh:
        return tomllib.load(fh)


def test_recipe_renders_and_parses():
    doc = _render()
    assert doc["package"]["name"] == "tensorguard"
    assert isinstance(doc["package"]["version"], str)


def test_version_matches_pyproject():
    doc = _render()
    assert doc["package"]["version"] == _pyproject()["project"]["version"]


def test_run_dep_pins_z3_like_pyproject():
    doc = _render()
    run = doc["requirements"]["run"]
    z3 = [d for d in run if d.startswith("z3")]
    assert z3, run
    assert ">=4.12" in z3[0]
    assert "<5" in z3[0]


def test_python_floor_matches():
    doc = _render()
    run = " ".join(doc["requirements"]["run"])
    proj_floor = _pyproject()["project"]["requires-python"]  # ">=3.9"
    floor = re.search(r"3\.\d+", proj_floor).group(0)
    assert f"python >={floor}" in run


def test_entry_points_match_pyproject():
    doc = _render()
    eps = set(doc["build"]["entry_points"])
    scripts = _pyproject()["project"]["scripts"]
    for name, target in scripts.items():
        assert f"{name} = {target}" in eps


def test_license_and_home_canonical():
    doc = _render()
    about = doc["about"]
    assert about["license"] == "MIT"
    assert about["license_file"] == "LICENSE"
    assert about["home"] == "https://github.com/thehalleyyoung/tensorguard"


def test_noarch_python():
    doc = _render()
    assert doc["build"]["noarch"] == "python"
