"""Best-paper artifact package manifest (Step 267).

This module turns the release/artifact packaging story into a deterministic,
checkable contract across the three supported fresh-machine paths:

* Docker: build from the top-level multi-stage Dockerfile, run the wheel-only
  image against the real shape-bug example, and use the capsule image for the
  full paper-evidence replay.
* Conda: render the conda-forge recipe, verify it mirrors pyproject metadata,
  and use conda-build's isolated test environment.
* Source: clone, create a new venv, install from pyproject, then run the same
  real TensorGuard smoke test plus the one-command reproduction gate.

The manifest is static except for the source smoke result, which is a verdict
over a checked-in buggy PyTorch module and therefore deterministic.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Dict, Iterable, List

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

OUT_JSON = REPO / "reproducibility" / "artifact_package.json"
OUT_MD = REPO / "reproducibility" / "artifact_package.md"
DOC = REPO / "docs" / "artifact" / "PACKAGING.md"

SMOKE_EXAMPLE = "examples/shape_bug.py"
SMOKE_INPUT_SHAPE = {"x": (1, 3, 32, 32)}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read(rel: str) -> str:
    return (REPO / rel).read_text(encoding="utf-8")


def _hashes(paths: Iterable[str]) -> Dict[str, str]:
    return {p: _sha256(REPO / p) for p in sorted(paths)}


def _source_smoke() -> Dict[str, object]:
    from src.api import verify_module

    result = verify_module(SMOKE_EXAMPLE, input_shapes=SMOKE_INPUT_SHAPE)
    categories = sorted({str(b.category.value if hasattr(b.category, "value") else b.category)
                         for b in result.bugs})
    return {
        "example": SMOKE_EXAMPLE,
        "input_shapes": {"x": [1, 3, 32, 32]},
        "expected_status": "UNSAFE",
        "observed_status": result.status,
        "n_bugs": len(result.bugs),
        "bug_categories": categories,
        "passed": result.status == "UNSAFE" and len(result.bugs) >= 1,
    }


def _mode_docker() -> Dict[str, object]:
    dockerfile = _read("Dockerfile")
    dockerignore = _read(".dockerignore")
    capsule = _read("capsule/Dockerfile.reproduce")
    capsule_ignore = _read("capsule/Dockerfile.reproduce.dockerignore")
    checks = {
        "multi_stage_wheel_build": "AS builder" in dockerfile
        and "AS runtime" in dockerfile
        and "python -m build --wheel" in dockerfile
        and "COPY --from=builder /dist/*.whl" in dockerfile,
        "runtime_is_wheel_only": "pip install --no-cache-dir /tmp/*.whl" in dockerfile,
        "non_root_runtime": "USER tg" in dockerfile,
        "tool_entrypoint": 'ENTRYPOINT ["tensorguard"]' in dockerfile,
        "roadmap_excluded": "100_STEPS.md" in dockerignore
        and "100_STEPS.md" in capsule_ignore,
        "capsule_replays_evidence": "capsule/reproduce.sh" in capsule
        and "requirements.lock.txt" in capsule,
    }
    return {
        "mode": "docker",
        "fresh_machine_command": "docker build -t tensorguard . && docker run --rm -v \"$PWD:/work\" tensorguard verify /work/examples/shape_bug.py -s x=1,3,32,32",
        "full_paper_evidence_command": "docker build -f capsule/Dockerfile.reproduce -t tensorguard-capsule . && docker run --rm tensorguard-capsule",
        "freshness_barriers": [
            "builds from python:3.12-slim",
            "runtime stage installs only the built wheel",
            "runs as an unprivileged user",
            "image context excludes .git, virtualenvs, caches, and 100_STEPS.md",
        ],
        "files": _hashes([
            "Dockerfile",
            ".dockerignore",
            "capsule/Dockerfile.reproduce",
            "capsule/Dockerfile.reproduce.dockerignore",
            "capsule/reproduce.sh",
            "capsule/requirements.lock.txt",
        ]),
        "checks": checks,
        "passed": all(checks.values()),
    }


def _mode_conda() -> Dict[str, object]:
    recipe = _read("conda-recipe/meta.yaml")
    pyproject = _read("pyproject.toml")
    checks = {
        "noarch_python": "noarch: python" in recipe,
        "isolated_test_command": "tensorguard --help" in recipe,
        "entrypoints_declared": "tensorguard = src.cli.main:main" in recipe
        and "tensorguard-precommit = src.precommit:main" in recipe,
        "z3_range_matches_pyproject": "z3 >=4.12,<5" in recipe
        and '"z3-solver>=4.12,<5"' in pyproject,
        "python_floor_matches_pyproject": "python >=3.9" in recipe
        and 'requires-python = ">=3.9"' in pyproject,
        "mit_license": "license: MIT" in recipe and "license_file: LICENSE" in recipe,
    }
    return {
        "mode": "conda",
        "fresh_machine_command": "python -m build --sdist && conda build conda-recipe/ --no-test && conda create -n tg-artifact tensorguard && conda run -n tg-artifact tensorguard --help",
        "full_paper_evidence_command": "conda run -n tg-artifact python reproducibility/reproduce_all.py --check",
        "freshness_barriers": [
            "conda-build creates an isolated host/test prefix",
            "recipe is noarch: python",
            "runtime dependencies mirror pyproject",
            "recipe smoke test runs the installed console script",
        ],
        "files": _hashes(["conda-recipe/meta.yaml", "pyproject.toml", "LICENSE"]),
        "checks": checks,
        "passed": all(checks.values()),
    }


def _mode_source(smoke: Dict[str, object]) -> Dict[str, object]:
    manifest = _read("MANIFEST.in")
    checks = {
        "pyproject_build_backend": "build-backend = \"setuptools.build_meta\"" in _read("pyproject.toml"),
        "roadmap_not_shipped": "exclude 100_STEPS.md" in manifest,
        "tests_not_shipped_in_sdist": "prune tests" in manifest,
        "docs_required_for_artifact": all(
            f"include {name}" in manifest
            for name in ("README.md", "SOUNDNESS_CONTRACT.md", "VERIFIABLE_FRAGMENT.md", "GETTING_STARTED.md")
        ),
        "one_command_reproduction": "reproduce_all.py --check" in _read("capsule/reproduce.sh"),
        "real_smoke_passed": bool(smoke["passed"]),
    }
    return {
        "mode": "source",
        "fresh_machine_command": "git clone https://github.com/thehalleyyoung/tensorguard && cd tensorguard && python -m venv .venv && . .venv/bin/activate && pip install -e '.[dev]' && tensorguard verify examples/shape_bug.py -s x=1,3,32,32",
        "full_paper_evidence_command": "python reproducibility/reproduce_all.py --check",
        "freshness_barriers": [
            "starts from a fresh clone",
            "creates a new virtual environment",
            "installs through pyproject metadata",
            "runs the same real bug smoke test before evidence replay",
        ],
        "files": _hashes([
            "pyproject.toml",
            "MANIFEST.in",
            "README.md",
            "capsule/reproduce.sh",
            SMOKE_EXAMPLE,
        ]),
        "checks": checks,
        "smoke": smoke,
        "passed": all(checks.values()),
    }


def build_manifest() -> Dict[str, object]:
    smoke = _source_smoke()
    modes: List[Dict[str, object]] = [
        _mode_docker(),
        _mode_conda(),
        _mode_source(smoke),
    ]
    return {
        "step": 267,
        "purpose": "best-paper artifact package across Docker, conda, and source install modes",
        "modes": modes,
        "n_modes": len(modes),
        "all_modes_passed": all(bool(m["passed"]) for m in modes),
        "real_smoke": smoke,
        "check_command": "python reproducibility/artifact_package.py --check",
        "docs": "docs/artifact/PACKAGING.md",
    }


def render_markdown(data: Dict[str, object]) -> str:
    lines = [
        "# Best-paper artifact package (Step 267)",
        "",
        "TensorGuard's artifact is packaged three ways — Docker, conda, and source — "
        "and each path has an explicit fresh-machine command, hidden-state barrier, "
        "and smoke/evidence command.",
        "",
        f"- modes checked: **{data['n_modes']}**",
        f"- all modes passed: **{data['all_modes_passed']}**",
        f"- real smoke: `{data['real_smoke']['example']}` -> "
        f"**{data['real_smoke']['observed_status']}** "
        f"({data['real_smoke']['n_bugs']} bugs)",
        f"- check command: `{data['check_command']}`",
        "",
        "| mode | fresh-machine command | full evidence command | passed |",
        "| --- | --- | --- | --- |",
    ]
    for mode in data["modes"]:  # type: ignore[index]
        lines.append(
            f"| {mode['mode']} | `{mode['fresh_machine_command']}` | "
            f"`{mode['full_paper_evidence_command']}` | {mode['passed']} |"
        )
    lines += ["", "## Freshness barriers", ""]
    for mode in data["modes"]:  # type: ignore[index]
        lines.append(f"### {mode['mode']}")
        lines.append("")
        for barrier in mode["freshness_barriers"]:
            lines.append(f"- {barrier}")
        lines.append("")
    return "\n".join(lines)


def run(check: bool = False) -> int:
    data = build_manifest()
    js = json.dumps(data, indent=2, sort_keys=True) + "\n"
    md = render_markdown(data)
    if check:
        ok = True
        if not OUT_JSON.exists() or OUT_JSON.read_text() != js:
            print(f"MISMATCH: {OUT_JSON}")
            ok = False
        if not OUT_MD.exists() or OUT_MD.read_text() != md:
            print(f"MISMATCH: {OUT_MD}")
            ok = False
        if not DOC.exists() or DOC.read_text() != md:
            print(f"MISMATCH: {DOC}")
            ok = False
        if not data["all_modes_passed"]:
            print("MISMATCH: one or more package modes failed")
            ok = False
        if ok:
            print("artifact_package: byte-identical; all package modes pass")
        return 0 if ok else 1
    OUT_JSON.write_text(js)
    OUT_MD.write_text(md)
    DOC.write_text(md)
    print(f"wrote {OUT_JSON}")
    print(f"wrote {OUT_MD}")
    print(f"wrote {DOC}")
    return 0


if __name__ == "__main__":
    sys.exit(run(check="--check" in sys.argv))
