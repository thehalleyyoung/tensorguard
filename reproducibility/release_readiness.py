"""Release-readiness gate for PyPI, conda, and Docker.

Step 286 turns release preparation into an executable checklist.  The gate is a
composition layer over existing evidence: benchmark dashboards, deployment
ratchets, artifact-package smoke tests, numeric-claim audits, security review
surfaces, and channel-specific packaging metadata.  A release is eligible only
when every required checklist item passes for every channel.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

OUT_JSON = REPO / "reproducibility" / "release_readiness.json"
OUT_MD = REPO / "reproducibility" / "release_readiness.md"


@dataclass(frozen=True)
class ChecklistItem:
    key: str
    area: str
    command: str
    evidence: Sequence[str]
    passed: bool
    detail: str
    required: bool = True


@dataclass(frozen=True)
class Channel:
    name: str
    publish_command: str
    items: Sequence[ChecklistItem]


def _read(rel: str) -> str:
    return (REPO / rel).read_text(encoding="utf-8")


def _json(rel: str) -> Mapping[str, object]:
    return json.loads(_read(rel))


def _version_from_pyproject() -> str:
    match = re.search(r'^version\s*=\s*"([^"]+)"', _read("pyproject.toml"), re.MULTILINE)
    if not match:
        raise ValueError("pyproject.toml has no project version")
    return match.group(1)


def _version_from_conda_recipe() -> str:
    match = re.search(r'{%\s*set\s+version\s*=\s*"([^"]+)"\s*%}', _read("conda-recipe/meta.yaml"))
    if not match:
        raise ValueError("conda-recipe/meta.yaml has no jinja version")
    return match.group(1)


def _status_line(ok: bool) -> str:
    return "pass" if ok else "FAIL"


def _dashboard_item() -> ChecklistItem:
    from evaluation import dashboard

    current = dashboard.compute_metrics()
    baseline = dashboard.load_baseline()
    result = dashboard.gate(current, baseline)
    md_current = dashboard.render_markdown(current, baseline)
    md_fresh = (REPO / "evaluation" / "dashboard.md").read_text(encoding="utf-8") == md_current
    ok = result.ok and md_fresh
    detail = (
        f"{len(current)} metrics, regressions={len(result.regressions)}, "
        f"orphans={len(result.orphans)}, unregistered={len(result.unregistered)}, "
        f"markdown_fresh={md_fresh}"
    )
    return ChecklistItem(
        key="benchmark-dashboard",
        area="benchmark dashboards",
        command="PYTHONPATH=. python evaluation/dashboard.py --check",
        evidence=["evaluation/dashboard.py", "evaluation/dashboard.md", "evaluation/dashboard_baseline.json"],
        passed=ok,
        detail=detail,
    )


def _deployment_item() -> ChecklistItem:
    from evaluation import deployment_dashboard

    manifest = deployment_dashboard.manifest()
    baseline = deployment_dashboard.load_baseline()
    result = deployment_dashboard.compare_to_baseline(manifest, baseline)
    md_current = deployment_dashboard.render_markdown(manifest, baseline)
    md_fresh = (REPO / "evaluation" / "deployment_dashboard.md").read_text(encoding="utf-8") == md_current
    ok = result.ok and md_fresh
    detail = (
        f"rows={len(deployment_dashboard.release_rows())}, regressions={len(result.regressions)}, "
        f"missing={len(result.missing)}, unregistered_supported={len(result.unregistered_supported)}, "
        f"markdown_fresh={md_fresh}"
    )
    return ChecklistItem(
        key="deployment-dashboard",
        area="benchmark dashboards",
        command="PYTHONPATH=. python evaluation/deployment_dashboard.py --check --gate",
        evidence=[
            "evaluation/deployment_dashboard.py",
            "evaluation/deployment_dashboard.md",
            "evaluation/deployment_dashboard_baseline.json",
        ],
        passed=ok,
        detail=detail,
    )


def _artifact_package_item(mode: str) -> ChecklistItem:
    from reproducibility import artifact_package

    data = artifact_package.build_manifest()
    selected = next(m for m in data["modes"] if m["mode"] == mode)
    checks = selected["checks"]
    failed = sorted(name for name, ok in checks.items() if not ok)
    return ChecklistItem(
        key=f"{mode}-artifact-package",
        area="artifact freshness",
        command="PYTHONPATH=. python reproducibility/artifact_package.py --check",
        evidence=["reproducibility/artifact_package.py", "reproducibility/artifact_package.json"],
        passed=bool(selected["passed"]),
        detail=f"checks={len(checks)}, failed={failed or 'none'}",
    )


def _numeric_audit_item() -> ChecklistItem:
    from reproducibility import audit_numeric_claims

    audit = audit_numeric_claims.run_audit()
    counts = audit["meta"]["counts"]
    return ChecklistItem(
        key="numeric-claim-audit",
        area="audit status",
        command="PYTHONPATH=. python reproducibility/audit_numeric_claims.py",
        evidence=["reproducibility/audit_numeric_claims.py", "reproducibility/numeric_claims_audit.json"],
        passed=bool(audit["passed"]),
        detail="statuses=" + ", ".join(f"{k}:{counts[k]}" for k in sorted(counts)),
    )


def _security_item() -> ChecklistItem:
    workflow = _read(".github/workflows/matrix.yml")
    ci = _read(".github/workflows/tensorguard-ci.yml")
    ok = all(
        [
            (REPO / "SECURITY.md").exists(),
            (REPO / "tests" / "test_security.py").exists(),
            "tests/test_security.py" in workflow,
            "python -m pytest tests/" in ci,
            "Analyzing a file never executes that file's code" in _read("SECURITY.md"),
        ]
    )
    return ChecklistItem(
        key="security-review",
        area="security review",
        command="python -m pytest tests/test_security.py -q",
        evidence=["SECURITY.md", "tests/test_security.py", ".github/workflows/matrix.yml"],
        passed=ok,
        detail="static-only threat model, regression test, and CI coverage are present",
    )


def _pypi_metadata_item() -> ChecklistItem:
    version = _version_from_pyproject()
    pyproject = _read("pyproject.toml")
    manifest = _read("MANIFEST.in")
    changelog = _read("CHANGELOG.md")
    citation = _read("CITATION.cff")
    ok = all(
        [
            'license = { text = "MIT" }' in pyproject,
            'readme = "README.md"' in pyproject,
            "exclude 100_STEPS.md" in manifest,
            "prune tests" in manifest,
            f'version: "{version}"' in citation,
            f"## [{version}]" in changelog,
        ]
    )
    return ChecklistItem(
        key="pypi-metadata",
        area="versioning and supply chain",
        command="python -m build --sdist --wheel",
        evidence=["pyproject.toml", "MANIFEST.in", "CHANGELOG.md", "CITATION.cff"],
        passed=ok,
        detail=f"version={version}, roadmap_excluded={'exclude 100_STEPS.md' in manifest}",
    )


def _conda_metadata_item() -> ChecklistItem:
    py_version = _version_from_pyproject()
    conda_version = _version_from_conda_recipe()
    recipe = _read("conda-recipe/meta.yaml")
    ok = all(
        [
            py_version == conda_version,
            "noarch: python" in recipe,
            "tensorguard --help" in recipe,
            "license_file: LICENSE" in recipe,
            "z3 >=4.12,<5" in recipe,
        ]
    )
    return ChecklistItem(
        key="conda-metadata",
        area="versioning and supply chain",
        command="conda build conda-recipe/",
        evidence=["conda-recipe/meta.yaml", "pyproject.toml", "LICENSE"],
        passed=ok,
        detail=f"pyproject_version={py_version}, conda_version={conda_version}",
    )


def _docker_metadata_item() -> ChecklistItem:
    dockerfile = _read("Dockerfile")
    dockerignore = _read(".dockerignore")
    ok = all(
        [
            "AS builder" in dockerfile and "AS runtime" in dockerfile,
            "python -m build --wheel" in dockerfile,
            "pip install --no-cache-dir /tmp/*.whl" in dockerfile,
            'ENTRYPOINT ["tensorguard"]' in dockerfile,
            "USER tg" in dockerfile,
            "100_STEPS.md" in dockerignore,
            ".git" in dockerignore,
        ]
    )
    return ChecklistItem(
        key="docker-metadata",
        area="versioning and supply chain",
        command="docker build -t tensorguard .",
        evidence=["Dockerfile", ".dockerignore"],
        passed=ok,
        detail="multi-stage wheel-only, non-root image with local roadmap excluded",
    )


def _common_items() -> List[ChecklistItem]:
    return [_dashboard_item(), _deployment_item(), _numeric_audit_item(), _security_item()]


def build_manifest() -> Dict[str, object]:
    common = _common_items()
    channels = [
        Channel(
            name="pypi",
            publish_command="python -m build --sdist --wheel && twine upload dist/tensorguard-*",
            items=[*common, _artifact_package_item("source"), _pypi_metadata_item()],
        ),
        Channel(
            name="conda",
            publish_command="conda build conda-recipe/ && anaconda upload <built-package>",
            items=[*common, _artifact_package_item("conda"), _conda_metadata_item()],
        ),
        Channel(
            name="docker",
            publish_command="docker build -t ghcr.io/thehalleyyoung/tensorguard:<version> . && docker push ghcr.io/thehalleyyoung/tensorguard:<version>",
            items=[*common, _artifact_package_item("docker"), _docker_metadata_item()],
        ),
    ]
    serialized = []
    for channel in channels:
        items = [
            {
                "key": item.key,
                "area": item.area,
                "command": item.command,
                "evidence": list(item.evidence),
                "passed": item.passed,
                "required": item.required,
                "detail": item.detail,
            }
            for item in channel.items
        ]
        serialized.append(
            {
                "channel": channel.name,
                "publish_command": channel.publish_command,
                "items": items,
                "required_items": sum(1 for item in channel.items if item.required),
                "passed_items": sum(1 for item in channel.items if item.passed),
                "release_ready": all(item.passed or not item.required for item in channel.items),
            }
        )
    return {
        "step": 286,
        "purpose": "release-readiness checklist gating PyPI, conda, and Docker releases",
        "version": _version_from_pyproject(),
        "channels": serialized,
        "all_channels_release_ready": all(bool(ch["release_ready"]) for ch in serialized),
        "check_command": "python reproducibility/release_readiness.py --check",
    }


def render_markdown(data: Mapping[str, object]) -> str:
    lines = [
        "# Release-readiness checklist (Step 286)",
        "",
        "A TensorGuard release is shippable only when PyPI, conda, and Docker all pass the same benchmark, audit, security, and artifact-freshness gates plus their channel-specific metadata checks.",
        "",
        f"- version: `{data['version']}`",
        f"- all channels release-ready: **{data['all_channels_release_ready']}**",
        f"- check command: `{data['check_command']}`",
        "",
    ]
    for channel in data["channels"]:  # type: ignore[index]
        lines.extend(
            [
                f"## {channel['channel']}",
                "",
                f"- publish command: `{channel['publish_command']}`",
                f"- passed items: **{channel['passed_items']}/{channel['required_items']}**",
                f"- release-ready: **{channel['release_ready']}**",
                "",
                "| Area | Gate | Status | Evidence | Detail |",
                "| --- | --- | --- | --- | --- |",
            ]
        )
        for item in channel["items"]:
            lines.append(
                "| {area} | `{gate}` | {status} | {evidence} | {detail} |".format(
                    area=item["area"],
                    gate=item["key"],
                    status=_status_line(bool(item["passed"])),
                    evidence=", ".join(f"`{path}`" for path in item["evidence"]),
                    detail=str(item["detail"]).replace("|", "\\|"),
                )
            )
        lines.append("")
    return "\n".join(lines)


def _dumps(data: object) -> str:
    return json.dumps(data, indent=2, sort_keys=True) + "\n"


def _problems(data: Mapping[str, object]) -> List[str]:
    problems = []
    for channel in data["channels"]:  # type: ignore[index]
        for item in channel["items"]:
            if item["required"] and not item["passed"]:
                problems.append(f"{channel['channel']}:{item['key']}: {item['detail']}")
    return problems


def run(check: bool = False) -> int:
    data = build_manifest()
    js = _dumps(data)
    md = render_markdown(data)
    if check:
        problems = _problems(data)
        if not OUT_JSON.exists() or OUT_JSON.read_text(encoding="utf-8") != js:
            problems.append(f"{OUT_JSON} is stale")
        if not OUT_MD.exists() or OUT_MD.read_text(encoding="utf-8") != md:
            problems.append(f"{OUT_MD} is stale")
        if problems:
            print("release_readiness: FAIL")
            for problem in problems:
                print(f"  - {problem}")
            return 1
        print("release_readiness: PASS")
        return 0
    OUT_JSON.write_text(js, encoding="utf-8")
    OUT_MD.write_text(md, encoding="utf-8")
    print(f"wrote {OUT_JSON}")
    print(f"wrote {OUT_MD}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    return run(check=args.check)


if __name__ == "__main__":
    sys.exit(main())
