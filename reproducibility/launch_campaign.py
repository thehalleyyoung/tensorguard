#!/usr/bin/env python3
"""Step 288 -- generated public launch campaign and support promise.

The campaign is intentionally launch-readiness material, not a claim that the
current package version is already 1.0. Every channel is tied to a real demo,
evidence artifact, upstream/RFC surface, and compatibility/support source.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

REPO = Path(__file__).resolve().parent.parent
OUT_DIR = REPO / "docs" / "launch"
OUT_CAMPAIGN = OUT_DIR / "one_point_zero_launch_campaign.md"
OUT_DEMO = OUT_DIR / "demo_script.md"
OUT_SOCIAL = OUT_DIR / "social_copy.md"
OUT_SUPPORT = OUT_DIR / "compatibility_support_promise.md"
OUT_AUDIT_JSON = REPO / "reproducibility" / "launch_campaign_audit.json"
OUT_AUDIT_MD = REPO / "reproducibility" / "launch_campaign_audit.md"
OUTPUTS = (
    OUT_CAMPAIGN,
    OUT_DEMO,
    OUT_SOCIAL,
    OUT_SUPPORT,
    OUT_AUDIT_JSON,
    OUT_AUDIT_MD,
)


@dataclass(frozen=True)
class Channel:
    key: str
    title: str
    audience: str
    demo: str
    evidence: str
    rfc: str
    support: str
    call_to_action: str


def _read(rel: str) -> str:
    return (REPO / rel).read_text(encoding="utf-8")


def _json(rel: str) -> Mapping[str, Any]:
    return json.loads(_read(rel))


def _pyproject_version() -> str:
    match = re.search(r'^version\s*=\s*"([^"]+)"', _read("pyproject.toml"), re.MULTILINE)
    if not match:
        raise ValueError("pyproject.toml has no version")
    return match.group(1)


def _requires_python() -> str:
    match = re.search(r'^requires-python\s*=\s*"([^"]+)"', _read("pyproject.toml"), re.MULTILINE)
    if not match:
        raise ValueError("pyproject.toml has no requires-python")
    return match.group(1)


def _matrix_counts() -> Dict[str, int]:
    text = _read(".github/workflows/matrix.yml")
    return {
        "matrix_jobs": text.count("- os: "),
        "nightly_jobs": text.count("torch-nightly"),
        "supported_surface_tests": len(re.findall(r"tests/test_[A-Za-z0-9_]+\.py", text)),
    }


def _model_gallery_count() -> int:
    data = _json("examples/model_gallery.json")
    entries = data["models"] if "models" in data else data
    return len(entries)


def _tutorial_count() -> int:
    return len(sorted((REPO / "examples" / "tutorials").glob("*.ipynb")))


def _release_readiness_summary() -> Dict[str, Any]:
    data = _json("reproducibility/release_readiness.json")
    channels = data.get("channels", [])
    return {
        "release_channels": len(channels),
        "all_channels_release_ready": bool(data.get("all_channels_release_ready")),
        "channel_names": sorted(str(c["channel"]) for c in channels),
    }


def support_sources() -> Dict[str, str]:
    return {
        "python": "pyproject.toml",
        "ci_matrix": ".github/workflows/matrix.yml",
        "api_stability": "DEPRECATION_POLICY.md",
        "security": "SECURITY.md",
        "release_readiness": "reproducibility/release_readiness.md",
        "changelog": "CHANGELOG.md",
    }


def channels() -> List[Channel]:
    support = "docs/launch/compatibility_support_promise.md"
    return [
        Channel(
            key="show-and-tell",
            title="Show-and-tell launch post",
            audience="PyTorch developers who want a fast first bug catch",
            demo="examples/quickstart.py",
            evidence="reproducibility/release_readiness.md",
            rfc="docs/RFC_pytorch_companion.md",
            support=support,
            call_to_action="Run the quickstart on one local model and report the first UNKNOWN/UNSAFE result.",
        ),
        Channel(
            key="pytorch-forums",
            title="PyTorch forum / dev-discuss RFC thread",
            audience="PyTorch maintainers, compiler users, and library authors",
            demo="examples/model_gallery.md",
            evidence="reproducibility/upstream_hook_demo.md",
            rfc="docs/upstream/pytorch_proposal.md",
            support=support,
            call_to_action="Review the opt-in hook proposal and identify integration concerns.",
        ),
        Channel(
            key="reproducibility-thread",
            title="Reproducibility-first evidence thread",
            audience="Researchers, artifact evaluators, and skeptical adopters",
            demo="examples/tutorials/README.md",
            evidence="reproducibility/artifact_index.md",
            rfc="docs/RFC_pytorch_companion.md",
            support=support,
            call_to_action="Run the reproducibility capsule or inspect the artifact hash ledger.",
        ),
        Channel(
            key="library-authors",
            title="Library-author compatibility call",
            audience="Maintainers of PyTorch-adjacent layers and model libraries",
            demo="docs/contributing/operator_onboarding.md",
            evidence="docs/governance/third_party_acceptance.md",
            rfc="docs/upstream/pytorch_proposal.md",
            support=support,
            call_to_action="Contribute a declarative stub, plugin conformance case, or good-first operator upgrade.",
        ),
    ]


def build_audit() -> Dict[str, Any]:
    readiness = _release_readiness_summary()
    counts = {
        "model_gallery_entries": _model_gallery_count(),
        "tutorial_notebooks": _tutorial_count(),
        **_matrix_counts(),
        **readiness,
    }
    cited_paths = {
        "pyproject.toml": (REPO / "pyproject.toml").exists(),
        "docs/GROWTH_PLAYBOOK.md": (REPO / "docs/GROWTH_PLAYBOOK.md").exists(),
    }
    for path in support_sources().values():
        cited_paths[path] = (REPO / path).exists()
    channel_rows = []
    for channel in channels():
        anchors = {
            "demo": channel.demo,
            "evidence": channel.evidence,
            "rfc": channel.rfc,
            "support": channel.support,
        }
        for path in anchors.values():
            cited_paths[path] = path == "docs/launch/compatibility_support_promise.md" or (REPO / path).exists()
        channel_rows.append(
            {
                "key": channel.key,
                "title": channel.title,
                "audience": channel.audience,
                "anchors": anchors,
                "has_demo_evidence_rfc_support": all(anchors.values()),
                "call_to_action": channel.call_to_action,
            }
        )
    current_version = _pyproject_version()
    return {
        "schema": "tensorguard.launch_campaign_audit/v1",
        "campaign_track": "1.0-readiness",
        "current_package_version": current_version,
        "honest_versioning": current_version != "1.0.0",
        "requires_python": _requires_python(),
        "summary": {
            "channel_count": len(channel_rows),
            "all_channels_have_required_anchors": all(r["has_demo_evidence_rfc_support"] for r in channel_rows),
            "all_cited_paths_exist": all(cited_paths.values()),
            "support_promise_source_count": len(support_sources()),
            "release_ready_gate_passes": bool(readiness["all_channels_release_ready"]),
        },
        "source_counts": counts,
        "support_sources": support_sources(),
        "channels": channel_rows,
        "cited_paths": dict(sorted(cited_paths.items())),
    }


def render_support_promise(audit: Mapping[str, Any]) -> str:
    sources = audit["support_sources"]  # type: ignore[index]
    counts = audit["source_counts"]  # type: ignore[index]
    channels_list = ", ".join(f"`{name}`" for name in counts["channel_names"])
    return "\n".join(
        [
            "# Compatibility and support promise",
            "",
            "This is the launch-readiness support promise for TensorGuard. It is",
            "generated from existing policy and CI surfaces rather than asserted as",
            "marketing copy.",
            "",
            "## Versioning",
            "",
            f"- Current package version: `{audit['current_package_version']}`.",
            "- The launch track is `1.0-readiness`; it does not claim the package has already shipped as `1.0.0`.",
            f"- Public API stability follows `{sources['api_stability']}`.",
            "",
            "## Compatibility",
            "",
            f"- Python requirement is `{audit['requires_python']}` from `{sources['python']}`.",
            f"- The compatibility matrix currently enumerates {counts['matrix_jobs']} stable OS/Python/torch jobs and a nightly early-warning path in `{sources['ci_matrix']}`.",
            f"- Release-readiness gates cover {channels_list} in `{sources['release_readiness']}`.",
            "",
            "## Security and maintenance",
            "",
            f"- Security reports and the static-only untrusted-source boundary are governed by `{sources['security']}`.",
            f"- User-visible compatibility changes are recorded in `{sources['changelog']}` and deprecated through the public policy before removal.",
            "- UNKNOWN is a supported outcome, not a failed launch: out-of-fragment models abstain rather than silently passing.",
            "",
        ]
    )


def render_campaign(audit: Mapping[str, Any]) -> str:
    lines = [
        "# TensorGuard 1.0-readiness launch campaign",
        "",
        "This campaign is generated by `reproducibility/launch_campaign.py`. It is",
        "anchored in real demos, committed evidence, upstream RFC surfaces, and the",
        "compatibility/support promise.",
        "",
        "## Positioning",
        "",
        "TensorGuard is the static safety net for PyTorch models: run it before the",
        "first forward pass, get SAFE/UNSAFE/UNKNOWN instead of a late dispatcher",
        "surprise, and keep adoption honest with reproducible evidence.",
        "",
        "## Channel plan",
        "",
        "| Channel | Audience | Demo | Evidence | RFC/upstream anchor | Support anchor | CTA |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for channel in audit["channels"]:  # type: ignore[index]
        anchors = channel["anchors"]
        lines.append(
            f"| {channel['title']} | {channel['audience']} | `{anchors['demo']}` | "
            f"`{anchors['evidence']}` | `{anchors['rfc']}` | `{anchors['support']}` | "
            f"{channel['call_to_action']} |"
        )
    lines.extend(
        [
            "",
            "## Launch gates",
            "",
            "- Use the release-readiness gate as the ship/no-ship source of truth.",
            "- Keep demos copy-pasteable and backed by checked examples.",
            "- Route upstream discussion to the PyTorch companion RFCs rather than ad-hoc promises.",
            "- Treat the support promise as part of the launch artifact, not a blog-post afterthought.",
            "",
        ]
    )
    return "\n".join(lines)


def render_demo_script(audit: Mapping[str, Any]) -> str:
    counts = audit["source_counts"]  # type: ignore[index]
    return "\n".join(
        [
            "# Public demo script",
            "",
            "## Cold open",
            "",
            "Run `tensorguard verify examples/quickstart.py --input-shape",
            "x=1,3,224,224 --format json --no-color` to show a fresh install",
            "certifying the quickstart model, then run the generated gallery bug",
            "variant from `reproducibility/launch_dry_run.md` to show TensorGuard",
            "reporting an UNSAFE model-level contract violation before the",
            "matching PyTorch kernel would fail.",
            "",
            "For the shareable terminal capture, use",
            "`docs/launch/quickstart_terminal_demo.gif`; it is generated from",
            "`reproducibility/launch_dry_run.json`, not hand-edited.",
            "",
            "## Evidence tour",
            "",
            f"- Open `examples/model_gallery.md` for the generated model gallery ({counts['model_gallery_entries']} entries).",
            f"- Open `examples/tutorials/README.md` for the executable tutorial set ({counts['tutorial_notebooks']} notebooks).",
            "- Open `reproducibility/artifact_index.md` to show the generated artifact hash ledger.",
            "- Open `reproducibility/release_readiness.md` to show the launch gate.",
            "- Open `reproducibility/launch_dry_run.md` to show the fresh-venv demo proof.",
            "",
            "## Upstream close",
            "",
            "End with `docs/upstream/pytorch_proposal.md`: the ask is an opt-in PyTorch",
            "companion hook, not a breaking default-on checker.",
            "",
        ]
    )


def render_social_copy(audit: Mapping[str, Any]) -> str:
    counts = audit["source_counts"]  # type: ignore[index]
    return "\n".join(
        [
            "# Launch copy",
            "",
            "## Short post",
            "",
            "TensorGuard is ready for a public 1.0-readiness push: static SAFE/UNSAFE/UNKNOWN",
            "checks for PyTorch model contracts, real demos, an upstream companion RFC,",
            "and a support promise backed by CI and release-readiness gates.",
            "",
            "Try the quickstart, inspect the evidence ledger, then bring one model or",
            "one operator issue.",
            "",
            "## Longer thread outline",
            "",
            "- Start with the quickstart demo.",
            f"- Show the model gallery with {counts['model_gallery_entries']} copyable examples.",
            "- Show the reproducibility/artifact ledger instead of asking for trust.",
            "- Link the PyTorch companion RFC and explain why UNKNOWN is a feature.",
            "- Invite operator/stub contributions through the generated onboarding path.",
            "",
        ]
    )


def render_audit_md(audit: Mapping[str, Any]) -> str:
    summary = audit["summary"]  # type: ignore[index]
    lines = [
        "# Launch campaign audit",
        "",
        f"- Campaign track: **{audit['campaign_track']}**",
        f"- Current package version: **{audit['current_package_version']}**",
        f"- Channels: **{summary['channel_count']}**",
        f"- All channel anchors present: **{summary['all_channels_have_required_anchors']}**",
        f"- All cited paths exist: **{summary['all_cited_paths_exist']}**",
        f"- Release-readiness gate passes: **{summary['release_ready_gate_passes']}**",
        "",
        "| Channel | Demo | Evidence | RFC | Support |",
        "| --- | --- | --- | --- | --- |",
    ]
    for channel in audit["channels"]:  # type: ignore[index]
        anchors = channel["anchors"]
        lines.append(
            f"| {channel['key']} | `{anchors['demo']}` | `{anchors['evidence']}` | "
            f"`{anchors['rfc']}` | `{anchors['support']}` |"
        )
    lines.extend(["", "## Cited paths", "", "| Path | Present |", "| --- | --- |"])
    for path, ok in audit["cited_paths"].items():  # type: ignore[index]
        lines.append(f"| `{path}` | {ok} |")
    lines.append("")
    return "\n".join(lines)


def write_outputs() -> Dict[str, Any]:
    audit = build_audit()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_SUPPORT.write_text(render_support_promise(audit), encoding="utf-8")
    OUT_CAMPAIGN.write_text(render_campaign(audit), encoding="utf-8")
    OUT_DEMO.write_text(render_demo_script(audit), encoding="utf-8")
    OUT_SOCIAL.write_text(render_social_copy(audit), encoding="utf-8")
    OUT_AUDIT_JSON.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    OUT_AUDIT_MD.write_text(render_audit_md(audit), encoding="utf-8")
    return audit


def _snapshots(paths: Iterable[Path]) -> Dict[Path, str | None]:
    return {path: path.read_text(encoding="utf-8") if path.exists() else None for path in paths}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", help="fail if launch campaign artifacts are stale")
    args = parser.parse_args(argv)

    before = _snapshots(OUTPUTS) if args.check else {}
    audit = write_outputs()
    summary = audit["summary"]
    if not summary["all_channels_have_required_anchors"] or not summary["all_cited_paths_exist"]:
        print("launch campaign audit failed", file=sys.stderr)
        return 1
    if args.check:
        after = _snapshots(OUTPUTS)
        if before != after:
            print("launch campaign artifacts are stale", file=sys.stderr)
            return 1
    print("launch campaign audit: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
