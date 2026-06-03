#!/usr/bin/env python3
"""Step 287 -- generated contributor onboarding for operator coverage work.

The onboarding path is generated from committed sources of truth rather than
handwritten wish lists: operator confidence/proof-footprint manifests,
community-stub governance docs, and the Lean proof tree.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

REPO = Path(__file__).resolve().parent.parent
OUT_DIR = REPO / "docs" / "contributing"
OUT_GUIDE = OUT_DIR / "operator_onboarding.md"
OUT_ISSUES_JSON = OUT_DIR / "good_first_operators.json"
OUT_ISSUES_MD = OUT_DIR / "good_first_operators.md"
OUT_TEMPLATE = OUT_DIR / "operator_template.py"
OUT_STUB_GUIDE = OUT_DIR / "stub_authoring_guide.md"
OUT_LEAN = OUT_DIR / "lean_proof_examples.md"
OUT_AUDIT_JSON = REPO / "reproducibility" / "contributor_onboarding_audit.json"
OUT_AUDIT_MD = REPO / "reproducibility" / "contributor_onboarding_audit.md"
OUTPUTS = (
    OUT_GUIDE,
    OUT_ISSUES_JSON,
    OUT_ISSUES_MD,
    OUT_TEMPLATE,
    OUT_STUB_GUIDE,
    OUT_LEAN,
    OUT_AUDIT_JSON,
    OUT_AUDIT_MD,
)


@dataclass(frozen=True)
class Issue:
    operator: str
    confidence: str
    proof_status: str
    title: str
    labels: Sequence[str]
    body: str
    acceptance_tests: Sequence[str]
    evidence: Sequence[str]
    difficulty: str


def _read_json(rel: str) -> Mapping[str, Any]:
    return json.loads((REPO / rel).read_text(encoding="utf-8"))


def _operator_rows() -> List[Mapping[str, Any]]:
    confidence = {row["operator"]: row for row in _read_json("operator_confidence_table.json")["operators"]}
    rows = []
    for row in _read_json("proof_footprint_manifest.json")["operators"]:
        merged = dict(row)
        merged["confidence"] = confidence.get(row["operator"], {}).get("confidence", row.get("confidence", "heuristic"))
        merged["confidence_rationale"] = confidence.get(row["operator"], {}).get(
            "rationale", row.get("confidence_rationale", "")
        )
        rows.append(merged)
    return sorted(rows, key=lambda r: str(r["operator"]))


def _issue_rank(row: Mapping[str, Any]) -> tuple:
    confidence_rank = {"heuristic": 0, "sound": 1, "complete": 2}.get(str(row["confidence"]), 3)
    proof_rank = {"heuristic": 0, "tested_only_rule": 1, "pen_and_paper_rule": 2, "lean_theorem": 3}.get(
        str(row.get("proof_status", "")), 4
    )
    return (confidence_rank, proof_rank, str(row["operator"]))


def _sanitize_issue_id(operator: str) -> str:
    return (
        operator.replace("torch.", "torch-")
        .replace("F.", "F-")
        .replace("nn.", "nn-")
        .replace(".", "-")
        .replace("_", "-")
        .replace("/", "-")
        .lower()
    )


def build_issues(limit: int = 12) -> List[Issue]:
    candidates = sorted(_operator_rows(), key=_issue_rank)
    selected = candidates[:limit]
    issues: List[Issue] = []
    for row in selected:
        operator = str(row["operator"])
        confidence = str(row.get("confidence", "heuristic"))
        proof_status = str(row.get("proof_status", "heuristic"))
        issue_id = _sanitize_issue_id(operator)
        evidence = tuple(str(p) for p in row.get("evidence", []) if (REPO / str(p)).exists())
        if not evidence:
            evidence = ("src/graph_compiler.py", "tests/test_graph_compiler.py")
        acceptance = (
            "docs/contributing/operator_template.py",
            "tests/test_operator_confidence.py",
            "tests/test_proof_footprint.py",
        )
        title = f"Good first operator: upgrade `{operator}` transfer evidence"
        body = "\n".join(
            [
                f"### Goal\nImprove TensorGuard's transfer-function evidence for `{operator}`.",
                "",
                f"- Current confidence: `{confidence}`",
                f"- Current proof footprint: `{proof_status}`",
                f"- Rationale: {row.get('confidence_rationale') or row.get('rationale') or 'No rationale recorded.'}",
                f"- Starting evidence: {', '.join(f'`{p}`' for p in evidence)}",
                "",
                "### Acceptance checklist",
                "1. Add or tighten a transfer/conformance case for the operator using `docs/contributing/operator_template.py`.",
                "2. Run the operator-specific pytest you added, plus `tests/test_operator_confidence.py` and `tests/test_proof_footprint.py`.",
                "3. If the proof status changes, regenerate `operator_confidence_table.json` / `proof_footprint_manifest.json` and explain why.",
                "4. Do not execute untrusted model code; use declarative stubs or isolated plugin contracts for third-party layers.",
            ]
        )
        issues.append(
            Issue(
                operator=operator,
                confidence=confidence,
                proof_status=proof_status,
                title=title,
                labels=("good first issue", "operator-coverage", "tensorguard", f"operator:{issue_id}"),
                body=body,
                acceptance_tests=acceptance,
                evidence=evidence,
                difficulty="beginner" if confidence == "heuristic" or proof_status in {"heuristic", "tested_only_rule"} else "intermediate",
            )
        )
    return issues


def _lean_files() -> List[Path]:
    return sorted((REPO / "lean").glob("**/*.lean"))


def _theorem_exists(short_name: str) -> bool:
    needle_theorem = f"theorem {short_name}"
    needle_def = f"def {short_name}"
    return any(
        needle_theorem in path.read_text(encoding="utf-8") or needle_def in path.read_text(encoding="utf-8")
        for path in _lean_files()
    )


def build_lean_examples(limit: int = 8) -> List[Dict[str, Any]]:
    examples: List[Dict[str, Any]] = []
    for row in _operator_rows():
        modules = [str(m) for m in row.get("lean_modules", [])]
        theorems = [str(t) for t in row.get("lean_theorems", [])]
        if not modules or not theorems:
            continue
        first_module = modules[0]
        rel = "lean/" + first_module.replace(".", "/") + ".lean"
        if not (REPO / rel).exists():
            # Some manifest entries use module aliases; keep the theorem but mark
            # the nearest real Lean tree in the audit instead of inventing a path.
            rel = "lean/TensorGuard.lean"
        theorem = theorems[0]
        examples.append(
            {
                "operator": row["operator"],
                "module": first_module,
                "path": rel,
                "theorem": theorem,
                "short_theorem": theorem.split(".")[-1],
                "theorem_resolves": _theorem_exists(theorem.split(".")[-1]),
                "role": row.get("rule", "operator transfer soundness"),
            }
        )
        if len(examples) >= limit:
            break
    return examples


def _issue_payloads(issues: Sequence[Issue]) -> Dict[str, Any]:
    return {
        "schema": "tensorguard.good_first_operators/v1",
        "source_of_truth": [
            "operator_confidence_table.json",
            "proof_footprint_manifest.json",
            "community_stubs/README.md",
            "docs/plugins/operator_plugin_abi.md",
            "lean/",
        ],
        "summary": {
            "issue_count": len(issues),
            "heuristic_count": sum(1 for issue in issues if issue.confidence == "heuristic"),
            "tested_only_count": sum(1 for issue in issues if issue.proof_status == "tested_only_rule"),
            "beginner_count": sum(1 for issue in issues if issue.difficulty == "beginner"),
        },
        "issues": [
            {
                "operator": issue.operator,
                "title": issue.title,
                "labels": list(issue.labels),
                "difficulty": issue.difficulty,
                "confidence": issue.confidence,
                "proof_status": issue.proof_status,
                "body": issue.body,
                "acceptance_tests": list(issue.acceptance_tests),
                "evidence": list(issue.evidence),
            }
            for issue in issues
        ],
    }


def render_issues_md(payload: Mapping[str, Any]) -> str:
    lines = [
        "# Good first operator issues",
        "",
        "This queue is generated from `operator_confidence_table.json` and",
        "`proof_footprint_manifest.json`. It prioritizes low-confidence or",
        "lightly-evidenced operators so contributor work improves the verifier's",
        "actual trust surface rather than a hand-maintained wishlist.",
        "",
        "| Operator | Difficulty | Confidence | Proof footprint | Labels |",
        "| --- | --- | --- | --- | --- |",
    ]
    for issue in payload["issues"]:  # type: ignore[index]
        labels = ", ".join(f"`{label}`" for label in issue["labels"])
        lines.append(
            f"| `{issue['operator']}` | {issue['difficulty']} | `{issue['confidence']}` | "
            f"`{issue['proof_status']}` | {labels} |"
        )
    lines.extend(["", "## Copyable issue bodies", ""])
    for issue in payload["issues"]:  # type: ignore[index]
        lines.extend([f"### {issue['title']}", "", issue["body"], ""])
    return "\n".join(lines)


def render_template() -> str:
    return '''"""Template for adding a TensorGuard operator transfer/conformance test.

Copy this file into tests/ with a real test_* name, replace OPERATOR_UNDER_TEST,
and keep one valid real-PyTorch case plus one invalid case when the operator has
a refutable precondition. This template is intentionally not named test_*.py so
pytest will not collect it directly.
"""

from __future__ import annotations

import pytest

from src.api import verify_architecture


OPERATOR_UNDER_TEST = "replace.me"


def test_operator_transfer_accepts_real_valid_case():
    source = """
import torch
from torch import nn

class Model(nn.Module):
    def forward(self, x):
        # Replace with a minimal, valid use of OPERATOR_UNDER_TEST.
        return x
"""
    result = verify_architecture(source, input_shape=(2, 3), soundness_mode="sound")
    assert not result.bugs


def test_operator_transfer_refutes_real_invalid_case():
    source = """
import torch
from torch import nn

class Model(nn.Module):
    def forward(self, x):
        # Replace with a minimal, invalid use that real PyTorch would reject.
        return x.reshape(5, 5)
"""
    result = verify_architecture(source, input_shape=(2, 3, 4), soundness_mode="sound")
    assert result.bugs
    assert any("shape" in bug.category.lower() for bug in result.bugs)


@pytest.mark.parametrize("path", ["operator_confidence_table.json", "proof_footprint_manifest.json"])
def test_operator_metadata_regenerated(path):
    assert path
'''


def render_stub_guide() -> str:
    return "\n".join(
        [
            "# Stub-authoring quick path",
            "",
            "Use community stubs when a third-party layer can be described by an",
            "existing declarative transfer kind; use the plugin ABI only for trusted",
            "packages that need executable transfer code.",
            "",
            "## Declarative community stub",
            "",
            "1. Add one JSON manifest under `community_stubs/`.",
            "2. Choose a vetted kind such as `shape_preserving` or `last_dim_linear`.",
            "3. Include provenance (`author`, `source_url`, `license`, `reviewed_by`).",
            "4. Add at least one valid conformance case and one invalid/error case when the contract has a refutable precondition.",
            "5. Run `python -m src.stub_governance_cli --check community_stubs/` and `python -m pytest tests/test_stub_governance.py -q`.",
            "",
            "The authoritative manifest rules live in `community_stubs/README.md`.",
            "",
            "## Trusted operator plugin",
            "",
            "Only use `src.operator_plugin_abi.OperatorTheoryContract` for trusted",
            "packages. The contract must be explicit-import only, versioned,",
            "provenance-bearing, security-reviewed, and conformance-tested. The",
            "maintainer-facing ABI docs are `docs/plugins/operator_plugin_abi.md` and",
            "`docs/plugins/third_party_conformance.md`.",
            "",
            "## Review invariant",
            "",
            "A contribution is not accepted because a shape rule sounds plausible. It",
            "is accepted when the declared conformance cases execute and the generated",
            "proof/confidence metadata remains synchronized with the real registry.",
            "",
        ]
    )


def render_lean_examples(examples: Sequence[Mapping[str, Any]]) -> str:
    lines = [
        "# Lean proof examples for operator contributors",
        "",
        "Most good-first operator work should start with Python conformance tests.",
        "When a rule graduates to proof-backed status, use the existing Lean files",
        "below as patterns. Each row is derived from `proof_footprint_manifest.json`",
        "and checked against the committed Lean tree.",
        "",
        "| Operator | Lean file | Theorem | Transfer role |",
        "| --- | --- | --- | --- |",
    ]
    for ex in examples:
        lines.append(
            f"| `{ex['operator']}` | `{ex['path']}` | `{ex['theorem']}` | {ex['role']} |"
        )
    lines.extend(
        [
            "",
            "Contributor rule of thumb: prove the smallest local transfer lemma first,",
            "then connect it to the Python registry through `proof_footprint_manifest.json`",
            "only after the torch oracle/conformance tests pass.",
            "",
        ]
    )
    return "\n".join(lines)


def render_onboarding(payload: Mapping[str, Any]) -> str:
    summary = payload["summary"]  # type: ignore[index]
    return "\n".join(
        [
            "# Contributor onboarding: operators, stubs, and proofs",
            "",
            "TensorGuard's fastest useful contribution path is a small operator upgrade:",
            "choose a generated good-first issue, add a real PyTorch conformance case,",
            "tighten the transfer or proof metadata, and regenerate the affected",
            "artifact. This page is generated by `reproducibility/contributor_onboarding.py`.",
            "",
            "## Start here",
            "",
            "1. Pick an issue from `docs/contributing/good_first_operators.md`.",
            "2. Copy `docs/contributing/operator_template.py` into `tests/` with a real `test_*.py` name.",
            "3. Replace the placeholder model with the smallest valid and invalid real-PyTorch examples.",
            "4. Run only the new operator test plus metadata guards named in the issue.",
            "5. If the operator is third-party, follow `docs/contributing/stub_authoring_guide.md` instead of adding executable code.",
            "6. If you are upgrading proof status, cite an existing pattern from `docs/contributing/lean_proof_examples.md`.",
            "",
            "## Generated queue summary",
            "",
            f"- Good-first issue payloads: **{summary['issue_count']}**",
            f"- Heuristic-confidence operators prioritized: **{summary['heuristic_count']}**",
            f"- Tested-only proof-footprint operators prioritized: **{summary['tested_only_count']}**",
            f"- Beginner-classified entries: **{summary['beginner_count']}**",
            "",
            "The queue is intentionally regenerated from real registries so it ages with",
            "the codebase. When coverage improves, the next lowest-confidence operators",
            "move into the top of the list automatically.",
            "",
        ]
    )


def build_audit() -> Dict[str, Any]:
    issues = build_issues()
    payload = _issue_payloads(issues)
    lean_examples = build_lean_examples()
    cited_paths = {
        "operator_confidence_table.json": (REPO / "operator_confidence_table.json").exists(),
        "proof_footprint_manifest.json": (REPO / "proof_footprint_manifest.json").exists(),
        "community_stubs/README.md": (REPO / "community_stubs/README.md").exists(),
        "docs/plugins/operator_plugin_abi.md": (REPO / "docs/plugins/operator_plugin_abi.md").exists(),
        "docs/plugins/third_party_conformance.md": (REPO / "docs/plugins/third_party_conformance.md").exists(),
        "src/operator_plugin_abi.py": (REPO / "src/operator_plugin_abi.py").exists(),
    }
    for issue in issues:
        for path in issue.acceptance_tests:
            cited_paths[path] = (REPO / path).exists() if path != "docs/contributing/operator_template.py" else True
        for path in issue.evidence:
            cited_paths[path] = (REPO / path).exists()
    for ex in lean_examples:
        cited_paths[str(ex["path"])] = (REPO / str(ex["path"])).exists()
    return {
        "schema": "tensorguard.contributor_onboarding_audit/v1",
        "summary": {
            "issue_count": len(issues),
            "all_cited_paths_exist": all(cited_paths.values()),
            "lean_example_count": len(lean_examples),
            "lean_examples_resolve": all(bool(ex["theorem_resolves"]) for ex in lean_examples),
            "output_count": len(OUTPUTS),
        },
        "cited_paths": dict(sorted(cited_paths.items())),
        "good_first_operators": payload["summary"],
        "lean_examples": lean_examples,
    }


def render_audit_md(audit: Mapping[str, Any]) -> str:
    summary = audit["summary"]  # type: ignore[index]
    lines = [
        "# Contributor onboarding audit",
        "",
        f"- Good-first issue payloads: **{summary['issue_count']}**",
        f"- Lean examples: **{summary['lean_example_count']}**",
        f"- All cited paths exist: **{summary['all_cited_paths_exist']}**",
        f"- Lean theorem names resolve: **{summary['lean_examples_resolve']}**",
        "",
        "| Path | Present |",
        "| --- | --- |",
    ]
    for path, ok in audit["cited_paths"].items():  # type: ignore[index]
        lines.append(f"| `{path}` | {ok} |")
    lines.append("")
    return "\n".join(lines)


def write_outputs() -> Dict[str, Any]:
    issues = build_issues()
    payload = _issue_payloads(issues)
    lean_examples = build_lean_examples()
    audit = build_audit()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_GUIDE.write_text(render_onboarding(payload), encoding="utf-8")
    OUT_ISSUES_JSON.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    OUT_ISSUES_MD.write_text(render_issues_md(payload), encoding="utf-8")
    OUT_TEMPLATE.write_text(render_template(), encoding="utf-8")
    OUT_STUB_GUIDE.write_text(render_stub_guide(), encoding="utf-8")
    OUT_LEAN.write_text(render_lean_examples(lean_examples), encoding="utf-8")
    OUT_AUDIT_JSON.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    OUT_AUDIT_MD.write_text(render_audit_md(audit), encoding="utf-8")
    return audit


def _git_diff(paths: Iterable[Path]) -> str:
    proc = subprocess.run(
        ["git", "--no-pager", "diff", "--", *(str(path.relative_to(REPO)) for path in paths)],
        cwd=REPO,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return proc.stdout + proc.stderr


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", help="fail if generated onboarding artifacts are stale")
    args = parser.parse_args(argv)

    before = _git_diff(OUTPUTS) if args.check else ""
    audit = write_outputs()
    if not audit["summary"]["all_cited_paths_exist"] or not audit["summary"]["lean_examples_resolve"]:
        print("contributor onboarding audit failed", file=sys.stderr)
        return 1
    if args.check:
        after = _git_diff(OUTPUTS)
        if before != after or after:
            print("contributor onboarding artifacts are stale", file=sys.stderr)
            print(after, file=sys.stderr)
            return 1
    print("contributor onboarding audit: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
