#!/usr/bin/env python3
"""Generate and audit TensorGuard's third-party acceptance governance.

Step 284 is intentionally a governance/documentation step, but the policy should
not be aspirational prose that drifts away from the repository. This script is
the executable link between the maintainer checklists and the real enforcement
surfaces: existing validators, tests, workflows, and policy docs.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

ROOT = Path(__file__).resolve().parent.parent
OUT_DOC = ROOT / "docs" / "governance" / "third_party_acceptance.md"
OUT_JSON = ROOT / "reproducibility" / "governance_acceptance_audit.json"
OUT_MD = ROOT / "reproducibility" / "governance_acceptance_audit.md"


@dataclass(frozen=True)
class Gate:
    name: str
    kind: str  # "automated" or "maintainer-judgment"
    checklist: str
    evidence: Tuple[str, ...]
    commands: Tuple[Tuple[str, ...], ...] = ()
    symbols: Tuple[str, ...] = ()


@dataclass(frozen=True)
class SubmissionClass:
    key: str
    title: str
    scope: str
    required_files: Tuple[str, ...]
    gates: Tuple[Gate, ...]


def policy() -> Tuple[SubmissionClass, ...]:
    return (
        SubmissionClass(
            key="verifier_backends",
            title="Verifier / SMT backend submissions",
            scope=(
                "Alternative solver backends or verifier-backend experiments must "
                "implement the shared SMT interface and prove concordance on the "
                "decidable VC fragments TensorGuard emits."
            ),
            required_files=(
                "src/smt/solver.py",
                "src/smt/cvc5_backend.py",
                "docs/decidability/smt_backends.md",
                "reproducibility/smt_backend_comparison.py",
                "tests/test_smt_backend_comparison.py",
            ),
            gates=(
                Gate(
                    "Shared backend interface",
                    "automated",
                    "Implement `SmtSolver`/`check_sat` semantics without changing the predicate IR.",
                    ("src/smt/solver.py",),
                    symbols=("src.smt.solver:SmtSolver",),
                ),
                Gate(
                    "Cross-backend concordance",
                    "automated",
                    "Run the Z3/cvc5 concordance harness and keep every VC verdict identical.",
                    ("reproducibility/smt_backend_comparison.py", "tests/test_smt_backend_comparison.py"),
                    commands=(
                        ("python", "reproducibility/smt_backend_comparison.py", "--check"),
                        ("python", "-m", "pytest", "tests/test_smt_backend_comparison.py", "-q"),
                    ),
                ),
                Gate(
                    "Security and governance review",
                    "maintainer-judgment",
                    "Confirm no new trust boundary, network access, or model-code execution path is introduced.",
                    ("SECURITY.md", "GOVERNANCE.md", "MAINTAINERS.md"),
                ),
            ),
        ),
        SubmissionClass(
            key="stubs_and_plugins",
            title="Community stubs and operator plugins",
            scope=(
                "Declarative community stubs are accepted only through the safe "
                "manifest path; executable operator plugins require explicit trust, "
                "ABI validation, security attestations, and conformance scenarios."
            ),
            required_files=(
                "community_stubs/README.md",
                "docs/plugins/operator_plugin_abi.md",
                "docs/plugins/third_party_conformance.md",
                "src/stub_governance.py",
                "src/operator_plugin_abi.py",
                "src/third_party_conformance.py",
                ".github/workflows/stub-registry.yml",
            ),
            gates=(
                Gate(
                    "Declarative stub manifest validation",
                    "automated",
                    "Reject code-bearing fields, require provenance, and run every manifest conformance case.",
                    ("src/stub_governance.py", "community_stubs/README.md", ".github/workflows/stub-registry.yml"),
                    commands=(
                        ("python", "-m", "src.stub_governance_cli", "--check", "community_stubs/"),
                        ("python", "-m", "pytest", "tests/test_stub_governance.py", "-q"),
                    ),
                    symbols=("src.stub_governance:validate_manifest",),
                ),
                Gate(
                    "Executable plugin ABI validation",
                    "automated",
                    "Validate ABI major version, provenance, security review attestations, and transfer conformance cases.",
                    ("src/operator_plugin_abi.py", "docs/plugins/operator_plugin_abi.md", "tests/test_operator_plugin_abi.py"),
                    commands=(("python", "-m", "pytest", "tests/test_operator_plugin_abi.py", "-q"),),
                    symbols=("src.operator_plugin_abi:validate_operator_theory",),
                ),
                Gate(
                    "Real verifier conformance",
                    "automated",
                    "Certify stubs/plugins against real `verify_architecture` verdicts in requested soundness modes.",
                    ("src/third_party_conformance.py", "docs/plugins/third_party_conformance.md", "tests/test_third_party_conformance.py"),
                    commands=(("python", "-m", "pytest", "tests/test_third_party_conformance.py", "-q"),),
                    symbols=("src.third_party_conformance:certify_stub_manifests", "src.third_party_conformance:certify_plugin_contracts"),
                ),
            ),
        ),
        SubmissionClass(
            key="corpora",
            title="Corpus and benchmark-case submissions",
            scope=(
                "New cases must be minimal, provenance-bearing, redistributable, "
                "runtime-validated against real PyTorch, and frozen by content hash."
            ),
            required_files=(
                "real_benchmarks/corpus_def.py",
                "real_benchmarks/build_manifest.py",
                "real_benchmarks/load.py",
                "real_benchmarks/manifest.json",
                "corpus_extended/provenance.py",
                "reproducibility/corpus_provenance_audit.py",
            ),
            gates=(
                Gate(
                    "Runtime ground truth and hash freeze",
                    "automated",
                    "Rebuild and load the corpus so clean cases execute, buggy cases fail as labeled, and hashes match.",
                    ("real_benchmarks/build_manifest.py", "real_benchmarks/load.py", "tests/test_real_benchmarks.py"),
                    commands=(
                        ("python", "-m", "real_benchmarks.build_manifest"),
                        ("python", "-m", "real_benchmarks.load"),
                        ("python", "-m", "pytest", "tests/test_real_benchmarks.py", "-q"),
                    ),
                ),
                Gate(
                    "Redistribution provenance audit",
                    "automated",
                    "Require source/provenance/license metadata and reject copied third-party code in generated corpora.",
                    ("corpus_extended/provenance.py", "reproducibility/corpus_provenance_audit.py", "tests/test_corpus_provenance.py"),
                    commands=(
                        ("python", "reproducibility/corpus_provenance_audit.py", "--check"),
                        ("python", "-m", "pytest", "tests/test_corpus_provenance.py", "-q"),
                    ),
                ),
                Gate(
                    "Anti-overfitting review",
                    "maintainer-judgment",
                    "Check whether the case belongs in dev, blind, or natural-distribution splits before tuning on it.",
                    ("corpus_extended/PRE_REGISTRATION.md", "reproducibility/blind_split_eval.py", "GOVERNANCE.md"),
                ),
            ),
        ),
        SubmissionClass(
            key="benchmark_submissions",
            title="Leaderboard and benchmark-result submissions",
            scope=(
                "External tool results are accepted as signed raw per-case verdicts; "
                "TensorGuard recomputes all metrics and rejects self-reported scores."
            ),
            required_files=(
                "docs/leaderboard/CONTRIBUTING.md",
                ".github/ISSUE_TEMPLATE/leaderboard_submission.md",
                ".github/workflows/leaderboard.yml",
                "reproducibility/validate_entry.py",
                "reproducibility/leaderboard.py",
                "benchmarks/leaderboard_entries/allowed_signers",
            ),
            gates=(
                Gate(
                    "Signed raw verdicts only",
                    "automated",
                    "Validate case ids, uppercase verdict tokens, SSH signatures, and absence of self-reported metrics.",
                    ("reproducibility/validate_entry.py", "benchmarks/leaderboard_entries/allowed_signers", ".github/workflows/leaderboard.yml"),
                    commands=(("python", "reproducibility/validate_entry.py"),),
                    symbols=("reproducibility.validate_entry:validate_entry",),
                ),
                Gate(
                    "Recomputed deterministic leaderboard",
                    "automated",
                    "Regenerate and byte-check the leaderboard from raw verdicts.",
                    ("reproducibility/leaderboard.py", "docs/leaderboard/CONTRIBUTING.md"),
                    commands=(
                        ("python", "reproducibility/leaderboard.py"),
                        ("python", "reproducibility/leaderboard.py", "--check"),
                    ),
                ),
                Gate(
                    "Anti-overfitting disclosure",
                    "maintainer-judgment",
                    "Review benchmark-specific tuning, abstention policy, and monthly refresh-window eligibility.",
                    (".github/ISSUE_TEMPLATE/leaderboard_submission.md", "docs/leaderboard/CONTRIBUTING.md", "MAINTAINERS.md"),
                ),
            ),
        ),
    )


def _path_exists(rel: str) -> bool:
    return (ROOT / rel).exists()


def _command_target_exists(cmd: Sequence[str]) -> bool:
    if not cmd:
        return False
    parts = list(cmd)
    for token in parts[1:]:
        if token.endswith(".py") or token.endswith(".md") or token.endswith(".json"):
            return _path_exists(token)
        if token.startswith("tests/") or token.startswith("community_stubs"):
            return _path_exists(token.rstrip("/"))
        if token.startswith("src.") or token.startswith("real_benchmarks."):
            rel = token.replace(".", "/") + ".py"
            return _path_exists(rel)
    return True


def _symbol_exists(spec: str) -> bool:
    module_name, _, attr = spec.partition(":")
    if not module_name or not attr:
        return False
    try:
        module = importlib.import_module(module_name)
    except Exception:
        return False
    obj: Any = module
    for part in attr.split("."):
        if not hasattr(obj, part):
            return False
        obj = getattr(obj, part)
    return True


def build_audit() -> Dict[str, Any]:
    classes = []
    missing_paths: List[str] = []
    missing_commands: List[str] = []
    missing_symbols: List[str] = []
    for cls in policy():
        gates = []
        for gate in cls.gates:
            evidence_ok = {p: _path_exists(p) for p in gate.evidence}
            command_ok = {" ".join(cmd): _command_target_exists(cmd) for cmd in gate.commands}
            symbol_ok = {s: _symbol_exists(s) for s in gate.symbols}
            missing_paths.extend(p for p, ok in evidence_ok.items() if not ok)
            missing_commands.extend(c for c, ok in command_ok.items() if not ok)
            missing_symbols.extend(s for s, ok in symbol_ok.items() if not ok)
            gates.append({
                "name": gate.name,
                "kind": gate.kind,
                "checklist": gate.checklist,
                "evidence": evidence_ok,
                "commands": command_ok,
                "symbols": symbol_ok,
            })
        classes.append({
            "key": cls.key,
            "title": cls.title,
            "scope": cls.scope,
            "required_files": {p: _path_exists(p) for p in cls.required_files},
            "gates": gates,
            "automated_gates": sum(1 for g in cls.gates if g.kind == "automated"),
            "maintainer_judgment_gates": sum(1 for g in cls.gates if g.kind == "maintainer-judgment"),
        })
        missing_paths.extend(p for p in cls.required_files if not _path_exists(p))

    total_gates = sum(len(cls.gates) for cls in policy())
    return {
        "schema_version": 1,
        "submission_classes": classes,
        "summary": {
            "class_count": len(classes),
            "gate_count": total_gates,
            "automated_gate_count": sum(c["automated_gates"] for c in classes),
            "maintainer_judgment_gate_count": sum(c["maintainer_judgment_gates"] for c in classes),
            "all_references_resolve": not (missing_paths or missing_commands or missing_symbols),
            "missing_paths": sorted(set(missing_paths)),
            "missing_commands": sorted(set(missing_commands)),
            "missing_symbols": sorted(set(missing_symbols)),
        },
    }


def render_policy_doc(audit: Dict[str, Any]) -> str:
    lines = [
        "# Third-party acceptance governance",
        "",
        "TensorGuard accepts outside contributions only when the review path is tied",
        "to a real verifier gate. This page is generated by",
        "`reproducibility/governance_acceptance.py`; the companion audit proves that",
        "each cited script, workflow, document, and symbol resolves in this repo.",
        "",
        "Gate kinds are deliberately explicit: **automated** means CI or a local",
        "command can enforce the requirement; **maintainer-judgment** means the",
        "requirement is a human review obligation backed by `GOVERNANCE.md`,",
        "`SECURITY.md`, or `MAINTAINERS.md`.",
        "",
    ]
    for cls in audit["submission_classes"]:
        lines.extend([f"## {cls['title']}", "", cls["scope"], ""])
        lines.extend(["### Required surfaces", ""])
        for path, ok in sorted(cls["required_files"].items()):
            status = "present" if ok else "missing"
            lines.append(f"- `{path}` — {status}")
        lines.extend(["", "### Maintainer checklist", ""])
        lines.append("| Gate | Kind | Checklist item | Evidence |")
        lines.append("| --- | --- | --- | --- |")
        for gate in cls["gates"]:
            evidence = ", ".join(f"`{p}`" for p in sorted(gate["evidence"]))
            lines.append(
                f"| {gate['name']} | {gate['kind']} | {gate['checklist']} | {evidence} |"
            )
        lines.append("")
    lines.extend([
        "## Acceptance rule",
        "",
        "A submission is mergeable only when every automated gate passes and the",
        "maintainer-judgment gates have an explicit reviewer sign-off in the PR.",
        "If a gate cannot be run in the contributor's environment, the PR must carry",
        "the exact skipped command and a maintainer-owned reproduction note.",
        "",
    ])
    return "\n".join(lines)


def render_audit_md(audit: Dict[str, Any]) -> str:
    summary = audit["summary"]
    lines = [
        "# Third-party acceptance governance audit",
        "",
        f"- Submission classes: **{summary['class_count']}**",
        f"- Gates: **{summary['gate_count']}** "
        f"({summary['automated_gate_count']} automated, "
        f"{summary['maintainer_judgment_gate_count']} maintainer-judgment)",
        f"- All references resolve: **{summary['all_references_resolve']}**",
        "",
        "| Class | Automated | Maintainer judgment | Required files present |",
        "| --- | ---: | ---: | --- |",
    ]
    for cls in audit["submission_classes"]:
        present = all(cls["required_files"].values())
        lines.append(
            f"| {cls['key']} | {cls['automated_gates']} | "
            f"{cls['maintainer_judgment_gates']} | {present} |"
        )
    lines.append("")
    if not summary["all_references_resolve"]:
        lines.extend(["## Missing references", ""])
        for key in ("missing_paths", "missing_commands", "missing_symbols"):
            for item in summary[key]:
                lines.append(f"- {key}: `{item}`")
        lines.append("")
    return "\n".join(lines)


def write_outputs() -> Dict[str, Any]:
    audit = build_audit()
    OUT_DOC.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    OUT_MD.write_text(render_audit_md(audit), encoding="utf-8")
    OUT_DOC.write_text(render_policy_doc(audit), encoding="utf-8")
    return audit


def _git_diff(paths: Iterable[Path]) -> str:
    rels = [str(path.relative_to(ROOT)) for path in paths]
    proc = subprocess.run(
        ["git", "--no-pager", "diff", "--", *rels],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return proc.stdout + proc.stderr


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", help="fail if outputs are stale")
    args = parser.parse_args(argv)

    before = _git_diff((OUT_DOC, OUT_JSON, OUT_MD)) if args.check else ""
    audit = write_outputs()
    if not audit["summary"]["all_references_resolve"]:
        print("governance acceptance audit failed: unresolved references", file=sys.stderr)
        return 1
    if args.check:
        after = _git_diff((OUT_DOC, OUT_JSON, OUT_MD))
        if before != after or after:
            print("governance acceptance artifacts are stale", file=sys.stderr)
            print(after, file=sys.stderr)
            return 1
    print("governance acceptance audit: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
