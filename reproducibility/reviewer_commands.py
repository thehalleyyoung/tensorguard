"""Reviewer-friendly one-command reproduction manifest.

This module publishes the commands a reviewer needs to regenerate the paper's
main tables/figures without reverse-engineering the Makefile.  The generated
JSON/Markdown manifest is deterministic and self-checking: every listed command
is validated against real repository files, Makefile targets, and committed
outputs.  The shell wrappers in ``scripts/`` delegate here so the documented
entrypoints and the machine-readable manifest cannot drift apart.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Dict, Iterable, List

REPO = Path(__file__).resolve().parent.parent
OUT_JSON = REPO / "reproducibility" / "reviewer_commands.json"
OUT_MD = REPO / "reproducibility" / "reviewer_commands.md"


MAIN_COMMAND = ["python", "reproducibility/reproduce_all.py", "--check"]
FAST_COMMAND = ["python", "reproducibility/reviewer_commands.py", "--dry-run"]


RESULTS: List[Dict[str, object]] = [
    {
        "id": "headline_60bug",
        "claim": "60-bug headline Refuted-Proof figure and README ratios",
        "command": "python reproducibility/reproduce_headline_60bug.py",
        "outputs": ["reproducibility/reproduce_headline_60bug.json"],
    },
    {
        "id": "precision_recall",
        "claim": "baseline precision/recall matrices and NA handling",
        "command": "make precision-recall",
        "outputs": ["evaluation/confusion_matrices.json", "evaluation/confusion_matrices.md"],
    },
    {
        "id": "significance",
        "claim": "McNemar, Holm, and paired-bootstrap significance tests",
        "command": "python evaluation/significance.py",
        "outputs": ["evaluation/significance.json", "evaluation/significance.md"],
    },
    {
        "id": "sound_mode_fp",
        "claim": "0% false-positive hunt on executing clean models",
        "command": "make sound-fp",
        "outputs": ["evaluation/sound_mode_fp.json", "evaluation/sound_mode_fp.md"],
    },
    {
        "id": "hard_recall",
        "claim": "latent-bug recall advantage over the strongest runtime baseline",
        "command": "make hard-recall",
        "outputs": ["evaluation/hard_recall.json", "evaluation/hard_recall.md"],
    },
    {
        "id": "differential_fuzz",
        "claim": "random valid-module false-positive fuzzing",
        "command": "make diff-fuzz",
        "outputs": ["evaluation/diff_fuzz.json", "evaluation/diff_fuzz.md"],
    },
    {
        "id": "negative_fuzz",
        "claim": "fault-injection false-negative fuzzing",
        "command": "make neg-fuzz",
        "outputs": ["evaluation/neg_fuzz.json", "evaluation/neg_fuzz.md"],
    },
    {
        "id": "triage_regressions",
        "claim": "50 minimized bug reproducers and clean siblings",
        "command": "make triage",
        "outputs": ["evaluation/triage_regressions.json", "evaluation/triage_regressions.md"],
    },
    {
        "id": "operator_coverage",
        "claim": "public operator coverage matrix",
        "command": "make operator-coverage",
        "outputs": ["evaluation/operator_coverage.json", "evaluation/operator_coverage.md"],
    },
    {
        "id": "real_model_operator_coverage",
        "claim": "torchvision/timm/HuggingFace frequency-weighted operator coverage",
        "command": "make real-model-operator-coverage",
        "outputs": ["evaluation/real_model_operator_coverage.json", "evaluation/real_model_operator_coverage.md"],
    },
    {
        "id": "deployment_gallery",
        "claim": "real-model deployment/export gallery and gates",
        "command": "make deployment-gallery",
        "outputs": ["evaluation/deployment_gallery.json", "evaluation/deployment_gallery.md"],
    },
    {
        "id": "pareto_curves",
        "claim": "hardware-normalized cost/latency Pareto curves",
        "command": "make pareto-curves",
        "outputs": ["evaluation/pareto_curves.json", "evaluation/pareto_curves.md"],
    },
    {
        "id": "paper_evidence",
        "claim": "single paper-evidence index of every regenerable table/figure",
        "command": "make paper-evidence",
        "outputs": ["reproducibility/paper_evidence_index.json", "reproducibility/paper_evidence_index.md"],
    },
    {
        "id": "artifact_index",
        "claim": "tamper-evident SHA-256 ledger of generated artifacts",
        "command": "make artifact-index",
        "outputs": ["reproducibility/artifact_index.json", "reproducibility/artifact_index.md"],
    },
    {
        "id": "camera_ready_paper",
        "claim": "camera-ready paper claim ledger generated from indexed evidence",
        "command": "make camera-ready-paper",
        "outputs": ["reproducibility/camera_ready_paper.json", "reproducibility/camera_ready_paper.md", "tool_paper.pdf"],
    },
]


def _make_targets() -> set[str]:
    text = (REPO / "Makefile").read_text()
    return set(re.findall(r"^([A-Za-z0-9_.-]+):", text, flags=re.MULTILINE))


def _command_exists(command: str, make_targets: set[str]) -> bool:
    parts = command.split()
    if not parts:
        return False
    if parts[0] == "make":
        return len(parts) == 2 and parts[1] in make_targets
    if parts[0] == "python" and len(parts) >= 2:
        script = parts[1]
        return script.endswith(".py") and (REPO / script).exists()
    return False


def _script_hash(path: Path) -> str:
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()


def measure() -> Dict[str, object]:
    make_targets = _make_targets()
    entries = []
    for row in RESULTS:
        outputs = [
            {
                "path": p,
                "present": (REPO / p).exists(),
            }
            for p in row["outputs"]  # type: ignore[index]
        ]
        command = str(row["command"])
        entries.append(
            {
                "id": row["id"],
                "claim": row["claim"],
                "command": command,
                "command_resolves": _command_exists(command, make_targets),
                "outputs": outputs,
                "all_outputs_present": all(o["present"] for o in outputs),
            }
        )

    scripts = [
        "scripts/reproduce_main_results.sh",
        "scripts/check_main_results.sh",
    ]
    return {
        "step": 126,
        "entrypoints": {
            "full_reproduction": "bash scripts/reproduce_main_results.sh",
            "determinism_check": "bash scripts/check_main_results.sh",
            "dry_run": "python reproducibility/reviewer_commands.py --dry-run",
        },
        "authoritative_full_command": " ".join(MAIN_COMMAND),
        "fast_listing_command": " ".join(FAST_COMMAND),
        "n_main_results": len(entries),
        "n_commands_resolving": sum(1 for e in entries if e["command_resolves"]),
        "n_results_with_outputs": sum(1 for e in entries if e["all_outputs_present"]),
        "all_commands_resolve": all(e["command_resolves"] for e in entries),
        "all_outputs_present": all(e["all_outputs_present"] for e in entries),
        "scripts": [
            {
                "path": s,
                "present": (REPO / s).exists(),
                "sha256": _script_hash(REPO / s) if (REPO / s).exists() else None,
            }
            for s in scripts
        ],
        "results": entries,
    }


def render_markdown(d: Dict[str, object]) -> str:
    lines = [
        "# Reviewer reproduction commands (Step 126)",
        "",
        "Two shell entrypoints expose the paper's main evidence without requiring a "
        "reviewer to inspect the Makefile:",
        "",
        f"- full reproduction + byte check: `{d['entrypoints']['full_reproduction']}`",
        f"- check-only wrapper: `{d['entrypoints']['determinism_check']}`",
        f"- command preview: `{d['entrypoints']['dry_run']}`",
        "",
        f"The authoritative full command is `{d['authoritative_full_command']}`. "
        f"The manifest covers **{d['n_main_results']}** main result groups; "
        f"commands resolve: **{d['all_commands_resolve']}**; outputs present: "
        f"**{d['all_outputs_present']}**.",
        "",
        "| result | claim | command | outputs present |",
        "| --- | --- | --- | --- |",
    ]
    for e in d["results"]:  # type: ignore[index]
        outs = ", ".join(o["path"] for o in e["outputs"])
        lines.append(
            f"| {e['id']} | {e['claim']} | `{e['command']}` | "
            f"{e['all_outputs_present']} ({outs}) |"
        )
    lines += [
        "",
        "## Wrapper scripts",
        "",
        "| script | present | sha256 |",
        "| --- | --- | --- |",
    ]
    for s in d["scripts"]:  # type: ignore[index]
        lines.append(f"| `{s['path']}` | {s['present']} | `{s['sha256']}` |")
    lines.append("")
    return "\n".join(lines)


def _write_or_check(check: bool) -> int:
    d = measure()
    js = json.dumps(d, indent=2, sort_keys=True) + "\n"
    md = render_markdown(d)
    if check:
        ok = True
        if not OUT_JSON.exists() or OUT_JSON.read_text() != js:
            print(f"MISMATCH: {OUT_JSON}")
            ok = False
        if not OUT_MD.exists() or OUT_MD.read_text() != md:
            print(f"MISMATCH: {OUT_MD}")
            ok = False
        if ok:
            print("reviewer_commands: byte-identical")
        return 0 if ok else 1
    OUT_JSON.write_text(js)
    OUT_MD.write_text(md)
    print(f"wrote {OUT_JSON}")
    print(f"wrote {OUT_MD}")
    return 0


def _dry_run() -> int:
    d = measure()
    print("Reviewer-facing TensorGuard reproduction commands:")
    for label, cmd in d["entrypoints"].items():  # type: ignore[union-attr]
        print(f"  {label}: {cmd}")
    print("\nMain result groups:")
    for e in d["results"]:  # type: ignore[index]
        print(f"  - {e['id']}: {e['command']}")
    return 0 if d["all_commands_resolve"] and d["all_outputs_present"] else 1


def _run_command(argv: Iterable[str]) -> int:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO) + os.pathsep + env.get("PYTHONPATH", "")
    proc = subprocess.run(list(argv), cwd=REPO, env=env)
    return proc.returncode


def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true", help="diff generated manifest against committed bytes")
    ap.add_argument("--dry-run", action="store_true", help="print reviewer commands without running them")
    ap.add_argument("--run", choices=["main", "fast"], help="execute the reviewer-facing reproduction command")
    args = ap.parse_args(argv)

    if args.dry_run:
        return _dry_run()
    if args.run == "main":
        return _run_command(MAIN_COMMAND)
    if args.run == "fast":
        return _run_command(FAST_COMMAND)
    return _write_or_check(check=args.check)


if __name__ == "__main__":
    sys.exit(main())
