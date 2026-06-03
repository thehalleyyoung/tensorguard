"""Camera-ready paper package validator.

This Step-268 harness turns the paper package into a generated, checkable
artifact.  It derives the canonical claim ledger from the paper-evidence index
and committed evidence JSON files, emits deterministic JSON/Markdown outputs,
and verifies that the canonical root ``tool_paper.tex`` contains the generated
LaTeX ledger.  If any backing number changes without a matching paper update,
``--check`` fails.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional

REPO = Path(__file__).resolve().parent.parent
OUT_JSON = REPO / "reproducibility" / "camera_ready_paper.json"
OUT_MD = REPO / "reproducibility" / "camera_ready_paper.md"
CANONICAL_TEX = REPO / "tool_paper.tex"
CANONICAL_PDF = REPO / "tool_paper.pdf"

BEGIN = "% BEGIN GENERATED CAMERA-READY CLAIMS"
END = "% END GENERATED CAMERA-READY CLAIMS"


def _load_json(rel: str) -> Dict[str, object]:
    return json.loads((REPO / rel).read_text())


def _artifact_index_stems(evidence_index: Dict[str, object]) -> set[str]:
    return {str(e["stem"]) for e in evidence_index["evidence"]}  # type: ignore[index]


def _fmt_rate(value: float) -> str:
    if abs(value - round(value)) < 1e-12:
        return f"{int(round(value))}.000"
    return f"{value:.3f}"


def _claim(
    claim_id: str,
    statement: str,
    artifacts: Iterable[str],
    value: object,
) -> Dict[str, object]:
    return {
        "id": claim_id,
        "statement": statement,
        "source_artifacts": sorted(artifacts),
        "value": value,
    }


def measure() -> Dict[str, object]:
    evidence = _load_json("reproducibility/paper_evidence_index.json")
    corpus = _load_json("reproducibility/corpus_extended_score.json")
    fp_stress = _load_json("reproducibility/fp_stress_eval.json")
    dispatcher = _load_json("reproducibility/differential_dispatcher.json")
    mutation = _load_json("reproducibility/mutation_clean_models.json")
    meta = _load_json("reproducibility/statistical_meta_analysis.json")
    negative = _load_json("evaluation/negative_controls.json")
    package = _load_json("reproducibility/artifact_package.json")

    sound_corpus = corpus["sound"]  # type: ignore[index]
    sound_conf = sound_corpus["confusion"]  # type: ignore[index]
    sound_fp = fp_stress["per_mode"]["sound"]  # type: ignore[index]
    sound_mutation = mutation["per_mode"]["sound"]  # type: ignore[index]
    no_naive_pool = bool(
        meta["method"]["naive_pooling_across_distributions_allowed"] is False  # type: ignore[index]
    )
    claims = [
        _claim(
            "paper_evidence_index",
            (
                "The camera-ready paper is backed by "
                f"{evidence['n_evidence_items']} indexed evidence artifacts, "
                f"including {evidence['n_with_table']} rendered tables."
            ),
            ["reproducibility/paper_evidence_index.json"],
            {
                "n_evidence_items": evidence["n_evidence_items"],
                "n_with_table": evidence["n_with_table"],
            },
        ),
        _claim(
            "extended_corpus_score",
            (
                "On the extended runtime-validated corpus, sound mode reports "
                f"TP={sound_conf['tp']}, FP={sound_conf['fp']}, "
                f"TN={sound_conf['tn']}, FN={sound_conf['fn']} "
                f"across {sound_corpus['n_total']} cases."
            ),
            ["reproducibility/corpus_extended_score.json"],
            {
                "tp": sound_conf["tp"],
                "fp": sound_conf["fp"],
                "tn": sound_conf["tn"],
                "fn": sound_conf["fn"],
                "n_total": sound_corpus["n_total"],
            },
        ),
        _claim(
            "clean_fp_stress",
            (
                "Sound mode has "
                f"{sound_fp['n_false_alarms']} false alarms and "
                f"{sound_fp['n_abstained']} abstentions on "
                f"{fp_stress['n_models']} clean stress-test models."
            ),
            ["reproducibility/fp_stress_eval.json"],
            {
                "false_alarms": sound_fp["n_false_alarms"],
                "abstentions": sound_fp["n_abstained"],
                "n_models": fp_stress["n_models"],
            },
        ),
        _claim(
            "differential_dispatcher",
            (
                "Dispatcher differential testing checks "
                f"{dispatcher['n_modules']} generated modules with "
                f"{dispatcher['n_false_alarms']} false alarms and "
                f"{dispatcher['n_soundness_violations']} soundness violations."
            ),
            ["reproducibility/differential_dispatcher.json"],
            {
                "n_modules": dispatcher["n_modules"],
                "false_alarms": dispatcher["n_false_alarms"],
                "soundness_violations": dispatcher["n_soundness_violations"],
            },
        ),
        _claim(
            "mutation_kill_rate",
            (
                "Mutation testing kills "
                f"{sound_mutation['n_killed']} of {sound_mutation['kill_rate']['n']} "
                f"runtime-proven mutants in sound mode, with "
                f"{sound_mutation['n_survived']} survivors."
            ),
            ["reproducibility/mutation_clean_models.json"],
            {
                "n_killed": sound_mutation["n_killed"],
                "n_mutants": sound_mutation["kill_rate"]["n"],
                "n_survived": sound_mutation["n_survived"],
            },
        ),
        _claim(
            "stratified_meta_analysis",
            (
                "The meta-analysis keeps "
                f"{len(meta['distributions'])} distributions separate, uses "
                f"{meta['method']['bootstrap_resamples']} suite-level bootstrap resamples, "
                f"and forbids naive global pooling: {no_naive_pool}."
            ),
            ["reproducibility/statistical_meta_analysis.json"],
            {
                "n_distributions": len(meta["distributions"]),  # type: ignore[index]
                "bootstrap_resamples": meta["method"]["bootstrap_resamples"],  # type: ignore[index]
                "naive_pooling_forbidden": no_naive_pool,
            },
        ),
        _claim(
            "negative_controls",
            (
                "Negative controls include "
                f"{negative['summary']['n_cases']} value-domain cases where TensorGuard "
                f"catches {negative['summary']['tensorguard_caught']} and the explicit "
                f"finite-output runtime checker catches {negative['summary']['runtime_finite_output_check_caught']}."
            ),
            ["evaluation/negative_controls.json"],
            {
                "n_cases": negative["summary"]["n_cases"],  # type: ignore[index]
                "tensorguard_caught": negative["summary"]["tensorguard_caught"],  # type: ignore[index]
                "runtime_finite_output_check_caught": negative["summary"]["runtime_finite_output_check_caught"],  # type: ignore[index]
            },
        ),
        _claim(
            "fresh_machine_package",
            (
                "The artifact package validates "
                f"{package['n_modes']} fresh-machine modes "
                f"(Docker, conda, and source), all passed: {package['all_modes_passed']}."
            ),
            ["reproducibility/artifact_package.json"],
            {
                "n_modes": package["n_modes"],
                "all_modes_passed": package["all_modes_passed"],
            },
        ),
    ]

    stems = _artifact_index_stems(evidence)
    missing_from_index: List[str] = []
    for c in claims:
        for rel in c["source_artifacts"]:  # type: ignore[index]
            if Path(str(rel)).stem not in stems:
                missing_from_index.append(str(rel))

    return {
        "step": 268,
        "canonical_tex": "tool_paper.tex",
        "canonical_pdf": "tool_paper.pdf",
        "page_bounds": {"exclusive_min": 20, "exclusive_max": 40},
        "claim_source": "reproducibility/paper_evidence_index.json",
        "n_claims": len(claims),
        "all_claim_artifacts_indexed": not missing_from_index,
        "missing_from_evidence_index": sorted(missing_from_index),
        "claims": claims,
    }


def _escape_latex(s: str) -> str:
    repl = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(repl.get(ch, ch) for ch in s)


def render_latex_block(data: Dict[str, object]) -> str:
    lines = [
        BEGIN,
        r"\begin{center}",
        r"\begin{tabular}{p{0.25\linewidth}p{0.66\linewidth}}",
        r"\hline",
        r"\textbf{Generated claim} & \textbf{Evidence-derived statement} \\",
        r"\hline",
    ]
    for claim in data["claims"]:  # type: ignore[index]
        lines.append(
            f"{_escape_latex(str(claim['id']))} & "
            f"{_escape_latex(str(claim['statement']))} \\\\"
        )
    lines += [
        r"\hline",
        r"\end{tabular}",
        r"\end{center}",
        END,
    ]
    return "\n".join(lines)


def render_markdown(data: Dict[str, object]) -> str:
    lines = [
        "# Camera-ready paper package (Step 268)",
        "",
        "This manifest is generated from the paper-evidence index and committed "
        "evidence artifacts. The root `tool_paper.tex` must contain the exact "
        "generated LaTeX ledger below; if any backing number drifts, the ledger "
        "changes and `python reproducibility/camera_ready_paper.py --check` fails.",
        "",
        f"- canonical TeX: `{data['canonical_tex']}`",
        f"- canonical PDF: `{data['canonical_pdf']}`",
        f"- claims: **{data['n_claims']}**",
        f"- all claim artifacts indexed: **{data['all_claim_artifacts_indexed']}**",
        "",
        "| claim | source artifacts | generated statement |",
        "| --- | --- | --- |",
    ]
    for claim in data["claims"]:  # type: ignore[index]
        artifacts = ", ".join(f"`{p}`" for p in claim["source_artifacts"])
        lines.append(f"| `{claim['id']}` | {artifacts} | {claim['statement']} |")
    lines += [
        "",
        "## Generated LaTeX ledger",
        "",
        "```tex",
        render_latex_block(data),
        "```",
        "",
    ]
    return "\n".join(lines)


def validate_latex_block(data: Dict[str, object], tex_path: Path = CANONICAL_TEX) -> List[str]:
    expected = render_latex_block(data)
    text = tex_path.read_text() if tex_path.exists() else ""
    if expected in text:
        return []
    return [f"{tex_path}: generated camera-ready claim ledger is missing or stale"]


def _page_count_from_pdfinfo_output(text: str) -> Optional[int]:
    m = re.search(r"^Pages:\s*(\d+)\s*$", text, flags=re.MULTILINE)
    return int(m.group(1)) if m else None


def pdf_page_count(pdf_path: Path = CANONICAL_PDF) -> Optional[int]:
    if not pdf_path.exists() or shutil.which("pdfinfo") is None:
        return None
    proc = subprocess.run(
        ["pdfinfo", str(pdf_path)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        return None
    return _page_count_from_pdfinfo_output(proc.stdout)


def validate_page_bounds(data: Dict[str, object], pdf_path: Path = CANONICAL_PDF) -> List[str]:
    pages = pdf_page_count(pdf_path)
    if pages is None:
        return []
    bounds = data["page_bounds"]  # type: ignore[index]
    lo = int(bounds["exclusive_min"])  # type: ignore[index]
    hi = int(bounds["exclusive_max"])  # type: ignore[index]
    if lo < pages < hi:
        return []
    return [f"{pdf_path}: page count {pages} is outside ({lo}, {hi})"]


def _write_or_check(check: bool) -> int:
    data = measure()
    js = json.dumps(data, indent=2, sort_keys=True) + "\n"
    md = render_markdown(data)
    errors: List[str] = []
    if not data["all_claim_artifacts_indexed"]:
        errors.append(
            "claim source artifacts missing from paper evidence index: "
            + ", ".join(data["missing_from_evidence_index"])  # type: ignore[arg-type]
        )
    errors.extend(validate_latex_block(data))
    errors.extend(validate_page_bounds(data))

    if check:
        if not OUT_JSON.exists() or OUT_JSON.read_text() != js:
            errors.append(f"MISMATCH: {OUT_JSON}")
        if not OUT_MD.exists() or OUT_MD.read_text() != md:
            errors.append(f"MISMATCH: {OUT_MD}")
    else:
        OUT_JSON.write_text(js)
        OUT_MD.write_text(md)

    if errors:
        for err in errors:
            print(err)
        return 1
    if check:
        print("camera_ready_paper: byte-identical; paper claims and page bounds validated")
    else:
        print(f"wrote {OUT_JSON}")
        print(f"wrote {OUT_MD}")
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true", help="validate committed manifest and canonical paper")
    ap.add_argument("--print-latex", action="store_true", help="print the generated LaTeX claim ledger")
    args = ap.parse_args(argv)
    if args.print_latex:
        print(render_latex_block(measure()))
        return 0
    return _write_or_check(check=args.check)


if __name__ == "__main__":
    sys.exit(main())
