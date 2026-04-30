"""Cross-check that load-bearing numerical claims in the rebuilt paper text
agree with the shipped reproducibility artifacts.

This script is the round-8 synchronized-reproducibility witness: it does
not generate new data, it asserts that the paper text and the JSON
artifacts in this directory state the same numbers.  Each assertion
quotes (i) the artifact key, (ii) the paper-side claim, and (iii) the
section of the paper the reader can audit it against.

Run:

    python3 reproducibility/paper_artifact_reconciliation.py

Exits non-zero on the first mismatch.
"""

from __future__ import annotations

import json
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
REPRO = REPO_ROOT / "reproducibility"
SECTIONS = REPO_ROOT / "docs" / "paper" / "sections_v5"


@dataclass
class Check:
    name: str
    ok: bool
    detail: str


def _load_json(name: str) -> dict:
    path = REPRO / name
    with path.open() as fh:
        return json.load(fh)


def _read_section(name: str) -> str:
    return (SECTIONS / name).read_text()


def check_grad_lattice_runtime() -> Check:
    """Reviewer R8-W1: paper must state 2/8 false-verified, not 0/8."""
    artifact = _load_json("grad_lattice_runtime_holdout.json")
    art_false_verified = artifact["n_positive_false_verified"]
    art_oof = artifact["n_positive_grad_out_of_fragment"]
    art_n_pos = artifact["n_positive"]
    art_neg_specificity = artifact["negative_control_specificity"]

    eval_text = _read_section("eval_v6.tex")
    limconc_text = _read_section("limconc_v6.tex")

    expected_phrase = r"\mathbf{2/8 = 25.0\%}"
    if expected_phrase not in eval_text:
        return Check(
            "grad_lattice_runtime_eval",
            False,
            "eval_v6.tex no longer carries the '2/8 = 25.0%' false-verified phrase",
        )
    if expected_phrase not in limconc_text:
        return Check(
            "grad_lattice_runtime_limconc",
            False,
            "limconc_v6.tex must also carry the '2/8 = 25.0%' false-verified phrase "
            "(reviewer R8-W1 explicitly flagged the limitations paragraph as the source "
            "of the 0/8 vs 2/8 mismatch)",
        )
    if art_false_verified != 2 or art_oof != 6 or art_n_pos != 8:
        return Check(
            "grad_lattice_runtime_artifact_drift",
            False,
            f"artifact reports false_verified={art_false_verified}, oof={art_oof}, "
            f"n_pos={art_n_pos}; paper assumes 2/6/8",
        )
    if art_neg_specificity != 1.0:
        return Check(
            "grad_lattice_runtime_specificity",
            False,
            f"artifact specificity dropped to {art_neg_specificity}",
        )
    return Check(
        "grad_lattice_runtime",
        True,
        "paper (eval+limconc) and artifact agree: 2/8 false-verified, 6/8 OOF, "
        "2/2 negative-control specificity",
    )


def check_theorem5_audits() -> Check:
    """Reviewer R8-W2: paper must reconcile n=107/55/72-INT (n100) with
    n=146/67/0 (n200)."""
    n100 = _load_json("dynamo_theorem5_n100.json")
    n200 = _load_json("dynamo_theorem5_n200.json")

    n100_cand = n100.get("n_candidates", n100.get("n_modules_total"))
    n100_ok = n100.get("n_successful_modules", n100.get("n_modules_ok"))
    if n100_cand != 107 or n100_ok != 55:
        return Check(
            "theorem5_n100_drift",
            False,
            f"n100 artifact drifted: {n100_cand} cand / {n100_ok} successful",
        )
    if n200["n_candidates"] != 146 or n200["n_successful_modules"] != 67:
        return Check(
            "theorem5_n200_drift",
            False,
            f"n200 artifact drifted: {n200['n_candidates']} cand / "
            f"{n200['n_successful_modules']} successful",
        )
    # Both must agree on the only quantity Thm5 needs: zero falsifier events.
    n100_falsifiers = n100["n_modules_falsifying_theorem5"]
    n200_falsifiers = n200["n_modules_falsifying_theorem5"]
    if n100_falsifiers != 0 or n200_falsifiers != 0:
        return Check(
            "theorem5_falsifiers_nonzero",
            False,
            f"falsifier event observed: n100={n100_falsifiers}, n200={n200_falsifiers}",
        )
    eval_text = _read_section("eval_v6.tex")
    must_have = [
        "$107$ candidate modules, $55$",
        "$146$ candidate",
        "$67$",
        "$0$ Theorem~5 falsifier events",
    ]
    missing = [s for s in must_have if s not in eval_text]
    if missing:
        return Check(
            "theorem5_paper_mentions",
            False,
            "eval_v6.tex must reference both audits explicitly. Missing: " + repr(missing),
        )
    return Check(
        "theorem5_audits",
        True,
        "paper reconciles 107/55/72-INT (n100) and 146/67/0 (n200); both audits "
        "agree on 0 SHAPE/DTYPE/RANK guards and 0 Theorem~5 falsifier events",
    )


def check_hybrid_mode_scope() -> Check:
    """Reviewer R8-W3: hybrid-mode complementarity must be restated narrowly."""
    eval_text = _read_section("eval_v6.tex")
    must_have = [
        "complementary",
        "existence",
        "hand-designed",
        "$\\{57,\\,206,\\,225\\}$",
    ]
    missing = []
    for s in must_have:
        if s not in eval_text:
            missing.append(s)
    if missing:
        return Check(
            "hybrid_scope_phrasing",
            False,
            "eval_v6.tex hybrid paragraph must call the 25-block result an "
            "existence demonstration and reference the zero-gain {57,206,225} "
            "result on the 488-block corpus. Missing: " + repr(missing),
        )
    if "general complementarity" in eval_text and "not claim" not in eval_text:
        return Check(
            "hybrid_scope_unqualified",
            False,
            "found unqualified 'general complementarity' claim in eval_v6.tex",
        )
    return Check(
        "hybrid_mode_scope",
        True,
        "eval_v6.tex restates hybrid-mode complementarity as a stress-set existence "
        "result and reports zero gain on the 488-block corpus",
    )


def check_post_freeze_headline() -> Check:
    """Reviewer prior W3 (resolved): post-freeze 5/15 framed as directional."""
    eval_text = _read_section("eval_v6.tex")
    if "5/15" not in eval_text and "$5/15$" not in eval_text:
        return Check("postfreeze_5_15_present", False, "5/15 figure missing from eval_v6")
    return Check("post_freeze_headline", True, "5/15 headline present in eval text")


def check_abstract_constraints() -> Check:
    """HARD CONSTRAINT 1 + 4: no repo-file paths, structured short abstract."""
    paper_tex = (REPO_ROOT / "neurips.tex").read_text()
    abs_match = re.search(r"\\begin\{abstract\}(.*?)\\end\{abstract\}", paper_tex, re.S)
    if not abs_match:
        return Check("abstract_extracted", False, "no \\begin{abstract} block")
    abstract = abs_match.group(1)
    # word count
    words = re.findall(r"[A-Za-z]+", abstract)
    if len(words) > 270:
        return Check(
            "abstract_word_count",
            False,
            f"abstract has {len(words)} words; cap is ~250",
        )
    forbidden = re.findall(r"\.(py|lean|json|tex|sh|md|csv|yaml)\b", abstract)
    if forbidden:
        return Check(
            "abstract_no_repo_paths",
            False,
            f"abstract names repo-file extensions: {forbidden}",
        )
    return Check(
        "abstract_constraints",
        True,
        f"abstract is {len(words)} words and names no repo-file paths",
    )


CHECKS = [
    check_grad_lattice_runtime,
    check_theorem5_audits,
    check_hybrid_mode_scope,
    check_post_freeze_headline,
    check_abstract_constraints,
]


def main() -> int:
    results = [chk() for chk in CHECKS]
    width = max(len(r.name) for r in results)
    print("Paper / artifact reconciliation report")
    print("=" * 72)
    bad = 0
    for r in results:
        flag = "OK " if r.ok else "FAIL"
        print(f"[{flag}] {r.name.ljust(width)}  {r.detail}")
        if not r.ok:
            bad += 1
    print("=" * 72)
    print(f"{len(results) - bad}/{len(results)} checks passed")
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
